"""
MERT-based singing scorer.

Loads the trained artifact from ml_training/ and predicts per-trait scores
(Pitch, Rhythm, Vibrato, Breath, Timbre) from a WAV file, following the
exact training-time preprocessing:

    - 24 kHz mono
    - 6 s chunks (min 3 s), no overlap
    - MERT last-4-layer fusion (mean across layers) + mean/std time pooling
    - StandardScaler (+ optional PCA) per training artifact
    - Ridge predict per target
    - Trimmed-mean aggregation (10% each tail) chunk -> song
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor

MERT_MODEL_NAME = "m-a-p/MERT-v1-95M"
MERT_SR = 24000
CHUNK_SECONDS = 6.0
MIN_CHUNK_SECONDS = 3.0
MERT_BATCH_SIZE = 8
TRIMMED_MEAN_PROPORTION = 0.1

ARTIFACT_PATH = Path(__file__).parent / "ml_training" / "mert_jenny_separate_models.pkl"

_device = "cuda" if torch.cuda.is_available() else "cpu"
_mert_model = None
_mert_processor = None
_artifact: Optional[dict] = None


def _load() -> tuple:
    global _mert_model, _mert_processor, _artifact
    if _artifact is None:
        if not ARTIFACT_PATH.exists():
            raise FileNotFoundError(f"MERT artifact not found at {ARTIFACT_PATH}")
        with open(ARTIFACT_PATH, "rb") as f:
            _artifact = pickle.load(f)
    if _mert_model is None:
        _mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(MERT_MODEL_NAME)
        _mert_model = (
            AutoModel.from_pretrained(MERT_MODEL_NAME, trust_remote_code=True)
            .to(_device)
            .eval()
        )
    return _mert_model, _mert_processor, _artifact


def _chunk(audio: np.ndarray) -> List[np.ndarray]:
    n = int(CHUNK_SECONDS * MERT_SR)
    m = int(MIN_CHUNK_SECONDS * MERT_SR)
    chunks: List[np.ndarray] = []
    for start in range(0, len(audio), n):
        c = audio[start:start + n]
        if len(c) >= m:
            chunks.append(c)
    return chunks


def _embed(chunks: List[np.ndarray], model, processor) -> np.ndarray:
    embs: List[np.ndarray] = []
    for i in range(0, len(chunks), MERT_BATCH_SIZE):
        batch = chunks[i:i + MERT_BATCH_SIZE]
        inputs = processor(
            batch, sampling_rate=MERT_SR, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
            fused = torch.stack(out.hidden_states[-4:], dim=0).mean(dim=0)  # (B, T, D)

            lens = [len(c) for c in batch]
            mx = max(lens)
            T = fused.shape[1]
            if any(l < mx for l in lens):
                valid = [max(1, int(round(l / mx * T))) for l in lens]
                mask = torch.zeros(len(batch), T, 1, device=_device)
                for bi, v in enumerate(valid):
                    mask[bi, :v, :] = 1.0
                lengths = mask.sum(dim=1).clamp(min=1)
                mean_pool = (fused * mask).sum(dim=1) / lengths
                std_pool = torch.sqrt(
                    ((fused - mean_pool.unsqueeze(1)) ** 2 * mask).sum(dim=1)
                    / lengths.clamp(min=2)
                )
            else:
                mean_pool = fused.mean(dim=1)
                std_pool = fused.std(dim=1)

            emb = torch.cat([mean_pool, std_pool], dim=1).cpu().numpy().astype(np.float32)
        embs.extend(emb)
    return np.stack(embs)


def _trimmed_mean(values: np.ndarray) -> float:
    if len(values) < 3:
        return float(values.mean())
    k = int(np.floor(len(values) * TRIMMED_MEAN_PROPORTION))
    if k <= 0 or 2 * k >= len(values):
        return float(values.mean())
    return float(np.sort(values)[k:len(values) - k].mean())


def predict_mert(wav_path: str) -> Dict[str, float]:
    """Return per-trait song-level scores: {Pitch, Rhythm, Vibrato, Breath, Timbre}."""
    model, processor, art = _load()

    y, _ = librosa.load(wav_path, sr=MERT_SR, mono=True)
    chunks = _chunk(y)
    if not chunks:
        return {t: 0.0 for t in art["target_columns"]}

    X = _embed(chunks, model, processor)
    X = art["scaler"].transform(X)
    if art.get("pca") is not None:
        X = art["pca"].transform(X)

    scores: Dict[str, float] = {}
    for target, mdl in art["models_by_target"].items():
        preds = mdl.predict(X)
        scores[target] = round(_trimmed_mean(np.asarray(preds)), 2)
    return scores


def warmup() -> None:
    """Pre-load MERT + artifact so first request isn't slow."""
    _load()
