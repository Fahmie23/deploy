#!/usr/bin/env python3
"""
AI Singing Judge API — MERT edition.

Pipeline:
    upload -> convert to WAV -> MERT scorer -> Gemini judge -> LLM critique
    -> weighted fusion (model vs Gemini)

Handcrafted features are no longer used; the deep model (MERT + Ridge per trait)
replaces them entirely.
"""

import os
import fcntl

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --- GPU worker assignment ---
GPU_COUNTER_FILE = "/tmp/singing_judge_gpu_counter"
NUM_GPUS = 1


def _claim_gpu() -> int:
    with open(GPU_COUNTER_FILE, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        content = f.read().strip()
        gpu_id = int(content) if content else 0
        next_id = (gpu_id + 1) % NUM_GPUS
        f.seek(0)
        f.truncate()
        f.write(str(next_id))
        fcntl.flock(f, fcntl.LOCK_UN)
    return gpu_id


WORKER_GPU = _claim_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(WORKER_GPU)
print(f"[GPU] Worker PID {os.getpid()} assigned to GPU {WORKER_GPU}")

GPU_VRAM_LIMIT_MB = 2048
try:
    import torch
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory
        fraction = min(GPU_VRAM_LIMIT_MB * 1024**2 / total_mem, 1.0)
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        print(f"[GPU] PyTorch VRAM limited to {GPU_VRAM_LIMIT_MB} MB ({fraction:.2%})")
except Exception as e:
    print(f"[GPU] Could not set PyTorch VRAM limit: {e}")

import asyncio
import shutil
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import librosa
from fastapi import FastAPI, File, HTTPException, Request, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import config
from mert_scorer import predict_mert, warmup as mert_warmup
from llm_judge import SingingJudge
from gemini_judge import GeminiSingingJudge, SCORE_KEYS

# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------
MODEL_WEIGHT = 0.30
GEMINI_WEIGHT = 0.70

ALLOWED_EXTS = {".wav", ".m4a", ".mp3", ".aac", ".ogg", ".flac", ".webm", ".mp4"}
MAX_AUDIO_DURATION_SECS = 8 * 60

# MERT training used capitalized trait names; Gemini / downstream use these:
MERT_TO_GEMINI = {
    "Pitch":   "pitch",
    "Rhythm":  "rhythm",
    "Vibrato": "vibrato",
    "Breath":  "breath_control",
    "Timbre":  "timbre",
}

_gpu_semaphore = asyncio.Semaphore(1)

# ----------------------------------------------------------------
# Auth & Rate limiting
# ----------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(_api_key_header)):
    if not config.API_KEY or api_key != config.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")


# ----------------------------------------------------------------
# Audio utilities
# ----------------------------------------------------------------
def convert_to_wav(input_path: str, sr: int = 16000) -> str:
    """Convert any audio to mono PCM WAV at `sr` Hz using ffmpeg."""
    in_p = Path(input_path)
    out_p = in_p.with_suffix(".wav")
    cmd = [
        "ffmpeg", "-y", "-i", str(in_p),
        "-ac", "1", "-ar", str(sr), "-sample_fmt", "s16",
        str(out_p),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Install with: sudo apt-get install -y ffmpeg")
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="ignore")[-2000:]
        raise RuntimeError(f"ffmpeg conversion failed:\n{err}")
    return str(out_p)


# ----------------------------------------------------------------
# Core evaluation
# ----------------------------------------------------------------
def predict_scores(wav_path: str) -> Dict[str, Any]:
    """Run MERT and map to Gemini-compatible score keys."""
    mert_raw = predict_mert(wav_path)

    scores: Dict[str, Any] = {
        gkey: mert_raw[mkey] for mkey, gkey in MERT_TO_GEMINI.items()
    }
    scores["overall_performance"] = round(
        sum(scores[k] for k in ["pitch", "rhythm", "vibrato", "breath_control", "timbre"]) / 5,
        2,
    )
    scores["_metadata"] = {
        "model_name": "MERT-v1-95M + Ridge(alpha=20) per trait",
        "scale": "0-100",
        "filename": os.path.basename(wav_path),
    }
    return scores


async def run_full_evaluation(audio_path: str, use_judge: bool = True) -> Dict[str, Any]:
    """MERT scores + Gemini evaluation + LLM critique, fused at configured weights.

    Parallelism:
        - Gemini starts immediately (needs only the audio file).
        - MERT runs under the GPU semaphore.
        - LLM judge starts as soon as MERT finishes (may overlap with Gemini).
        - Fusion waits for both Gemini and LLM judge.
    """
    total_start = time.time()
    print(f"--- Processing: {os.path.basename(audio_path)} ---")

    # --- Gemini: fire immediately, runs concurrently with MERT ---
    async def _run_gemini() -> Dict[str, Any]:
        t0 = time.time()
        try:
            gemini_judge = GeminiSingingJudge()
            result = await asyncio.to_thread(gemini_judge.evaluate, audio_path)
            print(f"[Timer] Gemini judge: {time.time() - t0:.2f}s")
            return result
        except Exception as e:
            print(f"[Gemini Error] {e}")
            return {"error": str(e)}
    gemini_task = asyncio.create_task(_run_gemini())

    # --- MERT: GPU-bound, semaphore wraps only this block ---
    t0 = time.time()
    async with _gpu_semaphore:
        predictions = await asyncio.to_thread(predict_scores, audio_path)
    print(f"[Timer] MERT prediction: {time.time() - t0:.2f}s")

    # --- LLM judge: starts right after MERT, may overlap with Gemini ---
    if use_judge:
        async def _run_llm_judge() -> Dict[str, Any]:
            t0 = time.time()
            try:
                judge = SingingJudge()
                result = await asyncio.to_thread(judge.evaluate, {"predictions": predictions})
                print(f"[Timer] LLM judge: {time.time() - t0:.2f}s")
                return result
            except Exception as e:
                print(f"[Judge Error] {e}")
                r = {key: predictions.get(key, 0) for key in SCORE_KEYS}
                r["criticism"] = f"Judge failed: {str(e)}"
                r["advice"] = ""
                return r

        llm_task = asyncio.create_task(_run_llm_judge())
        gemini_result, result = await asyncio.gather(gemini_task, llm_task)
    else:
        gemini_result = await gemini_task
        result = {key: predictions.get(key, 0) for key in SCORE_KEYS}

    # --- Weighted fusion ---
    gemini_scores: Dict[str, Any] = {}
    if "error" not in gemini_result:
        for key in SCORE_KEYS:
            gemini_scores[key] = gemini_result.get(key, 0)
    else:
        print(f"[Gemini Warning] {gemini_result.get('error')}")

    if gemini_scores:
        for key in SCORE_KEYS:
            model_score = result.get(key, 0)
            gemini_score = gemini_scores.get(key, 0)
            result[f"weighted_{key}"] = round(
                model_score * MODEL_WEIGHT + gemini_score * GEMINI_WEIGHT, 2
            )
            result[f"gemini_{key}"] = gemini_score
        result["gemini_criticism"] = gemini_result.get("criticism", "")
        result["gemini_advice"] = gemini_result.get("advice", "")
    else:
        for key in SCORE_KEYS:
            result[f"weighted_{key}"] = result.get(key, 0)
            result[f"gemini_{key}"] = None
        result["gemini_criticism"] = None
        result["gemini_advice"] = None

    result["_metadata"] = predictions.get("_metadata", {})
    result["_metadata"]["model_weight"] = MODEL_WEIGHT
    result["_metadata"]["gemini_weight"] = GEMINI_WEIGHT
    result["_metadata"]["gemini_available"] = bool(gemini_scores)

    print(f"[Timer] Total pipeline: {time.time() - total_start:.2f}s")
    return result


# ----------------------------------------------------------------
# FastAPI
# ----------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        mert_warmup()
        torch.cuda.empty_cache()
        print("[Startup] MERT model + artifact warmed up.")
    except Exception as e:
        print(f"[Startup] MERT warmup failed: {e}")
    yield


app = FastAPI(title="AI Singing Judge API (MERT)", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/stats")
async def stats_endpoint():
    """Return GPU memory usage from within the server process."""
    if not torch.cuda.is_available():
        return {"gpu": "unavailable"}
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_name": props.name,
        "allocated_mb": round(torch.cuda.memory_allocated(0) / 1024**2, 2),
        "reserved_mb": round(torch.cuda.memory_reserved(0) / 1024**2, 2),
        "total_mb": round(props.total_memory / 1024**2, 2),
        "free_mb": round((props.total_memory - torch.cuda.memory_reserved(0)) / 1024**2, 2),
    }


@app.post("/judge", dependencies=[Security(verify_api_key)])
@limiter.limit(config.RATE_LIMIT)
async def judge_singing_endpoint(request: Request, file: UploadFile = File(...)):
    """Upload audio (wav/m4a/mp3/etc), run MERT + Gemini + LLM critique."""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {sorted(ALLOWED_EXTS)}",
        )

    temp_input = f"/tmp/temp_upload_{uuid.uuid4().hex[:8]}{ext}"
    temp_wav: Optional[str] = None

    try:
        with open(temp_input, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if ext != ".wav":
            temp_wav = convert_to_wav(temp_input, sr=16000)
            audio_for_pipeline = temp_wav
        else:
            audio_for_pipeline = temp_input

        duration = librosa.get_duration(path=audio_for_pipeline)
        if duration > MAX_AUDIO_DURATION_SECS:
            raise HTTPException(
                status_code=400,
                detail=f"Audio too long ({duration:.0f}s). Max {MAX_AUDIO_DURATION_SECS}s.",
            )

        result = await run_full_evaluation(audio_for_pipeline, True)
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in [temp_input, temp_wav]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
