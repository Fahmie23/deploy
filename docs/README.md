# AI Singing Judge API (MERT edition)

FastAPI service that scores a vocal performance using MERT embeddings + per-trait
Ridge regressors, fused with a Gemini audio judge and accompanied by a Groq LLM
critique.

## Layout

```
deploy/
├── main.py                              # FastAPI app (entry point)
├── mert_scorer.py                       # MERT feature + Ridge inference
├── llm_judge.py                         # Groq critique
├── gemini_judge.py                      # Gemini audio judge
├── config.py                            # Env loader
├── requirements.txt
├── .env.example
└── ml_training/
    └── mert_jenny_separate_models.pkl   # Trained artifact
```

## Setup

```bash
# 1. Create virtualenv
python -m venv venv
source venv/bin/activate          # Linux / macOS
# .\venv\Scripts\activate         # Windows

# 2. Install deps
pip install -r requirements.txt

# 3. (Linux) Install ffmpeg
sudo apt-get install -y ffmpeg
# (Windows) winget install ffmpeg  — or download from ffmpeg.org

# 4. Configure secrets
cp .env.example .env
# edit .env with your GROQ_API_KEY and GEMINI_API_KEY
```

## Run

### Development (single worker)
```bash
python main.py
```

### Production (multi-worker, tune for your GPU)
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
```

Each worker loads its own MERT (~3 GB VRAM). On a 15 GB T4, keep workers ≤ 2.

## Usage

```bash
curl -F "file=@song.wav" http://localhost:8000/judge
```

Supported upload formats: wav, m4a, mp3, aac, ogg, flac, webm, mp4
Max duration: 8 minutes.

### Response shape

```json
{
  "pitch": 78.1, "rhythm": 72.4, "vibrato": 65.0,
  "breath_control": 70.2, "timbre": 74.8, "overall_performance": 72.1,

  "gemini_pitch": 80, "gemini_rhythm": 75, ...,
  "weighted_pitch": 79.4, "weighted_rhythm": 74.2, ...,

  "criticism": "The singer shows strong pitch control ...",
  "advice": "Focus on consistent airflow ...",

  "_metadata": {
    "model_name": "MERT-v1-95M + Ridge(alpha=20) per trait",
    "scale": "0-100",
    "filename": "song.wav",
    "model_weight": 0.30,
    "gemini_weight": 0.70,
    "gemini_available": true
  }
}
```

## Notes

- Training used **no Demucs** and **no VAD**. Do not pre-separate vocals before
  `/judge` — feed the raw mix for consistent inference.
- The first request is slow (~3 s) because MERT is loaded lazily. A startup
  hook pre-warms the model to avoid this.
- To switch to the larger MERT-v1-330M, set `MERT_MODEL_SIZE = "big"` in
  `config.py` — but you must retrain the artifact on the same backbone.

## Windows

`main.py` uses `fcntl` for GPU-worker indexing, which is Unix-only. For Windows,
remove the `_claim_gpu()` block and hard-code `WORKER_GPU = 0`. Or run under
**WSL2** with CUDA-on-WSL for zero code changes.
