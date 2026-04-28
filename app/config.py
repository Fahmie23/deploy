import os
from dotenv import load_dotenv

load_dotenv()

# Fetch variables
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://vllm:8007/v1")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

API_KEY = os.getenv("API_KEY")
RATE_LIMIT = os.getenv("RATE_LIMIT", "10/minute")

if not GEMINI_API_KEY:
    import warnings
    warnings.warn("GEMINI_API_KEY is not set — Gemini judge will return hardcoded mock scores.")

# ── MERT Model Toggle ──────────────────────────────────────────────────────
# Switch between "small" and "big" to select the MERT model variant.
#   "small" → m-a-p/MERT-v1-95M   (12 layers, 768-dim, ~1-2 GB VRAM)
#   "big"   → m-a-p/MERT-v1-330M  (24 layers, 1024-dim, ~4-6 GB VRAM)
MERT_MODEL_SIZE = "small"  # <-- change this: "small" or "big"

MERT_MODELS = {
    "small": "m-a-p/MERT-v1-95M",
    "big": "m-a-p/MERT-v1-330M",
}
MERT_MODEL_NAME = MERT_MODELS[MERT_MODEL_SIZE]