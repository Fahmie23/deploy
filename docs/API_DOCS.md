# VSING Singing Judge API — Integration Guide

## Prerequisites

Before you can use the API, make sure the following are in place on your machine.

### 1. Docker Desktop is installed and running

Download from: https://www.docker.com/products/docker-desktop

Once installed, open **Docker Desktop** and wait until the bottom-left status shows **"Engine running"**.

> If you enabled "Start Docker Desktop when you sign in" in Docker Desktop settings, this happens automatically on boot and you can skip this step.

---

### 2. You have the API key

The API key is found in the `.env` file inside the deploy folder:

```
API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Keep this key — you will need it in every request.

---

### 3. Start the API

Open **WSL2** by pressing **Win + R** and typing:

```
wsl -d Ubuntu-22.04
```

Then navigate to the deploy folder:

```bash
cd /mnt/c/Users/fahmi/Downloads/deploy
```

Then start all containers:

```bash
docker compose up -d
```

The first time you run this, it will take a few minutes as the models are downloaded and loaded. On subsequent runs it starts much faster.

---

### 4. Verify the API is ready

Run this to check if the server is up:

```bash
curl http://localhost/vsing_api/stats
```

You should see a JSON response like:

```json
{
  "gpu_name": "NVIDIA GeForce RTX 3060",
  "allocated_mb": 312.5,
  "reserved_mb": 512.0,
  "total_mb": 12288.0,
  "free_mb": 11776.0
}
```

If you get a connection error, wait 30–60 seconds and try again — the models are still loading in the background.

---

## Base URL

```
http://localhost/vsing_api
```

---

## Authentication

Every request to `/judge` must include the API key in the request header:

```
X-API-Key: <your-api-key>
```

---

## Rate Limit

**10 requests per minute.** Exceeding this returns HTTP `429`.

---

## Endpoints

### 1. `GET /vsing_api/stats`

Check if the server is up and view GPU memory usage. No authentication required.

**Request**
```http
GET http://localhost/vsing_api/stats
```

**Response (200)**
```json
{
  "gpu_name": "NVIDIA GeForce RTX 3060",
  "allocated_mb": 312.5,
  "reserved_mb": 512.0,
  "total_mb": 12288.0,
  "free_mb": 11776.0
}
```

---

### 2. `POST /vsing_api/judge`

Upload an audio file and receive a singing evaluation with scores and feedback.

**Request**

| Part | Value |
|------|-------|
| Method | `POST` |
| URL | `http://localhost/vsing_api/judge` |
| Header | `X-API-Key: <your-api-key>` |
| Body | `multipart/form-data` with field `file` |

**Supported audio formats:** `.wav`, `.mp3`, `.m4a`, `.aac`, `.ogg`, `.flac`, `.webm`, `.mp4`  
**Max duration:** 8 minutes

**Response (200)**
```json
{
  "pitch": 72.5,
  "rhythm": 65.0,
  "vibrato": 58.3,
  "breath_control": 61.2,
  "timbre": 70.1,
  "overall_performance": 65.4,
  "criticism": "The singer demonstrates confident pitch control with good stability on sustained notes...",
  "advice": "Focus on consistent airflow to improve breath control and support your vibrato...",
  "weighted_pitch": 72.5,
  "weighted_rhythm": 65.0,
  "weighted_vibrato": 58.3,
  "weighted_breath_control": 61.2,
  "weighted_timbre": 70.1,
  "weighted_overall_performance": 65.4,
  "gemini_pitch": null,
  "gemini_rhythm": null,
  "gemini_vibrato": null,
  "gemini_breath_control": null,
  "gemini_timbre": null,
  "gemini_overall_performance": null,
  "gemini_criticism": null,
  "gemini_advice": null,
  "_metadata": {
    "model_name": "MERT-v1-95M + Ridge(alpha=20) per trait",
    "scale": "0-100",
    "filename": "recording.wav",
    "gemini_available": false
  }
}
```

**Score fields** (all on a 0–100 scale):

| Field | Description |
|-------|-------------|
| `pitch` | Pitch accuracy and stability |
| `rhythm` | Rhythmic precision |
| `vibrato` | Vibrato control |
| `breath_control` | Breath support and management |
| `timbre` | Vocal tone quality |
| `overall_performance` | Average of the five traits |
| `criticism` | Paragraph analysis of the performance |
| `advice` | Improvement advice targeting the weakest areas |

---

## Error Responses

| Status | Meaning |
|--------|---------|
| `400` | Bad request — unsupported file type or audio too long |
| `403` | Missing or invalid API key |
| `429` | Rate limit exceeded (10 requests/minute) |
| `500` | Internal server error during processing |

---

## Code Examples

### Python

Install the `requests` library first if you haven't:

```bash
pip install requests
```

```python
import requests

API_URL = "http://localhost/vsing_api/judge"
API_KEY = "<your-api-key>"

with open("recording.mp3", "rb") as f:
    response = requests.post(
        API_URL,
        headers={"X-API-Key": API_KEY},
        files={"file": ("recording.mp3", f, "audio/mpeg")},
    )

result = response.json()
print("Pitch:", result["pitch"])
print("Rhythm:", result["rhythm"])
print("Feedback:", result["criticism"])
print("Advice:", result["advice"])
```

---

### JavaScript (fetch)

```javascript
const fs = require("fs");

const formData = new FormData();
formData.append("file", new Blob([fs.readFileSync("recording.mp3")]), "recording.mp3");

const response = await fetch("http://localhost/vsing_api/judge", {
  method: "POST",
  headers: {
    "X-API-Key": "<your-api-key>",
  },
  body: formData,
});

const result = await response.json();
console.log(result);
```

---

### curl

```bash
curl -X POST http://localhost/vsing_api/judge \
  -H "X-API-Key: <your-api-key>" \
  -F "file=@recording.mp3"
```

---

## Quick Reference

| Task | Command / URL |
|------|--------------|
| Start the API | `docker compose up -d` |
| Stop the API | `docker compose down` |
| Check health | `curl http://localhost/vsing_api/stats` |
| Submit audio | `POST http://localhost/vsing_api/judge` |
| View logs | `docker compose logs -f singing-judge` |
