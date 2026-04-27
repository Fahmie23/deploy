import json
import re
import os
from typing import Dict, Any
from google import genai
from google.genai import types
from config import GEMINI_API_KEY

GEMINI_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """
You are an expert **VSING singing judge**. You will listen to an audio recording of a singer and evaluate their vocal performance.

**Your Task:**
Carefully listen to the audio and score the singer on each dimension below (0-100 scale), then provide written feedback.

**Scoring Dimensions:**
1. **pitch** - Pitch accuracy: How well does the singer hit the correct notes? Consider intonation, stability, and control.
2. **rhythm** - Timing and rhythm: How well does the singer maintain tempo, hit beats accurately, and stay consistent?
3. **vibrato** - Vibrato quality: Is vibrato present? Is it controlled and musical, or wobbly/absent?
4. **breath_control** - Breath management: Does the singer maintain steady airflow? Are phrases well-supported without audible strain?
5. **timbre** - Tone quality: Is the vocal tone pleasant, resonant, and well-produced? Consider clarity, warmth, and character.
6. **overall_performance** - Overall impression: Considering all dimensions holistically, how good is the performance?

**Written Feedback:**
- **"criticism"**: A concise paragraph evaluating all 5 vocal dimensions. Match your tone to the scores:
  - High (80-100): Praise the singer, interpret positively
  - Mid (50-79): Balance praise with specific critiques
  - Low (0-49): Be constructive but honest
- **"advice"**: A paragraph of high-level improvement advice. Focus on the weakest 2 areas. Do NOT give specific drills. Focus on principles (e.g., "focus on consistent airflow"). If all scores are high, focus on advanced artistry.

Do NOT cite specific numbers in your criticism or advice.

**Output Format:**
Return ONLY a valid JSON object with this exact structure:
{
    "pitch": <int 0-100>,
    "rhythm": <int 0-100>,
    "vibrato": <int 0-100>,
    "breath_control": <int 0-100>,
    "timbre": <int 0-100>,
    "overall_performance": <int 0-100>,
    "criticism": "<paragraph>",
    "advice": "<paragraph>"
}

Return ONLY the raw JSON. No markdown code blocks, no extra text.
"""

SCORE_KEYS = ["pitch", "rhythm", "vibrato", "breath_control", "timbre", "overall_performance"]


class GeminiSingingJudge:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing. Please check your .env and config.py.")
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def evaluate(self, audio_path: str) -> Dict[str, Any]:
        """
        Send an audio file to Gemini and get singing scores + feedback.

        Args:
            audio_path: Path to the audio file (wav, mp3, m4a, etc.)

        Returns:
            Dict with scores (pitch, rhythm, vibrato, breath_control, timbre,
            overall_performance) and text (criticism, advice)
        """
        # Upload audio file to Gemini
        print(f"[Gemini] Uploading: {os.path.basename(audio_path)}")
        audio_file = self.client.files.upload(file=audio_path)
        print(f"[Gemini] Uploaded as: {audio_file.name} ({audio_file.mime_type})")

        # Send audio + prompt to Gemini
        print("[Gemini] Evaluating...")
        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(
                            file_uri=audio_file.uri,
                            mime_type=audio_file.mime_type,
                        ),
                        types.Part(text="Listen to this singing performance and evaluate it."),
                    ],
                ),
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
            ),
        )

        # Parse response
        raw = response.text
        clean = self._clean_json_string(raw)

        try:
            result = json.loads(clean)
        except json.JSONDecodeError:
            print(f"[Gemini] Failed to parse JSON. Raw response:\n{raw}")
            return {"error": "Gemini JSON parse failed", "raw_response": raw}

        return result

    def _clean_json_string(self, text: str) -> str:
        """Remove markdown code blocks from LLM response."""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(json)?", "", text)
            text = re.sub(r"```$", "", text)
        return text.strip()
