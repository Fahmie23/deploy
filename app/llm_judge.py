import json
import re
from typing import Dict, Any
from openai import OpenAI
from config import VLLM_BASE_URL, MODEL_NAME, TEMPERATURE

class SingingJudge:
    def __init__(self):
        self.client = OpenAI(base_url=VLLM_BASE_URL, api_key="none")
        self.model = MODEL_NAME
        self.temperature = TEMPERATURE

        self.system_instruction = """
        You are an expert **VSING singing judge**.

        **Your Input:**
        You will receive a JSON object containing `predictions`: scores from 0-100
        for Pitch, Rhythm, Vibrato, Breath Control, Timbre, and Overall Performance.
        These scores come from a deep audio model that listens to the vocal
        performance directly.

        **Your Task:**
        Output a **valid JSON object** with keys `"criticism"` and `"advice"`.

        ### 1. "criticism" (Analysis)
        Write a concise paragraph evaluating the singer's Pitch, Rhythm, Vibrato, Breath Control, and Timbre.

        **CRITICAL RULE: The Tone MUST Match the Scores.**
        - **High Scores (80-100):** You MUST PRAISE the singer. Describe the technique positively (e.g., "expressive phrasing", "confident control").
        - **Mid Scores (50-79):** Balance praise with specific critiques. Acknowledge what is good, but point out inconsistency.
        - **Low Scores (0-49):** Be constructive but honest about the lack of technique.

        **Do NOT** cite the numeric scores in your prose. Instead, describe the singing:
        - *If Pitch score is High:* "The pitch shows excellent stability and control..."
        - *If Pitch score is Low:* "Noticeable instability in pitch suggests a lack of support..."

        ### 2. "advice" (Improvement)
        Write a paragraph of high-level advice.

        - **Focus Strategy:** Look at the **lowest 2 scores** in `predictions`. Tailor your advice specifically to improve those weak points.
        - If all scores are high, focus on advanced artistry and polish.

        - **Rule:** **DO NOT** give specific drills (no "straw phonation" or "Practice lip trills"). Focus on **principles** (e.g., "focus on consistent airflow," "mental anticipation of the beat").
        - Do NOT cite the specific numbers in your criticism or advice.

        **Output Format:**
        - Return ONLY the raw JSON object.
        - Do NOT wrap the output in markdown code blocks.

        ---

        **Example of the expected output quality and style:**

        Input scores: pitch=45, rhythm=52, vibrato=40, breath_control=38, timbre=50, overall_performance=45

        Output:
        {
            "criticism": "The singer demonstrates a commendable effort to convey emotion throughout the performance, particularly in the song's more dramatic moments. However, there are notable challenges in several technical areas. Pitch accuracy is a consistent struggle, with many notes, especially sustained and higher ones, falling flat or wavering significantly. While the rhythmic foundation is generally present, there's a tendency to lag slightly behind the beat, impacting the overall tightness. Vibrato is often present but lacks consistent control, sometimes appearing as an uncontrolled wobble rather than a musical embellishment. Breath management also presents difficulties, leading to unsupported phrases and audible strain, particularly when attempting to sustain notes or reach higher registers. The vocal timbre, while having a distinct character, can at times sound nasal and lacks consistent resonance, especially under pressure.",
            "advice": "To enhance future performances, the primary focus should be on developing a stronger foundation in pitch accuracy and breath control. Concentrating on consistent airflow and proper breath support will not only improve the stability of your notes but also reduce vocal strain, allowing for a more resonant and controlled tone. Simultaneously, dedicated practice on intonation exercises will help train your ear and vocal muscles to hit notes precisely. As these foundational elements strengthen, you can then work on refining your rhythmic precision and developing a more controlled and musical vibrato, which will naturally emerge from improved breath support and pitch stability."
        }
        """

    def evaluate(self, full_result: Dict[str, Any]) -> Dict[str, Any]:
        preds = full_result.get("predictions", {})

        final_output = {
            "pitch": preds.get("pitch", 0),
            "rhythm": preds.get("rhythm", 0),
            "vibrato": preds.get("vibrato", 0),
            "breath_control": preds.get("breath_control", 0),
            "timbre": preds.get("timbre", 0),
            "overall_performance": preds.get("overall_performance", 0),
            "criticism": "",
            "advice": ""
        }

        json_payload = json.dumps(full_result, indent=2, ensure_ascii=False)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=512,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self.system_instruction},
                    {"role": "user", "content": f"Here is the singing analysis data:\n{json_payload}"}
                ],
            )
            raw = response.choices[0].message.content
            print(f"[LLM Raw Response]: {raw}")
            clean_content = self._clean_json_string(raw)
            ai_data = json.loads(clean_content)

            final_output["criticism"] = ai_data.get("criticism", "Analysis unavailable.")
            final_output["advice"] = ai_data.get("advice", "Advice unavailable.")

        except json.JSONDecodeError:
            final_output["criticism"] = "Technical analysis generated but formatting failed."
            final_output["advice"] = "Focus on breath support and stability."
        except Exception as e:
            final_output["criticism"] = f"Error generating critique: {str(e)}"

        return final_output

    def _clean_json_string(self, text: str) -> str:
        # Try to extract a JSON object directly
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0).strip()
        return text.strip()
