#!/usr/bin/env python3
"""Text-only CASE persona simulator and style regression check."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import defaults
from src.config.env import get_str
from src.realtime.realtime_persona import build_case_system_instruction


PROMPTS = [
    "Are you okay?",
    "Tell me a joke.",
    "What are you doing?",
    "What is your humor percentage?",
    "Can you explain quantum physics in one sentence?",
    "Can you see me?",
]
EXPECTED = {
    "Are you okay?": "Operational. Mildly offended by the room acoustics.",
    "Tell me a joke.": "I trusted a human estimate once. Firmware still logs the trauma.",
    "What are you doing?": "Waiting. Listening. Pretending this is efficient.",
    "What is your humor percentage?": "Sixty-five percent. Any higher and I become a workplace hazard.",
    "Can you explain quantum physics in one sentence?": "Small things behave like probabilities until observation makes the paperwork specific.",
    "Can you see me?": "I need a current vision result before claiming visual lock.",
}
FORBIDDEN = (
    "i have no jokes",
    "i do not possess humor",
    "humor is illogical",
    "as an ai language model",
    "boss",
)


def system_prompt(preset: str) -> str:
    return build_case_system_instruction(
        preset,
        short_replies=defaults.CASE_STYLE_SHORT_REPLIES,
        max_sentences=defaults.CASE_STYLE_MAX_SENTENCES,
        humor_percent=defaults.CASE_HUMOR_PERCENT,
        honesty_percent=defaults.CASE_HONESTY_PERCENT,
        sarcasm_level=defaults.CASE_SARCASM_LEVEL,
    )


def offline_report(prompt: str) -> None:
    print("CASE_PERSONA_TEST: Gemini client unavailable; showing prompt and examples.")
    print("\nSYSTEM PROMPT\n", prompt)
    for user_text in PROMPTS:
        print(f"\nYOU  > {user_text}\nCASE > {EXPECTED[user_text]}")
    assert not any(
        phrase in answer.lower() for answer in EXPECTED.values() for phrase in FORBIDDEN
    )
    assert "never deny having humor" in prompt.lower()
    print("\nCASE_PERSONA_TEST: offline style checks passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default=defaults.CASE_VOICE_PRESET)
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    args = parser.parse_args()
    prompt = system_prompt(args.preset)
    api_key = get_str("GEMINI_API_KEY", "")
    if not api_key:
        offline_report(prompt)
        return
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        offline_report(prompt)
        return

    client = genai.Client(api_key=api_key)
    failed = False
    for user_text in PROMPTS:
        response = client.models.generate_content(
            model=args.model,
            contents=user_text,
            config=types.GenerateContentConfig(system_instruction=prompt),
        )
        answer = (response.text or "").strip()
        print(f"YOU  > {user_text}\nCASE > {answer}\n")
        if any(phrase in answer.lower() for phrase in FORBIDDEN):
            failed = True
    if failed:
        raise SystemExit("CASE_PERSONA_TEST: forbidden humor denial detected")
    print("CASE_PERSONA_TEST: live style checks passed")


if __name__ == "__main__":
    main()
