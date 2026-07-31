#!/usr/bin/env python3
"""Minimal Gemini Live connection test using text input and audio output."""

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.realtime.realtime_config import GEMINI_API_KEY, GEMINI_LIVE_MODEL


async def run(prompt: str) -> None:
    if not GEMINI_API_KEY:
        raise SystemExit("GEMINI_API_KEY is missing; add it to CASE/.env")
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)
    config = {
        "response_modalities": ["AUDIO"],
        "output_audio_transcription": {},
    }
    audio_bytes = 0
    transcript = []
    async with client.aio.live.connect(
        model=GEMINI_LIVE_MODEL,
        config=config,
    ) as session:
        await session.send_realtime_input(text=prompt)
        async for response in session.receive():
            data = getattr(response, "data", None)
            if isinstance(data, bytes):
                audio_bytes += len(data)
            content = getattr(response, "server_content", None)
            output = getattr(content, "output_transcription", None)
            if output and getattr(output, "text", None):
                transcript.append(output.text)
            if content and getattr(content, "turn_complete", False):
                break
    print(f"Connected model: {GEMINI_LIVE_MODEL}")
    print(f"Audio received: {audio_bytes} bytes")
    print("Transcript:", "".join(transcript).strip() or "(not returned)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Say: CASE realtime voice is online.")
    args = parser.parse_args()
    asyncio.run(asyncio.wait_for(run(args.prompt), timeout=30.0))


if __name__ == "__main__":
    main()
