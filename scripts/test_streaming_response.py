#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

try:
    from google import genai
except ImportError as exc:
    raise SystemExit(
        "google-genai is not installed. Activate the CASE venv and run: "
        "python3 -m pip install -r requirements.txt"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cognition.personality import GEMINI_MODEL, ResponseChunker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show Gemini streaming fragments and CASE TTS text chunks."
    )
    parser.add_argument("prompt", help="Prompt to stream from Gemini")
    parser.add_argument(
        "--model",
        default=GEMINI_MODEL,
        help=f"Gemini model name (default: {GEMINI_MODEL})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is missing from the environment or .env")

    client = genai.Client(api_key=api_key)
    chunker = ResponseChunker()
    started_at = time.monotonic()
    first_fragment_at = None
    queued_count = 0

    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print("Streaming response:\n")

    stream = client.models.generate_content_stream(
        model=args.model,
        contents=(
            "Speak naturally. For simple questions, be concise. For detailed "
            f"questions, answer fully. User question: {args.prompt}"
        ),
    )

    for response_chunk in stream:
        fragment = getattr(response_chunk, "text", None) or ""
        if not fragment:
            continue

        elapsed = time.monotonic() - started_at
        if first_fragment_at is None:
            first_fragment_at = elapsed
        print(f"[{elapsed:6.3f}s] LLM fragment: {fragment!r}")

        for speech_chunk in chunker.feed(fragment):
            queued_count += 1
            print(
                f"[{time.monotonic() - started_at:6.3f}s] "
                f"TTS chunk queued #{queued_count}: {speech_chunk!r}"
            )

    for speech_chunk in chunker.flush():
        queued_count += 1
        print(
            f"[{time.monotonic() - started_at:6.3f}s] "
            f"TTS chunk queued #{queued_count}: {speech_chunk!r}"
        )

    total = time.monotonic() - started_at
    first = f"{first_fragment_at:.3f}s" if first_fragment_at is not None else "n/a"
    print(
        f"\nDone: first_llm_chunk={first}, full_response={total:.3f}s, "
        f"tts_chunks={queued_count}"
    )


if __name__ == "__main__":
    main()
