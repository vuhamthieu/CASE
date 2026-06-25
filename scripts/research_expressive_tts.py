#!/usr/bin/env python3
"""Dependency-free planning harness for future expressive TTS providers."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.env import get_str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("none", "elevenlabs", "cartesia"), default="none")
    parser.add_argument("--text", default="Operational. Mildly disappointed, but stable.")
    parser.add_argument("--output-dir", type=Path, default=Path("research/voice_samples"))
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.provider == "none":
        print("EXPRESSIVE_TTS_RESEARCH: dry-run only")
        print(f"provider=none\ntext={args.text}\noutput_dir={output_dir}")
        return

    prefix = args.provider.upper()
    key = get_str(f"{prefix}_API_KEY", "")
    voice_id = get_str(f"{prefix}_VOICE_ID", "")
    if not key or not voice_id:
        print(
            f"EXPRESSIVE_TTS_RESEARCH: {args.provider} key/voice ID missing. "
            "No request sent; Gemini Live native remains the runtime fallback."
        )
        return
    print(
        f"EXPRESSIVE_TTS_RESEARCH: {args.provider} credentials are configured, "
        "but its optional SDK adapter is a TODO. No request sent."
    )


if __name__ == "__main__":
    main()
