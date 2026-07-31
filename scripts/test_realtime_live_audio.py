#!/usr/bin/env python3
"""Open a short standalone microphone-to-Gemini-Live audio session."""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.realtime.realtime_voice_engine import RealtimeVoiceEngine


async def run(
    duration: float,
    disable_barge_in: bool,
    dump_wav: bool,
    half_duplex: bool | None,
    voice: str | None,
    persona: str | None,
) -> None:
    engine = RealtimeVoiceEngine(
        message_bus=None,
        enable_barge_in=False if disable_barge_in else None,
        dump_model_audio_wav=True if dump_wav else None,
        half_duplex=half_duplex,
        voice_name=voice,
        persona_name=persona,
    )
    ok = await engine.run_session(time.monotonic(), max_duration_sec=duration)
    if not ok:
        raise SystemExit("Realtime audio test failed; inspect REALTIME logs above")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--disable-barge-in", action="store_true")
    parser.add_argument("--dump-wav", action="store_true")
    duplex = parser.add_mutually_exclusive_group()
    duplex.add_argument("--half-duplex", action="store_true")
    duplex.add_argument("--full-duplex", action="store_true")
    parser.add_argument("--voice")
    parser.add_argument("--persona")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    logging.basicConfig(level=logging.INFO)
    half_duplex = True if args.half_duplex else False if args.full_duplex else None
    asyncio.run(
        run(
            args.duration,
            args.disable_barge_in,
            args.dump_wav,
            half_duplex,
            args.voice,
            args.persona,
        )
    )


if __name__ == "__main__":
    main()
