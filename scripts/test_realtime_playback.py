#!/usr/bin/env python3
"""Generate, save, and play a 24 kHz PCM tone through realtime output."""

import argparse
import asyncio
import logging
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.realtime.realtime_audio_io import RealtimeAudioOutput


async def run(seconds: float) -> None:
    sample_rate = 24_000
    samples = int(sample_rate * seconds)
    timeline = np.arange(samples, dtype=np.float64) / sample_rate
    fade_samples = min(int(sample_rate * 0.02), samples // 2)
    envelope = np.ones(samples, dtype=np.float64)
    if fade_samples:
        fade = np.linspace(0.0, 1.0, fade_samples)
        envelope[:fade_samples] = fade
        envelope[-fade_samples:] = fade[::-1]
    tone = (0.18 * 32767 * envelope * np.sin(2 * np.pi * 440 * timeline))
    pcm = np.clip(np.rint(tone), -32768, 32767).astype(np.int16).tobytes()

    path = ROOT / "output" / "realtime_debug" / "test_tone_24k.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)

    output = RealtimeAudioOutput()
    print("REALTIME_AUDIO_TEST: playing 24kHz int16 mono test tone")
    await output.start()
    try:
        output.enqueue(pcm)
        if not await output.wait_until_drained(timeout=seconds + 5.0):
            raise RuntimeError("playback did not drain before timeout")
    finally:
        await output.stop()
    print(f"REALTIME_AUDIO_TEST: saved {path.relative_to(ROOT)}")
    print("REALTIME_AUDIO_TEST: done")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run(args.seconds))


if __name__ == "__main__":
    main()
