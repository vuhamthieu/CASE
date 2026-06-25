#!/usr/bin/env python3
"""List Raspberry Pi outputs or play a short test tone on one device."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.output_device import play_int16_mono


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List audio devices.")
    parser.add_argument("--device", help="Output device index or name.")
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--frequency", type=float, default=440.0)
    args = parser.parse_args()

    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit("Install sounddevice before running this test.") from exc

    if args.list:
        devices = sd.query_devices()
        print(devices)
        print(f"Default devices (input, output): {sd.default.device}")
        print("\nOutput-capable devices:")
        for index, info in enumerate(devices):
            channels = int(info.get("max_output_channels", 0))
            if channels > 0:
                print(f"  {index}: {info['name']} ({channels} out)")
        print("\nInput-capable devices:")
        for index, info in enumerate(devices):
            channels = int(info.get("max_input_channels", 0))
            if channels > 0:
                print(f"  {index}: {info['name']} ({channels} in)")
        return
    if args.duration <= 0 or args.frequency <= 0:
        raise SystemExit("--duration and --frequency must be positive")
    if args.device:
        os.environ["CASE_AUDIO_OUTPUT_DEVICE"] = args.device

    sample_rate = 48_000
    frame_count = int(round(sample_rate * args.duration))
    phase = np.arange(frame_count, dtype=np.float32) / sample_rate
    audio = np.sin(2.0 * np.pi * args.frequency * phase) * 0.25
    fade_frames = min(frame_count // 2, int(sample_rate * 0.02))
    if fade_frames:
        fade = np.linspace(0.0, 1.0, fade_frames, dtype=np.float32)
        audio[:fade_frames] *= fade
        audio[-fade_frames:] *= fade[::-1]
    pcm = np.clip(np.rint(audio * 32767.0), -32768, 32767).astype(np.int16)

    try:
        result = play_int16_mono(pcm, sample_rate, post_guard_sec=0.10)
    except Exception as exc:
        raise SystemExit(f"AUDIO_TEST: playback failed: {exc}") from exc
    print(
        "AUDIO_TEST: played 440 Hz tone "
        f"device={result['device_name']!r} rate={result['sample_rate']} "
        f"channels={result['channels']} underflow={result['underflow']}"
    )


if __name__ == "__main__":
    main()
