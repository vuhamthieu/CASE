#!/usr/bin/env python3
"""Record a repeatable CASE command corpus with expected transcript files."""

from __future__ import annotations

import argparse
import sys
from math import gcd
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.input_device import configured_input_device


PHRASES = (
    "hey case tell me a joke",
    "can you tell me about yourself",
    "can you see me",
    "take a picture",
    "what are you doing",
    "tell me more",
    "can you explain that",
    "yeah continue",
    "stop talking",
    "go idle",
)
TARGET_RATE = 16_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/stt_test_phrases")
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--device", help="PortAudio input index or name")
    args = parser.parse_args()
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit("sounddevice is required") from exc

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if args.device is not None else configured_input_device()
    try:
        device = int(device) if isinstance(device, str) and device.isdigit() else device
        info = sd.query_devices(device, "input")
    except Exception as exc:
        raise SystemExit(f"Could not resolve input device {device!r}: {exc}") from exc
    native_rate = int(round(float(info["default_samplerate"])))
    print(f"STT_RECORD: device={info['name']!r} native_rate={native_rate}")

    for index, phrase in enumerate(PHRASES, start=1):
        slug = f"phrase_{index:02d}"
        input(f"\n[{index}/{len(PHRASES)}] Say: {phrase!r}\nPress Enter to record...")
        frames = int(round(args.duration * native_rate))
        recording = sd.rec(
            frames,
            samplerate=native_rate,
            channels=1,
            dtype="int16",
            device=device,
            blocking=True,
        ).reshape(-1)
        if native_rate != TARGET_RATE:
            divisor = gcd(native_rate, TARGET_RATE)
            recording = resample_poly(
                recording.astype(np.float32),
                TARGET_RATE // divisor,
                native_rate // divisor,
            )
            recording = np.clip(
                np.rint(recording), -32768, 32767
            ).astype(np.int16)
        wavfile.write(output_dir / f"{slug}.wav", TARGET_RATE, recording)
        (output_dir / f"{slug}.txt").write_text(phrase + "\n", encoding="utf-8")
        rms = float(np.sqrt(np.mean(recording.astype(np.float32) ** 2)))
        print(f"STT_RECORD: saved {slug}.wav rms={rms:.1f}")


if __name__ == "__main__":
    main()
