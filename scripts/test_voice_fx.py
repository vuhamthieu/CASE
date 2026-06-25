#!/usr/bin/env python3
"""Apply a CASE VoiceFX preset to a WAV or generated speech-like signal."""

import argparse
import sys
import wave
from math import gcd
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.voice_fx import VOICE_FX_PRESETS, VoiceFX


def load_audio(path: Path) -> bytes:
    with wave.open(str(path), "rb") as source:
        rate = source.getframerate()
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        frames = source.readframes(source.getnframes())
    if sample_width != 2:
        raise ValueError(
            f"input must be 16-bit PCM WAV; sample width is {sample_width} bytes"
        )
    audio = np.frombuffer(frames, dtype="<i2")
    if channels > 1:
        audio = audio.reshape(-1, channels).astype(np.float32).mean(axis=1)
        audio = np.clip(np.rint(audio), -32768, 32767).astype(np.int16)
    if rate != 24_000:
        try:
            from scipy.signal import resample_poly
        except ImportError as exc:
            raise RuntimeError(
                "scipy is required to resample non-24kHz WAV input"
            ) from exc
        divisor = gcd(int(rate), 24_000)
        audio = resample_poly(audio.astype(np.float32), 24_000 // divisor, int(rate) // divisor)
        audio = np.clip(np.rint(audio), -32768, 32767).astype(np.int16)
    return audio.tobytes()


def generated_signal() -> bytes:
    rate = 24_000
    timeline = np.arange(rate * 2, dtype=np.float32) / rate
    envelope = np.minimum(1.0, np.maximum(0.0, np.sin(2 * np.pi * 3.0 * timeline)))
    signal = envelope * (
        0.13 * np.sin(2 * np.pi * 130 * timeline)
        + 0.06 * np.sin(2 * np.pi * 260 * timeline)
        + 0.03 * np.sin(2 * np.pi * 2100 * timeline)
    )
    return np.clip(np.rint(signal * 32767), -32768, 32767).astype(np.int16).tobytes()


def write_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(pcm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--preset", choices=VOICE_FX_PRESETS, default="cinematic_robot_v1")
    parser.add_argument("--output", type=Path, default=Path("output/voice_fx_debug/test_fx.wav"))
    args = parser.parse_args()
    source = args.input.expanduser() if args.input else None
    if source and not source.is_file():
        parser.error(f"input WAV does not exist: {source}")
    pcm = load_audio(source) if source else generated_signal()
    processed = VoiceFX(24_000, args.preset).process_int16_mono(pcm)
    destination = args.output if args.output.is_absolute() else ROOT / args.output
    write_wav(destination, processed)
    print(f"VOICE_FX_TEST: input={source or '(generated signal)'}")
    print(f"VOICE_FX_TEST: preset={args.preset}")
    print(f"VOICE_FX_TEST: output={destination}")


if __name__ == "__main__":
    main()
