#!/usr/bin/env python3
"""Interactively record acted wake acknowledgements for CASE."""

from __future__ import annotations

import argparse
import os
import sys
import wave
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.audio_format import resample_audio
from src.audio.output_device import play_int16_mono
from src.audio.playback_manager import close_playback_manager
from src.config import defaults


ACK_LINES = {
    "yes": '"Yes!" (clear, responsive)',
    "im_listening": '"I\'m listening." (calm and present)',
}


def prepare_raw_recording(audio: np.ndarray) -> np.ndarray:
    """Convert capture samples to PCM16 without trimming or normalization."""
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not samples.size:
        raise ValueError("recording is empty")
    peak = float(np.max(np.abs(samples)))
    if peak < 0.002:
        raise ValueError("recording is too quiet; check the microphone gain")
    return np.ascontiguousarray(
        np.clip(np.rint(samples * 32767.0), -32768, 32767).astype("<i2")
    )


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(np.ascontiguousarray(audio, dtype="<i2").tobytes())


def record_one(
    prompt: str,
    *,
    device: int | str | None,
    capture_rate: int,
    output_rate: int,
    duration: float,
) -> np.ndarray:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice is required to record wake acknowledgements"
        ) from exc
    print(f"\nPerform: {prompt}")
    input("Press Enter, then perform the line... ")
    print("Recording...")
    captured = sd.rec(
        int(round(capture_rate * duration)),
        samplerate=capture_rate,
        channels=1,
        dtype="float32",
        device=device,
        blocking=True,
    )[:, 0]
    if capture_rate != output_rate:
        captured = (
            resample_audio(captured, capture_rate, output_rate)[:, 0].astype(np.float32)
            / 32768.0
        )
    return prepare_raw_recording(captured)


def main() -> None:
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--only", choices=tuple(ACK_LINES))
    parser.add_argument("--device", help="microphone device index or name")
    parser.add_argument(
        "--sample-rate",
        type=int,
        choices=(22_050, 44_100),
        default=44_100,
        help="saved WAV sample rate",
    )
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--playback-device", help="speaker device index or name")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / defaults.WAKE_ACK_RECORDED_RAW_DIR,
    )
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")

    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit(
            "WAKE_ACK_RECORD: sounddevice is required; install requirements.txt"
        ) from exc

    device: int | str | None = args.device
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    info = sd.query_devices(device, "input")
    capture_rate = int(round(float(info["default_samplerate"])))
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.playback_device:
        os.environ["CASE_AUDIO_OUTPUT_DEVICE"] = args.playback_device
        os.environ["AUDIO_OUTPUT_DEVICE"] = args.playback_device

    selected = [args.only] if args.only else list(ACK_LINES)
    print(
        f"WAKE_ACK_RECORD: microphone={info['name']!r} capture_rate={capture_rate} "
        f"save_rate={args.sample_rate} mono duration={args.duration:.1f}s"
    )
    try:
        for name in selected:
            path = output_dir / f"{name}.wav"
            while True:
                try:
                    audio = record_one(
                        ACK_LINES[name],
                        device=device,
                        capture_rate=capture_rate,
                        output_rate=args.sample_rate,
                        duration=args.duration,
                    )
                except ValueError as exc:
                    print(f"WAKE_ACK_RECORD: {exc}; retrying")
                    continue
                write_wav(path, audio, args.sample_rate)
                peak = int(np.max(np.abs(audio.astype(np.int32))))
                peak_dbfs = 20.0 * np.log10(peak / 32767.0) if peak else float("-inf")
                print(
                    f"WAKE_ACK_RECORD: raw_saved={path} "
                    f"duration={len(audio) / args.sample_rate:.3f}s "
                    f"peak={peak_dbfs:.1f}dBFS clipping={peak >= 32767}"
                )
                while True:
                    choice = input("[A]ccept, [P]lay, [R]etry, or [S]kip? ").strip().lower()
                    if choice in {"", "a", "accept"}:
                        break
                    if choice in {"p", "play"}:
                        result = play_int16_mono(
                            audio,
                            args.sample_rate,
                            safe_mode=True,
                        )
                        print(f"WAKE_ACK_RECORD: playback underflow={result['underflow']}")
                        continue
                    if choice in {"r", "retry"}:
                        path.unlink(missing_ok=True)
                        break
                    if choice in {"s", "skip"}:
                        path.unlink(missing_ok=True)
                        break
                    print("Choose A, P, R, or S.")
                if choice in {"", "a", "accept", "s", "skip"}:
                    break
    except KeyboardInterrupt:
        print("\nWAKE_ACK_RECORD: stopped")
    finally:
        close_playback_manager()


if __name__ == "__main__":
    main()
