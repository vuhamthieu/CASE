from __future__ import annotations

import argparse
import re
import wave
from math import gcd
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TARGET_SAMPLE_RATE = 16_000
QUIET_RMS_THRESHOLD = 0.01
CATEGORIES = ("positive_real", "negative_hard", "negative_normal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record real CASE wake word dataset samples."
    )
    parser.add_argument(
        "--category",
        choices=CATEGORIES,
        required=True,
        help="Dataset category to record.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of samples to record. Default: 100.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Seconds per sample. Default: 2.0.",
    )
    parser.add_argument(
        "--device",
        help="Optional sounddevice input device index or name.",
    )
    return parser.parse_args()


def parse_device(device: str | None) -> int | str | None:
    if device is None:
        return None

    try:
        return int(device)
    except ValueError:
        return device


def get_input_device_info(device: int | str | None):
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice is required. Install it with: "
            "python3 -m pip install sounddevice"
        ) from exc

    device_info = sd.query_devices(device=device, kind="input")
    max_channels = int(device_info["max_input_channels"])
    if max_channels < 1:
        raise RuntimeError(f"Selected device has no input channels: {device_info}")

    sample_rate = int(round(device_info["default_samplerate"]))
    return sd, device_info, sample_rate, max_channels


def next_sample_number(output_dir: Path, category: str) -> int:
    pattern = re.compile(rf"^{re.escape(category)}_(\d{{4}})\.wav$")
    highest = 0
    for wav_path in output_dir.glob(f"{category}_*.wav"):
        match = pattern.match(wav_path.name)
        if match:
            highest = max(highest, int(match.group(1)))

    return highest + 1


def record_native_audio(
    sd,
    device: int | str | None,
    sample_rate: int,
    max_channels: int,
    duration: float,
) -> np.ndarray:
    frames = int(round(sample_rate * duration))

    try:
        recording = sd.rec(
            frames,
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            device=device,
        )
        sd.wait()
        return np.asarray(recording, dtype=np.int16).reshape(-1)
    except Exception:
        channels = min(2, max_channels)
        recording = sd.rec(
            frames,
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
            device=device,
        )
        sd.wait()
        audio = np.asarray(recording, dtype=np.int16)
        if audio.ndim == 1:
            return audio

        mono = audio.astype(np.int32).mean(axis=1)
        return np.clip(
            np.rint(mono),
            np.iinfo(np.int16).min,
            np.iinfo(np.int16).max,
        ).astype(np.int16)


def resample_to_16khz(audio: np.ndarray, source_sample_rate: int) -> np.ndarray:
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required. Install it with: python3 -m pip install scipy"
        ) from exc

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if source_sample_rate != TARGET_SAMPLE_RATE:
        divisor = gcd(source_sample_rate, TARGET_SAMPLE_RATE)
        audio = resample_poly(
            audio,
            TARGET_SAMPLE_RATE // divisor,
            source_sample_rate // divisor,
        )

    return np.clip(
        np.rint(audio),
        np.iinfo(np.int16).min,
        np.iinfo(np.int16).max,
    ).astype(np.int16)


def audio_level(audio: np.ndarray) -> tuple[float, float]:
    if len(audio) == 0:
        return 0.0, 0.0

    normalized = audio.astype(np.float32) / np.iinfo(np.int16).max
    rms = float(np.sqrt(np.mean(normalized * normalized)))
    peak = float(np.max(np.abs(normalized)))
    return rms, peak


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.int16).reshape(-1)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(TARGET_SAMPLE_RATE)
        wav_file.writeframes(audio.tobytes())


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be at least 1")
    if args.duration <= 0:
        raise ValueError("--duration must be greater than 0")

    device = parse_device(args.device)
    sd, device_info, native_sample_rate, max_channels = get_input_device_info(device)
    output_dir = ROOT / "data" / "wakeword" / "hey_case" / args.category
    next_number = next_sample_number(output_dir, args.category)

    print(
        f"Recording category '{args.category}' to {output_dir}",
        flush=True,
    )
    print(
        f"Input device: {device_info['name']} "
        f"({native_sample_rate} Hz, max input channels={max_channels})",
        flush=True,
    )
    print(
        f"Each sample: {args.duration:.2f}s native audio -> "
        f"{TARGET_SAMPLE_RATE} Hz mono int16 WAV",
        flush=True,
    )

    try:
        for sample_index in range(args.count):
            sample_number = next_number + sample_index
            output_path = output_dir / f"{args.category}_{sample_number:04d}.wav"
            print(
                f"\nSample {sample_index + 1}/{args.count}: {output_path.name}",
                flush=True,
            )
            input("Press Enter, then record the sample...")
            print("Recording...", flush=True)

            native_audio = record_native_audio(
                sd=sd,
                device=device,
                sample_rate=native_sample_rate,
                max_channels=max_channels,
                duration=args.duration,
            )
            audio_16khz = resample_to_16khz(native_audio, native_sample_rate)
            write_wav(output_path, audio_16khz)

            rms, peak = audio_level(audio_16khz)
            quiet_warning = " too quiet" if rms < QUIET_RMS_THRESHOLD else ""
            print(
                f"Saved {output_path} "
                f"(rms={rms:.4f}, peak={peak:.4f}){quiet_warning}",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\nRecording stopped.", flush=True)


if __name__ == "__main__":
    main()
