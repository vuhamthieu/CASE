#!/usr/bin/env python3
"""Inspect and play a WAV through CASE's normalized output path."""

from __future__ import annotations

import argparse
import os
import sys
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.audio_format import load_wav_int16, normalize_for_playback


def write_pcm16_wav(path: Path, audio, sample_rate: int) -> None:
    channels = audio.shape[1] if audio.ndim == 2 else 1
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(audio.astype("<i2", copy=False).tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--target-rate", type=int, default=44_100)
    parser.add_argument("--target-channels", type=int, default=2)
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--write-test-files", action="store_true")
    args = parser.parse_args()

    path = args.wav if args.wav.is_absolute() else ROOT / args.wav
    if not path.is_file():
        raise SystemExit(f"AUDIO_SPEED_TEST: WAV missing: {path}")
    if args.target_rate <= 0 or args.target_channels <= 0:
        parser.error("target rate and channels must be positive")

    audio, source_rate, source_channels = load_wav_int16(path)
    normalized = normalize_for_playback(
        audio,
        source_rate,
        args.target_rate,
        args.target_channels,
    )
    duration_in = len(audio) / float(source_rate)
    duration_out = len(normalized) / float(args.target_rate)
    resampled = source_rate != args.target_rate
    print(
        f"AUDIO_SPEED_TEST: source_rate={source_rate} "
        f"source_channels={source_channels} duration_in={duration_in:.3f}s"
    )
    print(
        f"AUDIO_SPEED_TEST: target_rate={args.target_rate} "
        f"target_channels={args.target_channels} duration_out={duration_out:.3f}s "
        f"resampled={resampled}"
    )
    if abs(duration_out - duration_in) / max(duration_in, 1e-9) > 0.02:
        print("AUDIO_SPEED_TEST: WARNING duration changed by more than 2%")

    if args.write_test_files:
        original_path = Path("/tmp/wake_ack_original.wav")
        converted_path = Path(f"/tmp/wake_ack_resampled_{args.target_rate}.wav")
        write_pcm16_wav(original_path, audio, source_rate)
        write_pcm16_wav(converted_path, normalized, args.target_rate)
        print(f"AUDIO_SPEED_TEST: wrote {original_path}")
        print(f"AUDIO_SPEED_TEST: wrote {converted_path}")

    if args.play:
        os.environ["AUDIO_OUTPUT_SAMPLE_RATE"] = str(args.target_rate)
        os.environ["AUDIO_OUTPUT_CHANNELS"] = str(args.target_channels)
        from src.audio.playback_manager import close_playback_manager, get_playback_manager

        manager = get_playback_manager()
        try:
            result = manager.play(audio, source_rate, safe_mode=True)
            print(
                "AUDIO_SPEED_TEST: playback "
                f"underflow={result['underflow']} "
                f"target_rate={result['sample_rate']} channels={result['channels']}"
            )
        finally:
            close_playback_manager()


if __name__ == "__main__":
    main()
