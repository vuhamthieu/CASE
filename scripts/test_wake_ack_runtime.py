#!/usr/bin/env python3
"""Inspect and optionally play cached wake acknowledgements through runtime audio."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.audio_format import convert_channels, load_wav_int16
from src.audio.output_device import play_int16_mono
from src.audio.playback_manager import close_playback_manager
from src.audio.wake_ack_audio import inspect_wake_ack
from src.config import defaults


def generated_path(key: str, directory: Path) -> Path:
    return directory / f"{key}.wav"


def check_file(path: Path, *, play: bool) -> bool:
    if not path.is_file():
        print(f"{path.name} MISSING path={path}")
        return False
    audio, sample_rate, channels = load_wav_int16(path)
    mono = convert_channels(audio, 1)[:, 0]
    stats = inspect_wake_ack(mono, sample_rate)
    ok = (
        stats.duration_sec * 1000.0 >= defaults.WAKE_ACK_MIN_TOTAL_DURATION_MS
        and stats.voiced_ms >= defaults.WAKE_ACK_MIN_VOICED_MS
        and not stats.clipped
    )
    print(
        f"{path.name} source_rate={sample_rate} channels={channels} "
        f"duration={stats.duration_sec:.3f}s voiced={stats.voiced_ms:.0f}ms "
        f"peak={stats.peak_dbfs:.1f}dBFS ok={ok} path={path}"
    )
    if play:
        result = play_int16_mono(
            np.ascontiguousarray(mono),
            sample_rate,
            post_guard_sec=defaults.WAKE_ACK_POST_PLAYBACK_GUARD_SEC,
            safe_mode=True,
            extra_tail_sec=defaults.WAKE_ACK_EXTRA_RUNTIME_TAIL_MS / 1000.0,
        )
        print(
            f"  playback target_rate={result['sample_rate']} "
            f"target_channels={result['channels']} "
            f"duration_in={result['duration_in']:.3f}s "
            f"duration_out={result['duration_out']:.3f}s "
            f"resampled={result['resampled']} underflow={result['underflow']}"
        )
        ok = ok and not bool(result["underflow"])
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all-default",
        action="store_true",
        help="Test the default runtime wake acknowledgement pool.",
    )
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Also test old optional wake acknowledgement files.",
    )
    parser.add_argument("--play", action="store_true")
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=ROOT / defaults.WAKE_ACK_WAV_DIR,
    )
    args = parser.parse_args()
    if not args.all_default and not args.include_legacy:
        parser.error("choose --all-default and/or --include-legacy")

    directory = args.generated_dir
    if not directory.is_absolute():
        directory = ROOT / directory
    keys: list[str] = []
    if args.all_default:
        keys.extend(defaults.DEFAULT_WAKE_ACK_POOL)
    if args.include_legacy:
        keys.extend(
            [
                "im_listening",
                "go_on",
                "im_here",
                "say_that_again",
                "still_with_you",
                *defaults.OPTIONAL_SHORT_WAKE_ACKS,
            ]
        )

    all_ok = True
    try:
        for key in keys:
            all_ok = check_file(generated_path(key, directory), play=args.play) and all_ok
    finally:
        close_playback_manager()
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
