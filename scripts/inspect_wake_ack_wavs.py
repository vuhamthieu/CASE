#!/usr/bin/env python3
"""Inspect generated wake acknowledgements and optional recorded experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.audio_format import convert_channels, load_wav_int16
from src.audio.wake_ack_audio import inspect_wake_ack
from src.config import defaults


DEFAULT_WAKE_ACK_FILES = tuple(
    f"{key}.wav" for key in defaults.DEFAULT_WAKE_ACK_POOL
)
OPTIONAL_SHORT_WAKE_ACK_FILES = tuple(
    f"{key}.wav" for key in defaults.OPTIONAL_SHORT_WAKE_ACKS
)
LEGACY_WAKE_ACK_FILES = (
    "im_listening.wav",
    "go_on.wav",
    "im_here.wav",
    "say_that_again.wav",
    "still_with_you.wav",
) + OPTIONAL_SHORT_WAKE_ACK_FILES


def _recommendation(kind: str, stats, sample_rate: int, channels: int) -> tuple[bool, str]:
    common = sample_rate in {22_050, 44_100} and channels == 1 and not stats.clipped
    if not common:
        return False, "format_or_clipping"
    if stats.duration_sec * 1000.0 + 0.1 < defaults.WAKE_ACK_MIN_TOTAL_DURATION_MS:
        return False, "total_too_short"
    if stats.voiced_ms + 0.1 < defaults.WAKE_ACK_MIN_VOICED_MS:
        return False, "voiced_too_short"
    if kind == "generated":
        return stats.passed, "ok" if stats.passed else "padding_or_peak"
    if kind == "recorded_raw":
        ok = -35.0 <= stats.peak_dbfs <= -1.0
        return ok, "ok" if ok else "peak_out_of_range"
    ok = (
        stats.leading_silence_ms >= 110
        and stats.trailing_silence_ms >= 330
        and -8.0 <= stats.peak_dbfs <= -1.0
    )
    return ok, "ok" if ok else "padding_or_peak"


def inspect_directory(
    directory: Path,
    *,
    kind: str,
    expected_files: tuple[str, ...] | None,
) -> tuple[int, int, list[str]]:
    print(f"\nWAKE_ACK_INSPECT: source={kind} directory={directory}")
    if expected_files is None:
        paths = sorted(directory.glob("*.wav")) if directory.is_dir() else []
        missing: list[str] = []
    else:
        paths = [directory / filename for filename in expected_files]
        missing = [path.name for path in paths if not path.is_file()]
    present = 0
    failed = 0
    for path in paths:
        if not path.is_file():
            print(f"{path.name} MISSING")
            continue
        present += 1
        try:
            loaded, sample_rate, channels = load_wav_int16(path)
            audio = convert_channels(loaded, 1)[:, 0]
            stats = inspect_wake_ack(audio, sample_rate)
        except Exception as exc:
            print(f"{path.name} error={exc} FAIL")
            failed += 1
            continue
        recommended, reason = _recommendation(kind, stats, sample_rate, channels)
        peak = (
            f"{stats.peak_dbfs:.1f}dBFS"
            if np.isfinite(stats.peak_dbfs)
            else "-infdBFS"
        )
        print(
            f"{path.name} sample_rate={sample_rate} channels={channels} "
            f"duration={stats.duration_sec:.2f}s "
            f"voiced={stats.voiced_ms:.0f}ms "
            f"lead_silence={stats.leading_silence_ms:.0f}ms "
            f"tail_silence={stats.trailing_silence_ms:.0f}ms "
            f"peak={peak} clipping={'yes' if stats.clipped else 'no'} "
            f"recommended={'ok' if recommended else 'bad'} reason={reason} path={path}"
        )
        failed += int(not recommended)
    if not paths:
        print("no WAV files")
    return present, failed + len(missing), missing


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generated-dir", type=Path, default=ROOT / defaults.WAKE_ACK_WAV_DIR
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=ROOT / defaults.WAKE_ACK_RECORDED_RAW_DIR
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT / defaults.WAKE_ACK_RECORDED_PROCESSED_DIR,
    )
    parser.add_argument(
        "--recorded-dir", type=Path, default=ROOT / defaults.WAKE_ACK_RECORDED_DIR
    )
    parser.add_argument(
        "--include-recorded-experiment",
        action="store_true",
        help="Also inspect archived/experimental recorded wake-ack folders.",
    )
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Also inspect old optional wake acknowledgement files.",
    )
    args = parser.parse_args()

    results = []
    expected_generated = DEFAULT_WAKE_ACK_FILES + (
        LEGACY_WAKE_ACK_FILES if args.include_legacy else ()
    )
    results.append(
        inspect_directory(
            resolve(args.generated_dir),
            kind="generated",
            expected_files=expected_generated,
        )
    )
    if args.include_recorded_experiment:
        results.append(
            inspect_directory(
                resolve(args.raw_dir),
                kind="recorded_raw",
                expected_files=expected_generated,
            )
        )
        results.append(
            inspect_directory(
                resolve(args.processed_dir),
                kind="recorded_processed",
                expected_files=None,
            )
        )
        runtime_result = inspect_directory(
            resolve(args.recorded_dir),
            kind="recorded",
            expected_files=expected_generated,
        )
        results.append(runtime_result)
        if runtime_result[2]:
            print(
                "\nWAKE_ACK_RECORDED: missing " + ", ".join(runtime_result[2])
            )
    else:
        print("\nWAKE_ACK_RECORDED: skipped; runtime default is cached_wav")
    total_present = sum(result[0] for result in results)
    total_failed = sum(result[1] for result in results)
    print(
        f"WAKE_ACK_INSPECT: total_present={total_present} "
        f"total_failed={total_failed}"
    )
    if total_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
