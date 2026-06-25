#!/usr/bin/env python3
"""Process, audition, and select recorded CASE wake acknowledgements."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.audio_format import convert_channels, load_wav_int16
from src.audio.output_device import play_int16_mono
from src.audio.playback_manager import close_playback_manager
from src.audio.recorded_wake_ack_processing import (
    WAKE_ACK_PROCESSING_PRESETS,
    process_recorded_wake_ack,
)
from src.audio.wake_ack_audio import inspect_wake_ack
from src.config import defaults


CANONICAL_ACKS = (
    "yes",
    "im_listening",
)


def write_wav(path: Path, audio, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(audio.astype("<i2", copy=False).tobytes())


def process_one(
    name: str,
    preset_name: str,
    *,
    raw_dir: Path,
    processed_dir: Path,
) -> Path | None:
    source = raw_dir / f"{name}.wav"
    if not source.is_file():
        print(f"WAKE_ACK_PROCESS: missing raw source={source}")
        return None
    loaded, sample_rate, _ = load_wav_int16(source)
    mono = convert_channels(loaded, 1)[:, 0]
    processed = process_recorded_wake_ack(
        mono,
        sample_rate,
        WAKE_ACK_PROCESSING_PRESETS[preset_name],
    )
    destination = processed_dir / f"{name}__{preset_name}.wav"
    write_wav(destination, processed, sample_rate)
    stats = inspect_wake_ack(processed, sample_rate)
    print(
        f"WAKE_ACK_PROCESS: preset={preset_name} source={source} "
        f"output={destination} sample_rate={sample_rate} "
        f"duration={stats.duration_sec:.3f}s peak={stats.peak_dbfs:.1f}dBFS "
        f"clipping={stats.clipped}"
    )
    return destination


def play_wav(path: Path, label: str) -> None:
    audio, sample_rate, _ = load_wav_int16(path)
    print(f"WAKE_ACK_AUDITION: playing {label} path={path}")
    result = play_int16_mono(audio, sample_rate, safe_mode=True)
    print(f"WAKE_ACK_AUDITION: underflow={result['underflow']}")


def load_manifest(path: Path) -> dict:
    if not path.is_file():
        return {"selections": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data.get("selections"), dict):
            data["selections"] = {}
        return data
    except (OSError, json.JSONDecodeError):
        return {"selections": {}}


def audition(
    names: list[str],
    *,
    raw_dir: Path,
    processed_dir: Path,
    runtime_dir: Path,
    manifest_path: Path,
) -> None:
    manifest = load_manifest(manifest_path)
    preset_names = tuple(WAKE_ACK_PROCESSING_PRESETS)
    for name in names:
        raw_path = raw_dir / f"{name}.wav"
        if not raw_path.is_file():
            print(f"WAKE_ACK_AUDITION: missing raw source={raw_path}; skipping")
            continue
        print(f"\n[{name}] raw is reference-only; runtime requires a processed take.")
        play_wav(raw_path, "raw reference")
        choices: list[tuple[str, Path]] = []
        for preset_name in preset_names:
            path = process_one(
                name,
                preset_name,
                raw_dir=raw_dir,
                processed_dir=processed_dir,
            )
            if path is not None:
                choices.append((preset_name, path))
                play_wav(path, preset_name)
        if not choices:
            continue
        for index, (preset_name, _) in enumerate(choices, 1):
            print(f"{index}) {preset_name}")
        print("0) skip")
        while True:
            answer = input(f"Choose processed take [1-{len(choices)}]: ").strip()
            if answer == "0":
                break
            if answer.isdigit() and 1 <= int(answer) <= len(choices):
                preset_name, selected = choices[int(answer) - 1]
                runtime_dir.mkdir(parents=True, exist_ok=True)
                destination = runtime_dir / f"{name}.wav"
                shutil.copy2(selected, destination)
                manifest["selections"][name] = {
                    "preset": preset_name,
                    "processed_source": str(selected),
                    "runtime_wav": str(destination),
                }
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(
                    json.dumps(manifest, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(
                    f"WAKE_ACK_SELECTED: name={name} preset={preset_name} "
                    f"path={destination}"
                )
                break
            print("Choose one of the listed processed presets, or 0 to skip.")


def main() -> None:
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--only", choices=CANONICAL_ACKS)
    parser.add_argument(
        "--preset",
        choices=tuple(WAKE_ACK_PROCESSING_PRESETS),
        default="case_robot",
    )
    parser.add_argument("--audition", action="store_true")
    parser.add_argument("--playback-device")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / defaults.WAKE_ACK_RECORDED_RAW_DIR,
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT / defaults.WAKE_ACK_RECORDED_PROCESSED_DIR,
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=ROOT / defaults.WAKE_ACK_RECORDED_DIR,
    )
    args = parser.parse_args()
    if not args.audition and not args.all and not args.only:
        parser.error("choose --all, --only, or --audition")
    if args.playback_device:
        os.environ["CASE_AUDIO_OUTPUT_DEVICE"] = args.playback_device
        os.environ["AUDIO_OUTPUT_DEVICE"] = args.playback_device

    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else ROOT / args.raw_dir
    processed_dir = (
        args.processed_dir
        if args.processed_dir.is_absolute()
        else ROOT / args.processed_dir
    )
    runtime_dir = (
        args.runtime_dir if args.runtime_dir.is_absolute() else ROOT / args.runtime_dir
    )
    names = [args.only] if args.only else list(CANONICAL_ACKS)
    processed_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.audition:
            audition(
                names,
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                runtime_dir=runtime_dir,
                manifest_path=ROOT / "assets/audio/wake_ack/recorded_selection.json",
            )
        else:
            for name in names:
                process_one(
                    name,
                    args.preset,
                    raw_dir=raw_dir,
                    processed_dir=processed_dir,
                )
    finally:
        close_playback_manager()


if __name__ == "__main__":
    main()
