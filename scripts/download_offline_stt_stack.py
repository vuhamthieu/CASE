#!/usr/bin/env python3
"""Explicit downloader for optional CASE offline speech models."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "ai" / "stt"
MODELS = {
    "vosk_lgraph": (
        "https://alphacephei.com/vosk/models/"
        "vosk-model-en-us-0.22-lgraph.zip",
        "vosk-model-en-us-0.22-lgraph.zip",
    ),
    "silero_vad": (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "asr-models/silero_vad.onnx",
        "silero_vad.onnx",
    ),
    "gtcrn": (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "speech-enhancement-models/gtcrn_simple.onnx",
        "gtcrn_simple.onnx",
    ),
    "smart_turn": (
        "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/"
        "smart-turn-v3.2-cpu.onnx",
        "smart-turn-v3.2-cpu.onnx",
    ),
    "sensevoice": (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
        "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2",
        "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2",
    ),
}
RECOMMENDED = ("vosk_lgraph", "silero_vad", "smart_turn")


def download(name: str, *, force: bool = False) -> None:
    url, filename = MODELS[name]
    DESTINATION.mkdir(parents=True, exist_ok=True)
    target = DESTINATION / filename
    if target.exists() and not force:
        print(f"OFFLINE_STT: keeping existing {target}")
    else:
        temporary = target.with_suffix(target.suffix + ".part")
        print(f"OFFLINE_STT: downloading {name} from {url}")
        with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(target)
        print(f"OFFLINE_STT: saved {target}")
    if name == "sensevoice":
        extract_sensevoice(target)
    elif name == "vosk_lgraph":
        extract_zip(target)


def extract_sensevoice(archive: Path) -> None:
    destination = DESTINATION.resolve()
    with tarfile.open(archive, "r:bz2") as bundle:
        for member in bundle.getmembers():
            member_path = (destination / member.name).resolve()
            if destination not in member_path.parents and member_path != destination:
                raise RuntimeError(f"unsafe archive path: {member.name}")
        bundle.extractall(destination)
    print(f"OFFLINE_STT: extracted SenseVoice under {destination}")


def extract_zip(archive: Path) -> None:
    destination = DESTINATION.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            member_path = (destination / member.filename).resolve()
            if destination not in member_path.parents and member_path != destination:
                raise RuntimeError(f"unsafe archive path: {member.filename}")
        bundle.extractall(destination)
    print(f"OFFLINE_STT: extracted {archive.name} under {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download optional offline STT/VAD models only when requested."
    )
    parser.add_argument("--model", choices=sorted(MODELS))
    parser.add_argument("--all-recommended", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.model and not args.all_recommended:
        parser.error("choose --model or --all-recommended")
    names = RECOMMENDED if args.all_recommended else (args.model,)
    for name in names:
        download(str(name), force=args.force)


if __name__ == "__main__":
    main()
