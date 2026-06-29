#!/usr/bin/env python3
"""Explicit downloader for CASE offline speech artifacts."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "ai" / "stt"
SILERO_VAD_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/"
    "src/silero_vad/data/silero_vad.onnx"
)
SILERO_VAD_RELATIVE_PATH = Path("ai/stt/silero_vad.onnx")
SILERO_VAD_MIN_BYTES = 100 * 1024
MODELS = {
    "vosk_lgraph": (
        "https://alphacephei.com/vosk/models/"
        "vosk-model-en-us-0.22-lgraph.zip",
        "vosk-model-en-us-0.22-lgraph.zip",
    ),
    "silero_vad": (
        SILERO_VAD_URL,
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


class InstallError(RuntimeError):
    """Raised when a runtime artifact cannot be safely installed."""


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def silero_vad_path(root: Path = ROOT) -> Path:
    return root / SILERO_VAD_RELATIVE_PATH


def smart_turn_path(root: Path = ROOT) -> Path:
    return root / "ai/stt/smart-turn-v3.2-cpu.onnx"


def install_silero_vad(
    *,
    root: Path = ROOT,
    force: bool = False,
    urlopen=urllib.request.urlopen,
    min_bytes: int = SILERO_VAD_MIN_BYTES,
) -> Path:
    target = silero_vad_path(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size >= min_bytes and not force:
        print(
            "INSTALL_SILERO_VAD: installed "
            f"path={_relative_to_root(target, root)} size={target.stat().st_size}"
        )
        return target

    temporary = target.with_name(f".{target.name}.part")
    if temporary.exists():
        temporary.unlink()
    print("INSTALL_SILERO_VAD: downloading")
    try:
        with urlopen(SILERO_VAD_URL) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if not temporary.exists():
            raise InstallError("download did not create a temp file")
        size = temporary.stat().st_size
        if size == 0:
            raise InstallError("downloaded file is zero bytes")
        if size < min_bytes:
            raise InstallError(
                f"downloaded file is too small: {size} bytes < {min_bytes} bytes"
            )
        temporary.replace(target)
        print(
            "INSTALL_SILERO_VAD: installed "
            f"path={_relative_to_root(target, root)} size={size}"
        )
        return target
    except Exception:
        if temporary.exists():
            temporary.unlink()
        print("INSTALL_SILERO_VAD: failed")
        raise


def check_baseline(root: Path = ROOT) -> bool:
    silero = silero_vad_path(root)
    ok = silero.exists() and silero.stat().st_size > 0
    status = "OK" if ok else "MISSING"
    print(f"{status:<7} PROFILE_REQUIRED {_relative_to_root(silero, root)}")
    if not smart_turn_path(root).exists():
        print(
            "OPTIONAL_smart_turn missing; CASE will use VAD/timing "
            "turn-ending fallback."
        )
    return ok


def install_baseline(*, root: Path = ROOT, force: bool = False) -> None:
    install_silero_vad(root=root, force=force)
    if not smart_turn_path(root).exists():
        print(
            "OPTIONAL_smart_turn missing; CASE will use VAD/timing "
            "turn-ending fallback."
        )


def download(name: str, *, force: bool = False) -> None:
    if name == "silero_vad":
        install_silero_vad(force=force)
        return
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
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--install-baseline", action="store_true")
    parser.add_argument("--install-silero-vad", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.check:
        check_baseline()
        return
    if args.install_baseline:
        install_baseline(force=args.force)
        return
    if args.install_silero_vad:
        install_silero_vad(force=args.force)
        return
    if not args.model and not args.all_recommended:
        parser.error(
            "choose --check, --install-baseline, --install-silero-vad, "
            "--model, or --all-recommended"
        )
    names = RECOMMENDED if args.all_recommended else (args.model,)
    for name in names:
        download(str(name), force=args.force)


if __name__ == "__main__":
    main()
