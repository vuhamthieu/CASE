#!/usr/bin/env python3
"""Render CASE's cached realtime wake acknowledgement with local Piper."""

import argparse
import subprocess
import sys
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIPER = ROOT / "ai" / "tts" / "piper" / "piper"
DEFAULT_MODEL = ROOT / "ai" / "tts" / "en_US-ryan-medium.onnx"
DEFAULT_OUTPUT = ROOT / "assets" / "audio" / "realtime_wake_ack.wav"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="I'm listening.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--piper", type=Path, default=DEFAULT_PIPER)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    missing = [path for path in (args.piper, args.model) if not path.is_file()]
    if missing:
        print("Cannot generate wake acknowledgement; missing:")
        for path in missing:
            print(f"  - {path}")
        print("Install the existing CASE Piper runtime or provide --piper/--model.")
        raise SystemExit(1)

    result = subprocess.run(
        [str(args.piper), "--model", str(args.model), "--output_raw"],
        input=args.text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0 or not result.stdout:
        print(result.stderr.decode("utf-8", errors="replace"))
        raise SystemExit("Piper failed to synthesize wake acknowledgement")

    output = args.output.expanduser()
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22_050)
        wav_file.writeframes(result.stdout)
    print(f"Generated: {output}")


if __name__ == "__main__":
    main()
