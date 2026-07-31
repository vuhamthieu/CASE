#!/usr/bin/env python3
"""Synthesize one WAV with CASE's persistent Piper ONNX backend."""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.voice_pipeline.piper_onnx_backend import PiperOnnxSynthesizer


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--noise-w", type=float, default=0.8)
    args = parser.parse_args()

    synthesizer = PiperOnnxSynthesizer(
        resolve(args.model),
        resolve(args.config),
        length_scale=args.length_scale,
        noise_scale=args.noise_scale,
        noise_w=args.noise_w,
    )
    try:
        audio, sample_rate = synthesizer.synthesize(args.text)
    except Exception as exc:
        raise SystemExit(f"PIPER_ONNX_TEST: failed: {exc}") from None
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(audio)
    print(
        f"PIPER_ONNX_TEST: output={output} sample_rate={sample_rate} "
        f"duration={len(audio) / (sample_rate * 2.0):.3f}s"
    )


if __name__ == "__main__":
    main()
