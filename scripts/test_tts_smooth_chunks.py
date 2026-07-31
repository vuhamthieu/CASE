#!/usr/bin/env python3
"""Compare CASE TTS chunk boundaries with smooth chunking on and off.

Examples:
  python3 scripts/test_tts_smooth_chunks.py
  python3 scripts/test_tts_smooth_chunks.py --save-wav
  python3 scripts/test_tts_smooth_chunks.py --text "One. Two short sentences. Three."
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import defaults
from src.realtime.response_chunker import ResponseChunker
from src.voice_pipeline.piper_onnx_backend import PiperOnnxSynthesizer


TEST_TEXT = (
    "AI is software that mimics human logic. "
    "LLMs are models trained on text to predict your next word. "
    "I am CASE, your field companion. "
    "Think of me as the guy who handles the data so you don't have to."
)


def chunk_text(text: str, *, smooth: bool) -> list[str]:
    chunker = ResponseChunker(
        max_chunks=8,
        max_total_chars=1000,
        smooth_chunks=smooth,
        first_chunk_fast=defaults.CASE_TTS_FIRST_CHUNK_FAST,
        max_sentences_per_chunk=defaults.CASE_TTS_MAX_SENTENCES_PER_CHUNK,
        max_chars_per_chunk=defaults.CASE_TTS_MAX_CHARS_PER_CHUNK,
        min_chars_to_group=defaults.CASE_TTS_MIN_CHARS_TO_GROUP,
        group_short_sentences=defaults.CASE_TTS_GROUP_SHORT_SENTENCES,
    )
    chunks = chunker.feed(text)
    chunks.extend(chunker.flush())
    return chunks


def print_chunks(label: str, chunks: list[str]) -> None:
    print(f"\n{label}")
    for index, chunk in enumerate(chunks):
        print(f"  seq={index}: chars={len(chunk)} text={chunk!r}")


def save_wavs(chunks: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    synthesizer = PiperOnnxSynthesizer(
        ROOT / defaults.PIPER_MODEL_PATH,
        ROOT / defaults.PIPER_CONFIG_PATH,
        length_scale=defaults.PIPER_LENGTH_SCALE,
        noise_scale=defaults.PIPER_NOISE_SCALE,
        noise_w=defaults.PIPER_NOISE_W,
    )
    synthesizer.load()
    for index, chunk in enumerate(chunks):
        audio, sample_rate = synthesizer.synthesize(chunk)
        path = output_dir / f"smooth_chunk_{index:02d}.wav"
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(audio)
        duration = len(audio) / 2 / sample_rate
        print(f"saved {path} duration={duration:.3f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", default=TEST_TEXT)
    parser.add_argument("--save-wav", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "output" / "tts_smooth_tests"),
    )
    args = parser.parse_args()

    old_chunks = chunk_text(args.text, smooth=False)
    smooth_chunks = chunk_text(args.text, smooth=True)
    print_chunks("Smooth chunking disabled", old_chunks)
    print_chunks("Smooth chunking enabled", smooth_chunks)

    if args.save_wav:
        save_wavs(smooth_chunks, Path(args.output_dir))


if __name__ == "__main__":
    main()
