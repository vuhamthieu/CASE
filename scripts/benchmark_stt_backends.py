#!/usr/bin/env python3
"""Benchmark offline CASE STT backends against the same recorded WAV corpus."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import defaults
from src.stt_backends.sherpa_sensevoice_backend import SherpaSenseVoiceBackend


def normalize(text: str) -> list[str]:
    return "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()


def word_error_rate(expected: str, actual: str) -> float:
    reference = normalize(expected)
    hypothesis = normalize(actual)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    previous = list(range(len(hypothesis) + 1))
    for row, expected_word in enumerate(reference, start=1):
        current = [row]
        for column, actual_word in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected_word != actual_word),
                )
            )
        previous = current
    return previous[-1] / len(reference)


def load_audio(path: Path) -> tuple[int, np.ndarray]:
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    rate, audio = wavfile.read(path)
    if audio.ndim > 1:
        audio = np.rint(audio.astype(np.float32).mean(axis=1)).astype(np.int16)
    if audio.dtype != np.int16:
        if np.issubdtype(audio.dtype, np.floating):
            audio = np.clip(audio, -1.0, 1.0) * 32767.0
        audio = np.clip(np.rint(audio), -32768, 32767).astype(np.int16)
    if rate != 16_000:
        audio = resample_poly(audio.astype(np.float32), 16_000, rate)
        audio = np.clip(np.rint(audio), -32768, 32767).astype(np.int16)
        rate = 16_000
    return rate, audio


class VoskBenchmark:
    def __init__(self, model_path: Path) -> None:
        from vosk import Model

        if not model_path.is_dir():
            raise FileNotFoundError(model_path)
        self.model = Model(str(model_path))

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        from vosk import KaldiRecognizer

        recognizer = KaldiRecognizer(self.model, sample_rate)
        payload = audio.astype("<i2").tobytes()
        for offset in range(0, len(payload), 8000):
            recognizer.AcceptWaveform(payload[offset : offset + 8000])
        return json.loads(recognizer.FinalResult()).get("text", "").strip()


def create_backend(name: str):
    if name == "vosk_lgraph":
        configured = ROOT / defaults.VOSK_LGRAPH_MODEL_PATH
        legacy = ROOT / Path(defaults.VOSK_LGRAPH_MODEL_PATH).name
        return VoskBenchmark(configured if configured.is_dir() else legacy)
    if name == "vosk_small":
        return VoskBenchmark(ROOT / defaults.VOSK_SMALL_MODEL_PATH)
    if name == "sherpa_sensevoice":
        return SherpaSenseVoiceBackend(ROOT / defaults.SHERPA_SENSEVOICE_MODEL_DIR)
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("vosk_lgraph", "vosk_small", "sherpa_sensevoice"),
        required=True,
    )
    args = parser.parse_args()
    audio_dir = args.audio_dir if args.audio_dir.is_absolute() else ROOT / args.audio_dir
    wavs = sorted(audio_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(
            f"No WAV files found under {audio_dir}.\n"
            "Record the fixed test corpus first:\n"
            "  python3 scripts/record_stt_test_phrases.py "
            f"--output-dir {args.audio_dir}"
        )

    for backend_name in args.backends:
        try:
            backend = create_backend(backend_name)
        except Exception as exc:
            print(f"\nBACKEND {backend_name}: unavailable: {exc}")
            continue
        print(f"\nBACKEND {backend_name}")
        for wav_path in wavs:
            expected_path = wav_path.with_suffix(".txt")
            expected = expected_path.read_text(encoding="utf-8").strip() if expected_path.is_file() else ""
            rate, audio = load_audio(wav_path)
            started = time.monotonic()
            transcript = backend.transcribe(audio, rate)
            latency = time.monotonic() - started
            duration = len(audio) / rate
            wer = word_error_rate(expected, transcript)
            print(
                f"{wav_path.name}: expected={expected!r} transcript={transcript!r} "
                f"latency={latency:.3f}s WER={wer:.3f} "
                f"RTF={latency / duration if duration else 0.0:.3f}"
            )


if __name__ == "__main__":
    main()
