"""Offline sherpa-onnx SenseVoice transcription backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class SherpaSenseVoiceBackend:
    def __init__(self, model_dir: str | Path, num_threads: int = 2) -> None:
        directory = Path(model_dir).expanduser().resolve()
        model = directory / "model.int8.onnx"
        if not model.is_file():
            model = directory / "model.onnx"
        tokens = directory / "tokens.txt"
        if not model.is_file() or not tokens.is_file():
            raise FileNotFoundError(
                f"SenseVoice model/tokens missing under: {directory}"
            )
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError("sherpa-onnx is not installed") from exc
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model),
            tokens=str(tokens),
            num_threads=num_threads,
            use_itn=True,
            debug=False,
        )

    def transcribe(self, pcm16: np.ndarray, sample_rate: int = 16_000) -> str:
        samples = np.asarray(pcm16, dtype=np.int16).reshape(-1).astype(np.float32)
        samples /= 32768.0
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self.recognizer.decode_stream(stream)
        return str(stream.result.text).strip()
