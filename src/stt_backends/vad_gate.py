"""Optional sherpa-onnx Silero VAD and GTCRN enhancement wrappers."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)


class SileroVadGate:
    def __init__(
        self,
        model_path: str | Path,
        *,
        sample_rate: int = 16_000,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 700,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.sample_rate = sample_rate
        self.enabled = False
        self._buffer = np.empty(0, dtype=np.float32)
        self._vad = None
        self._window_size = 512
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Silero VAD model missing: {self.model_path}")
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError("sherpa-onnx is not installed") from exc

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(self.model_path)
        config.silero_vad.threshold = float(threshold)
        config.silero_vad.min_speech_duration = min_speech_ms / 1000.0
        config.silero_vad.min_silence_duration = min_silence_ms / 1000.0
        config.sample_rate = sample_rate
        self._window_size = int(config.silero_vad.window_size)
        self._vad = sherpa_onnx.VoiceActivityDetector(
            config,
            buffer_size_in_seconds=30,
        )
        self.enabled = True

    def is_speech(self, pcm16: bytes) -> bool | None:
        """Return current Silero speech state, or None if API cannot expose it."""
        if not self.enabled or self._vad is None:
            return None
        samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        self._buffer = np.concatenate((self._buffer, samples))
        while len(self._buffer) >= self._window_size:
            self._vad.accept_waveform(self._buffer[: self._window_size])
            self._buffer = self._buffer[self._window_size :]
        state = getattr(self._vad, "is_speech_detected", None)
        if callable(state):
            return bool(state())
        if state is not None:
            return bool(state)
        return None

    def reset(self) -> None:
        self._buffer = np.empty(0, dtype=np.float32)
        reset = getattr(self._vad, "reset", None)
        if callable(reset):
            reset()


class GtcrnDenoiser:
    """Optional GTCRN wrapper; disabled by default because it costs Pi CPU."""

    def __init__(self, model_path: str | Path, sample_rate: int = 16_000) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.sample_rate = sample_rate
        self.enabled = False
        self._denoiser = None
        if not self.model_path.is_file():
            raise FileNotFoundError(f"GTCRN model missing: {self.model_path}")
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError("sherpa-onnx is not installed") from exc
        try:
            config = sherpa_onnx.OfflineSpeechDenoiserConfig()
            config.model.gtcrn.model = str(self.model_path)
            config.model.num_threads = 1
            self._denoiser = sherpa_onnx.OfflineSpeechDenoiser(config)
        except Exception as exc:
            raise RuntimeError(f"sherpa-onnx GTCRN API unavailable: {exc}") from exc
        self.enabled = True

    def process(self, pcm16: bytes) -> tuple[bytes, float]:
        if not self.enabled or self._denoiser is None:
            return pcm16, 0.0
        started = time.monotonic()
        samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        result = self._denoiser.run(samples, self.sample_rate)
        enhanced = np.asarray(result.samples, dtype=np.float32)
        output = np.clip(np.rint(enhanced * 32767.0), -32768, 32767).astype("<i2")
        return output.tobytes(), time.monotonic() - started
