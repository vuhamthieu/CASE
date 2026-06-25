"""Persistent Piper ONNX synthesis for CASE's copied local voice."""

from __future__ import annotations

import io
import json
import logging
import wave
from pathlib import Path
from typing import Any

from .voice_backend import VoiceOutputBackend


logger = logging.getLogger(__name__)


class PiperOnnxSynthesizer:
    """Load one Piper voice once and return mono signed-16 PCM."""

    def __init__(
        self,
        model_path: str | Path,
        config_path: str | Path,
        *,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.config_path = Path(config_path).expanduser().resolve()
        self.length_scale = float(length_scale)
        self.noise_scale = float(noise_scale)
        self.noise_w = float(noise_w)
        self.voice = None
        self.synthesis_config = None
        self._style_warning_logged = False
        self.sample_rate = self._read_sample_rate()

    def load(self) -> None:
        if self.voice is not None:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Piper ONNX model missing: {self.model_path}")
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Piper config missing: {self.config_path}")
        logger.info("PIPER_ONNX: model=%s", self.model_path)
        logger.info("PIPER_ONNX: config=%s", self.config_path)
        try:
            from piper.voice import PiperVoice
        except ImportError as exc:
            raise RuntimeError(
                "piper-tts Python package is required for persistent Piper ONNX "
                "inference; install requirements.txt"
            ) from exc

        self.voice = PiperVoice.load(
            str(self.model_path),
            config_path=str(self.config_path),
            use_cuda=False,
        )
        voice_rate = getattr(getattr(self.voice, "config", None), "sample_rate", None)
        if voice_rate:
            self.sample_rate = int(voice_rate)
        try:
            from piper.config import SynthesisConfig

            self.synthesis_config = SynthesisConfig(
                length_scale=self.length_scale,
                noise_scale=self.noise_scale,
                noise_w_scale=self.noise_w,
            )
        except (ImportError, TypeError):
            self.synthesis_config = None
        logger.info("PIPER_ONNX: loaded sample_rate=%s", self.sample_rate)

    def synthesize(
        self,
        text: str,
        *,
        length_scale: float | None = None,
        noise_scale: float | None = None,
        noise_w: float | None = None,
    ) -> tuple[bytes, int]:
        self.load()
        assert self.voice is not None
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Piper text is empty")

        synthesis_config = self._synthesis_config_for(
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w=noise_w,
        )
        audio = self._synthesize_chunks(cleaned, synthesis_config)
        if not audio:
            audio = self._synthesize_wav(cleaned, synthesis_config)
        if not audio:
            raise RuntimeError("Piper ONNX produced no audio")
        return audio, self.sample_rate

    def _synthesis_config_for(
        self,
        *,
        length_scale: float | None,
        noise_scale: float | None,
        noise_w: float | None,
    ):
        if all(value is None for value in (length_scale, noise_scale, noise_w)):
            return self.synthesis_config
        try:
            from piper.config import SynthesisConfig

            return SynthesisConfig(
                length_scale=self.length_scale if length_scale is None else length_scale,
                noise_scale=self.noise_scale if noise_scale is None else noise_scale,
                noise_w_scale=self.noise_w if noise_w is None else noise_w,
            )
        except (ImportError, TypeError):
            if not self._style_warning_logged:
                logger.warning(
                    "PIPER_ONNX: per-utterance synthesis parameters unavailable; "
                    "using loaded defaults"
                )
                self._style_warning_logged = True
            return self.synthesis_config

    def _synthesize_chunks(self, text: str, synthesis_config) -> bytes:
        kwargs: dict[str, Any] = {}
        if synthesis_config is not None:
            kwargs["syn_config"] = synthesis_config
        try:
            chunks = self.voice.synthesize(text, **kwargs)
        except (AttributeError, TypeError):
            return b""
        if chunks is None:
            return b""
        output = bytearray()
        try:
            for chunk in chunks:
                payload = getattr(chunk, "audio_int16_bytes", None)
                if payload is None and isinstance(chunk, (bytes, bytearray)):
                    payload = bytes(chunk)
                if payload:
                    output.extend(payload)
        except TypeError:
            return b""
        return bytes(output)

    def _synthesize_wav(self, text: str, synthesis_config) -> bytes:
        synthesize_wav = getattr(self.voice, "synthesize_wav", None)
        if synthesize_wav is None:
            return b""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as output:
            try:
                if synthesis_config is not None:
                    synthesize_wav(text, output, syn_config=synthesis_config)
                else:
                    synthesize_wav(
                        text,
                        output,
                        length_scale=self.length_scale,
                        noise_scale=self.noise_scale,
                        noise_w=self.noise_w,
                    )
            except TypeError:
                synthesize_wav(text, output)
        buffer.seek(0)
        with wave.open(buffer, "rb") as source:
            self.sample_rate = source.getframerate()
            if source.getnchannels() != 1 or source.getsampwidth() != 2:
                raise RuntimeError("Piper ONNX returned unsupported WAV format")
            return source.readframes(source.getnframes())

    def _read_sample_rate(self) -> int:
        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
            return int(data["audio"]["sample_rate"])
        except Exception:
            return 22_050


class PiperOnnxBackend(VoiceOutputBackend):
    """Message-bus adapter; CASEVoice owns synthesis and playback."""

    def __init__(self, message_bus: Any) -> None:
        self.message_bus = message_bus

    async def speak(self, text: str, *, interruptible: bool = False) -> None:
        if text.strip():
            await self.message_bus.publish("AI_SPEAK", text.strip())
