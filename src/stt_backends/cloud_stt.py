"""Cloud STT provider interface for final CASE command transcription."""

from __future__ import annotations

import io
import logging
import time
import wave
from dataclasses import dataclass
from typing import Protocol

import numpy as np


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CloudSttResult:
    text: str
    provider: str
    latency_sec: float


class CloudSttProvider(Protocol):
    name: str

    def transcribe(self, waveform: np.ndarray, sample_rate: int) -> CloudSttResult:
        ...


def waveform_to_wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    samples = np.asarray(waveform, dtype="<i2").reshape(-1)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(samples.tobytes())
    return buffer.getvalue()


class GeminiCloudSttProvider:
    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        prompt: str | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini cloud STT")
        self.api_key = api_key
        self.model = model
        self.prompt = prompt or (
            "Transcribe this short user command audio exactly. "
            "The robot's name is CASE. The speaker may say CASE as a name. "
            "When the phrase refers to the robot, prefer 'you, CASE' or 'CASE' "
            "over 'UK case'. "
            "Common project terms include CASE, GTA 6, Grand Theft Auto 6, "
            "ESP32, Raspberry Pi, PCA9685, Piper, Vosk, and Gemini. "
            "Return only the spoken words as plain text. "
            "Do not explain, summarize, translate, or add punctuation unless obvious."
        )

    def transcribe(self, waveform: np.ndarray, sample_rate: int) -> CloudSttResult:
        started = time.monotonic()
        wav_bytes = waveform_to_wav_bytes(waveform, sample_rate)
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is not installed; cannot use Gemini cloud STT"
            ) from exc

        client = genai.Client(api_key=self.api_key)
        part = types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")
        response = client.models.generate_content(
            model=self.model,
            contents=[self.prompt, part],
        )
        text = str(getattr(response, "text", "") or "").strip()
        return CloudSttResult(
            text=text,
            provider=self.name,
            latency_sec=time.monotonic() - started,
        )


def build_cloud_stt_provider(
    provider: str,
    *,
    api_key: str,
    model: str,
) -> CloudSttProvider:
    normalized = str(provider or "").strip().lower()
    if normalized == "gemini":
        return GeminiCloudSttProvider(api_key=api_key, model=model)
    raise ValueError(f"unsupported cloud STT provider: {provider!r}")
