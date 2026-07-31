"""Output backend contracts for CASE voice pipelines."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator


class VoiceOutputBackend(ABC):
    @abstractmethod
    async def speak(self, text: str, *, interruptible: bool = False) -> None:
        pass

    async def speak_stream(
        self,
        text_chunks: AsyncIterator[str],
        *,
        interruptible: bool = False,
    ) -> None:
        chunks = [chunk async for chunk in text_chunks]
        await self.speak("".join(chunks), interruptible=interruptible)


class LocalCaseTTSBackend(VoiceOutputBackend):
    """Publish through the existing CASEVoice local TTS event path."""

    def __init__(self, message_bus: Any) -> None:
        self.message_bus = message_bus

    async def speak(self, text: str, *, interruptible: bool = False) -> None:
        if text.strip():
            await self.message_bus.publish("AI_SPEAK", text.strip())


class CachedWavBackend(VoiceOutputBackend):
    """Reserved for fixed phrases; arbitrary text falls back to local TTS."""

    def __init__(self, fallback: LocalCaseTTSBackend) -> None:
        self.fallback = fallback

    async def speak(self, text: str, *, interruptible: bool = False) -> None:
        await self.fallback.speak(text, interruptible=interruptible)


class GeminiLiveNativeBackend(VoiceOutputBackend):
    """Marker backend: Gemini Live owns synthesis inside its active session."""

    async def speak(self, text: str, *, interruptible: bool = False) -> None:
        raise RuntimeError("Gemini Live native output is controlled by its session")


def create_voice_output_backend(name: str, message_bus: Any) -> VoiceOutputBackend:
    local = LocalCaseTTSBackend(message_bus)
    if name == "piper_onnx":
        from .piper_onnx_backend import PiperOnnxBackend

        return PiperOnnxBackend(message_bus)
    if name == "local_case_tts":
        return local
    if name == "cached_wav":
        return CachedWavBackend(local)
    if name == "gemini_live_native":
        return GeminiLiveNativeBackend()
    raise ValueError(f"unknown voice output backend: {name}")
