"""Transcript input backend boundary for CASE voice pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Awaitable, Callable


class TranscriptInputBackend(ABC):
    @abstractmethod
    async def listen_once(self, timeout_sec: float) -> str:
        pass


class LocalVoskTranscriptBackend(TranscriptInputBackend):
    """Adapter for an existing asynchronous local-Vosk listen callback."""

    def __init__(self, listener: Callable[[float], Awaitable[str]]) -> None:
        self.listener = listener

    async def listen_once(self, timeout_sec: float) -> str:
        return (await self.listener(timeout_sec)).strip()


class GeminiLiveTranscriptionResearchBackend(TranscriptInputBackend):
    async def listen_once(self, timeout_sec: float) -> str:
        raise RuntimeError(
            "gemini_live_transcription_research is not enabled in hybrid v1"
        )
