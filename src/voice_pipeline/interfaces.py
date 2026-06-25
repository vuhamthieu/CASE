"""Minimal asynchronous contracts for swappable voice providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass(frozen=True)
class VoicePipelineEvent:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)


class StreamingSTTProvider(ABC):
    @abstractmethod
    async def start(self) -> None:
        pass

    @abstractmethod
    async def stop(self) -> None:
        pass

    @abstractmethod
    async def audio_in(self, pcm_16k: bytes) -> None:
        pass

    @abstractmethod
    async def transcript_events(self) -> AsyncIterator[VoicePipelineEvent]:
        pass


class StreamingLLMProvider(ABC):
    @abstractmethod
    async def stream_reply(
        self,
        text: str,
        context: dict[str, Any],
    ) -> AsyncIterator[str]:
        pass


class StreamingTTSProvider(ABC):
    @abstractmethod
    async def stream_audio(
        self, text_stream: AsyncIterator[str]
    ) -> AsyncIterator[bytes]:
        pass
