"""Sentence-only streaming text chunker for CASE's Piper TTS pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ResponseChunkerConfig:
    max_chunks: int = 4
    max_total_chars: int = 360
    chunk_mode: str = "sentence"


class ResponseChunker:
    """Convert streamed LLM deltas into complete sentence TTS chunks.

    The constructor keeps older size-policy arguments for compatibility, but
    sentence mode deliberately ignores them for emission decisions. Piper gets a
    chunk only when a full sentence is available, or when final flush has a
    punctuation-less remainder.
    """

    def __init__(
        self,
        *,
        min_chars: int = 35,
        max_chars: int = 110,
        absolute_max_chars: int = 160,
        first_chunk_target_chars: int = 40,
        first_chunk_max_chars: int = 55,
        normal_chunk_target_chars: int = 80,
        normal_chunk_max_chars: int = 110,
        flush_first_on_soft_punctuation: bool = True,
        min_first_chunk_chars: int = 16,
        max_chunks: int = 4,
        max_total_chars: int = 360,
        prefer_sentence_boundary: bool = True,
        merge_tiny_chunks: bool = True,
        tiny_chunk_max_chars: int = 25,
        single_chunk_under_chars: int = 130,
        chunk_mode: str = "sentence",
    ) -> None:
        self.config = ResponseChunkerConfig(
            max_chunks=max(1, int(max_chunks)),
            max_total_chars=max(1, int(max_total_chars)),
            chunk_mode=str(chunk_mode or "sentence").lower(),
        )
        self.buffer = ""
        self.emitted_count = 0
        self.emitted_chars = 0

    @property
    def exhausted(self) -> bool:
        return False

    def feed(self, delta: str) -> list[str]:
        if not delta or self.exhausted:
            return []
        self.buffer += delta
        return self._take_sentence_chunks(final=False)

    def flush(self) -> list[str]:
        if self.exhausted:
            self.buffer = ""
            return []
        return self._take_sentence_chunks(final=True)

    def _take_sentence_chunks(self, *, final: bool) -> list[str]:
        chunks: list[str] = []
        while not self.exhausted:
            split_at = self._sentence_split()
            if split_at is None:
                if final:
                    remainder = self._clean(self.buffer)
                    self.buffer = ""
                    if remainder:
                        self._record_emit(remainder)
                        chunks.append(remainder)
                break

            candidate = self._clean(self.buffer[:split_at])
            self.buffer = self.buffer[split_at:].lstrip()
            if not candidate:
                break
            self._record_emit(candidate)
            chunks.append(candidate)

        if self.exhausted:
            self.buffer = ""
        return chunks

    def _sentence_split(self) -> int | None:
        match = re.search(r"[.!?](?=\s|$)", self.buffer)
        if not match:
            return None
        return match.end()

    def _record_emit(self, text: str) -> None:
        self.emitted_count += 1
        self.emitted_chars += len(text)

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join(text.strip().split())
