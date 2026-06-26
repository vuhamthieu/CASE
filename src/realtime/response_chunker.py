"""Streaming text chunker for CASE's Piper TTS pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


WEAK_TRAILING_WORDS = {
    "the",
    "a",
    "an",
    "to",
    "of",
    "and",
    "but",
    "or",
    "with",
    "in",
    "on",
    "your",
    "my",
    "is",
    "are",
}

DEPENDENT_START_WORDS = {
    "before",
    "after",
    "because",
    "while",
    "when",
    "if",
    "that",
    "which",
    "who",
    "where",
    "until",
}


@dataclass(frozen=True)
class ResponseChunkerConfig:
    min_chars: int = 35
    max_chars: int = 110
    absolute_max_chars: int = 160
    max_chunks: int = 4
    max_total_chars: int = 360
    prefer_sentence_boundary: bool = True
    merge_tiny_chunks: bool = True
    tiny_chunk_max_chars: int = 25
    single_chunk_under_chars: int = 130


class ResponseChunker:
    """Convert streamed LLM deltas into ordered TTS-safe text chunks."""

    def __init__(
        self,
        *,
        min_chars: int = 35,
        max_chars: int = 110,
        absolute_max_chars: int = 160,
        max_chunks: int = 4,
        max_total_chars: int = 360,
        prefer_sentence_boundary: bool = True,
        merge_tiny_chunks: bool = True,
        tiny_chunk_max_chars: int = 25,
        single_chunk_under_chars: int = 130,
    ) -> None:
        self.config = ResponseChunkerConfig(
            min_chars=max(1, int(min_chars)),
            max_chars=max(1, int(max_chars)),
            absolute_max_chars=max(1, int(absolute_max_chars)),
            max_chunks=max(1, int(max_chunks)),
            max_total_chars=max(1, int(max_total_chars)),
            prefer_sentence_boundary=bool(prefer_sentence_boundary),
            merge_tiny_chunks=bool(merge_tiny_chunks),
            tiny_chunk_max_chars=max(1, int(tiny_chunk_max_chars)),
            single_chunk_under_chars=max(1, int(single_chunk_under_chars)),
        )
        self.buffer = ""
        self.emitted_count = 0
        self.emitted_chars = 0

    @property
    def exhausted(self) -> bool:
        return (
            self.emitted_count >= self.config.max_chunks
            or self.emitted_chars >= self.config.max_total_chars
        )

    def feed(self, delta: str) -> list[str]:
        if not delta or self.exhausted:
            return []
        self.buffer += delta
        return self._take_ready_chunks(final=False)

    def flush(self) -> list[str]:
        if self.exhausted:
            self.buffer = ""
            return []
        return self._take_ready_chunks(final=True)

    def _take_ready_chunks(self, *, final: bool) -> list[str]:
        chunks: list[str] = []

        while not self.exhausted:
            if final and self.emitted_count == self.config.max_chunks - 1:
                split_at = len(self.buffer)
                candidate = self._clean(self.buffer)
                if not candidate:
                    break
                self.buffer = ""
                candidate = self._fit_total_budget(candidate)
                if not candidate:
                    break
                chunks.append(candidate)
                self.emitted_count += 1
                self.emitted_chars += len(candidate)
                break

            split_at = self._find_split(final=final)
            if split_at is None:
                break
            candidate = self._clean(self.buffer[:split_at])
            if (
                not final
                and self.emitted_count == self.config.max_chunks - 1
                and not candidate.endswith((".", "?", "!"))
            ):
                break
            if final and self._should_hold_short_response(candidate):
                split_at = len(self.buffer)
                candidate = self._clean(self.buffer)
            self.buffer = self.buffer[split_at:].lstrip()
            candidate = self._fit_total_budget(candidate)
            if not candidate:
                break
            chunks.append(candidate)
            self.emitted_count += 1
            self.emitted_chars += len(candidate)

        if self.exhausted:
            self.buffer = ""

        return chunks

    def _find_split(self, *, final: bool) -> Optional[int]:
        if not self.buffer.strip():
            return None

        sentence = self._sentence_split()
        if sentence is not None:
            return sentence

        soft = self._soft_punctuation_split()
        if soft is not None:
            return soft

        length = self._length_split()
        if length is not None:
            return length

        if final:
            cleaned = self._clean(self.buffer)
            if cleaned and cleaned.lower() not in {"i", "the", "a", "an"}:
                return len(self.buffer)
        return None

    def _sentence_split(self) -> Optional[int]:
        if not self.config.prefer_sentence_boundary:
            return None
        matches = list(re.finditer(r"[.!?](?=\s|$)", self.buffer))
        if not matches:
            return None
        chosen: Optional[int] = None
        for match in matches:
            split_at = match.end()
            candidate = self._clean(self.buffer[:split_at])
            if self.config.merge_tiny_chunks and len(candidate) <= self.config.tiny_chunk_max_chars:
                chosen = split_at
                continue
            if len(candidate) < self.config.min_chars and self.emitted_count > 0:
                chosen = split_at
                continue
            if (
                self.config.merge_tiny_chunks
                and len(candidate) < self.config.single_chunk_under_chars
                and self._remaining_can_complete_short_response(split_at)
            ):
                chosen = split_at
                continue
            if chosen is not None and len(candidate) <= self.config.absolute_max_chars:
                return split_at
            return split_at
        if chosen is not None and len(self.buffer) > self.config.absolute_max_chars:
            return chosen
        return chosen if self.exhausted else None

    def _soft_punctuation_split(self) -> Optional[int]:
        if len(self.buffer) < self.config.max_chars:
            return None
        if self._buffer_is_complete_short_response():
            return None
        for match in re.finditer(r"[,;:](?=\s|$)", self.buffer):
            split_at = match.end()
            candidate = self._clean(self.buffer[:split_at])
            if len(candidate) < self.config.min_chars:
                continue
            if len(candidate) > self.config.absolute_max_chars:
                continue
            if self._ends_with_weak_word(candidate):
                continue
            if self._next_starts_dependent(split_at):
                continue
            return split_at
        return None

    def _length_split(self) -> Optional[int]:
        if len(self.buffer) < self.config.absolute_max_chars:
            return None
        if self._buffer_is_complete_short_response():
            return None

        limit = self.config.absolute_max_chars
        for match in reversed(list(re.finditer(r"\s+", self.buffer[: limit + 1]))):
            split_at = match.start()
            if split_at < self.config.min_chars:
                break
            candidate = self._clean(self.buffer[:split_at])
            if (
                candidate
                and not self._ends_with_weak_word(candidate)
                and not self._next_starts_dependent(split_at)
            ):
                return split_at

        if len(self.buffer) >= self.config.absolute_max_chars:
            return self.config.absolute_max_chars
        return None

    def _fit_total_budget(self, text: str) -> str:
        remaining = self.config.max_total_chars - self.emitted_chars
        if remaining <= 0:
            return ""
        if len(text) <= remaining:
            return text

        clipped = text[:remaining].rstrip()
        split_at = max(
            clipped.rfind("."),
            clipped.rfind("?"),
            clipped.rfind("!"),
            clipped.rfind(","),
            clipped.rfind(";"),
            clipped.rfind(" "),
        )
        if split_at >= self.config.min_chars:
            clipped = clipped[: split_at + 1].rstrip()
        return clipped

    @staticmethod
    def _ends_with_weak_word(text: str) -> bool:
        words = re.findall(r"[a-zA-Z']+", text.lower())
        if not words:
            return False
        return words[-1].strip("'") in WEAK_TRAILING_WORDS

    def _next_starts_dependent(self, split_at: int) -> bool:
        remainder = self.buffer[split_at:].lstrip()
        match = re.match(r"([a-zA-Z']+)", remainder)
        if not match:
            return False
        return match.group(1).strip("'").lower() in DEPENDENT_START_WORDS

    def _remaining_can_complete_short_response(self, split_at: int) -> bool:
        remainder = self.buffer[split_at:].lstrip()
        if not remainder:
            return True
        combined = self._clean(self.buffer)
        if len(combined) > self.config.single_chunk_under_chars:
            return False
        return bool(re.search(r"[.!?](?=\s|$)", remainder))

    def _should_hold_short_response(self, candidate: str) -> bool:
        if not self.config.merge_tiny_chunks:
            return False
        full = self._clean(self.buffer)
        return (
            candidate != full
            and len(full) <= self.config.single_chunk_under_chars
            and bool(re.search(r"[.!?](?=\s|$)", full))
        )

    def _buffer_is_complete_short_response(self) -> bool:
        return (
            self.config.merge_tiny_chunks
            and len(self._clean(self.buffer)) <= self.config.single_chunk_under_chars
            and bool(re.search(r"[.!?](?=\s|$)", self.buffer))
        )

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join(text.strip().split())
