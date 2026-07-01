"""Sentence-only streaming text chunker for CASE's Piper TTS pipeline."""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResponseChunkerConfig:
    max_chunks: int = 4
    max_total_chars: int = 360
    chunk_mode: str = "sentence"
    smooth_chunks: bool = True
    first_chunk_fast: bool = True
    max_sentences_per_chunk: int = 2
    max_chars_per_chunk: int = 170
    min_chars_to_group: int = 45
    group_short_sentences: bool = True


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
        smooth_chunks: bool = True,
        first_chunk_fast: bool = True,
        max_sentences_per_chunk: int = 2,
        max_chars_per_chunk: int = 170,
        min_chars_to_group: int = 45,
        group_short_sentences: bool = True,
        turn_id: int | None = None,
    ) -> None:
        self.config = ResponseChunkerConfig(
            max_chunks=max(1, int(max_chunks)),
            max_total_chars=max(1, int(max_total_chars)),
            chunk_mode=str(chunk_mode or "sentence").lower(),
            smooth_chunks=bool(smooth_chunks),
            first_chunk_fast=bool(first_chunk_fast),
            max_sentences_per_chunk=max(1, int(max_sentences_per_chunk)),
            max_chars_per_chunk=max(1, int(max_chars_per_chunk)),
            min_chars_to_group=max(0, int(min_chars_to_group)),
            group_short_sentences=bool(group_short_sentences),
        )
        self.buffer = ""
        self.emitted_count = 0
        self.emitted_chars = 0
        self.turn_id = turn_id
        self._pending_sentences: list[str] = []
        logger.info(
            "TTS_SMOOTH_CHUNK_MODE: enabled=%s first_chunk_fast=%s "
            "max_sentences=%s max_chars=%s",
            self.config.smooth_chunks,
            self.config.first_chunk_fast,
            self.config.max_sentences_per_chunk,
            self.config.max_chars_per_chunk,
        )

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
                        chunks.extend(self._accept_sentence(remainder, final=True))
                break

            candidate = self._clean(self.buffer[:split_at])
            self.buffer = self.buffer[split_at:].lstrip()
            if not candidate:
                break
            chunks.extend(self._accept_sentence(candidate, final=final))

        if self.exhausted:
            self.buffer = ""
        if final:
            chunks.extend(self._flush_pending())
        return chunks

    def _accept_sentence(self, sentence: str, *, final: bool) -> list[str]:
        if not self.config.smooth_chunks:
            self._record_emit(sentence)
            return [sentence]

        if self.emitted_count == 0 and self.config.first_chunk_fast:
            self._record_emit(sentence)
            logger.info(
                "TTS_CHUNK_FAST_FIRST: turn=%s seq=0 chars=%s",
                self.turn_id,
                len(sentence),
            )
            return [sentence]

        if not self.config.group_short_sentences:
            self._record_emit(sentence)
            return [sentence]

        if len(sentence) > self.config.max_chars_per_chunk:
            flushed = self._flush_pending()
            self._record_emit(sentence)
            return [*flushed, sentence]

        if not self._pending_sentences:
            if final:
                self._record_emit(sentence)
                return [sentence]
            self._pending_sentences.append(sentence)
            return []

        candidate_sentences = [*self._pending_sentences, sentence]
        combined = " ".join(candidate_sentences)
        if (
            len(candidate_sentences) <= self.config.max_sentences_per_chunk
            and len(combined) <= self.config.max_chars_per_chunk
        ):
            self._pending_sentences = []
            self._record_emit(combined)
            logger.info(
                "TTS_CHUNK_GROUPED: turn=%s seq=%s sentences=%s chars=%s",
                self.turn_id,
                self.emitted_count - 1,
                len(candidate_sentences),
                len(combined),
            )
            return [combined]

        flushed = self._flush_pending()
        if final:
            self._record_emit(sentence)
            return [*flushed, sentence]
        self._pending_sentences.append(sentence)
        return flushed

    def _flush_pending(self) -> list[str]:
        if not self._pending_sentences:
            return []
        text = " ".join(self._pending_sentences)
        self._pending_sentences = []
        self._record_emit(text)
        return [text]

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
