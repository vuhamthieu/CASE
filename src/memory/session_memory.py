"""In-process short-term memory for the current CASE runtime session."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Iterable

from .memory_types import TurnRecord


_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "is",
    "are",
    "you",
    "me",
    "my",
    "your",
    "case",
}


@dataclass(frozen=True)
class MemoryContext:
    recent_turns: list[TurnRecord]
    recent_topics: list[str]
    recent_jokes: list[str]
    recent_roasts: list[str]
    user_facts: list[str]
    user_preferences: list[str]


class SessionMemory:
    """Bounded memory that survives only for the running process."""

    def __init__(
        self,
        *,
        history_turns: int = 8,
        recent_topics_limit: int = 20,
        recent_jokes_limit: int = 20,
    ) -> None:
        self.recent_turns: deque[TurnRecord] = deque(maxlen=max(1, history_turns))
        self.recent_topics: deque[str] = deque(maxlen=max(1, recent_topics_limit))
        self.recent_jokes: deque[str] = deque(maxlen=max(1, recent_jokes_limit))
        self.recent_roasts: deque[str] = deque(maxlen=max(1, recent_jokes_limit))
        self.user_facts: deque[str] = deque(maxlen=20)
        self.user_preferences: deque[str] = deque(maxlen=20)

    def add_turn(self, user_text: str, assistant_text: str) -> TurnRecord:
        record = TurnRecord(user_text=user_text.strip(), assistant_text=assistant_text.strip())
        self.recent_turns.append(record)
        for topic in self._extract_topics(user_text):
            if topic not in self.recent_topics:
                self.recent_topics.append(topic)
        return record

    def add_joke(self, text: str) -> None:
        cleaned = text.strip()
        if cleaned:
            self.recent_jokes.append(cleaned)

    def add_roast(self, text: str) -> None:
        cleaned = text.strip()
        if cleaned:
            self.recent_roasts.append(cleaned)

    def context(self) -> MemoryContext:
        return MemoryContext(
            recent_turns=list(self.recent_turns),
            recent_topics=list(self.recent_topics),
            recent_jokes=list(self.recent_jokes),
            recent_roasts=list(self.recent_roasts),
            user_facts=list(self.user_facts),
            user_preferences=list(self.user_preferences),
        )

    @staticmethod
    def _extract_topics(text: str) -> Iterable[str]:
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9']+", text.lower())
        for word in words:
            if len(word) >= 4 and word not in _STOPWORDS:
                yield word
