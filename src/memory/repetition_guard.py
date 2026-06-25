"""Simple joke/roast repetition guard for CASE."""

from __future__ import annotations

import hashlib
import logging
import re
from collections import deque


logger = logging.getLogger(__name__)


JOKE_FALLBACKS = [
    "I asked my diagnostics for a joke. It returned your wiring diagram.",
    "My task scheduler walked into a bar. It waited for a mutex and left.",
    "I told my battery to stay positive. It filed a voltage complaint.",
]

ROAST_FALLBACKS = [
    "You gave a Pi 4 a personality, then got surprised when it developed opinions. Bold engineering.",
    "I would roast you, but your power budget already did most of the work.",
    "Your cable management has the confidence of experimental firmware.",
]


class RepetitionGuard:
    def __init__(self, *, limit: int = 20, similarity_threshold: float = 0.72) -> None:
        self.limit = max(1, int(limit))
        self.similarity_threshold = float(similarity_threshold)
        self._recent: dict[str, deque[str]] = {
            "joke": deque(maxlen=self.limit),
            "roast": deque(maxlen=self.limit),
        }
        self._fallback_index = {"joke": 0, "roast": 0}

    def remember(self, kind: str, text: str) -> None:
        normalized = self.normalize(text)
        if normalized:
            self._recent.setdefault(kind, deque(maxlen=self.limit)).append(normalized)

    def check(self, kind: str, text: str) -> tuple[bool, str | None]:
        normalized = self.normalize(text)
        if not normalized:
            logger.info("REPETITION_GUARD: checked kind=%s similar=False", kind)
            return False, None
        for previous in self._recent.setdefault(kind, deque(maxlen=self.limit)):
            if normalized == previous:
                logger.info(
                    "REPETITION_GUARD: rejected kind=%s reason=exact_duplicate",
                    kind,
                )
                return True, "exact_duplicate"
            if self.jaccard(normalized, previous) >= self.similarity_threshold:
                logger.info(
                    "REPETITION_GUARD: rejected kind=%s reason=near_duplicate",
                    kind,
                )
                return True, "near_duplicate"
        logger.info("REPETITION_GUARD: checked kind=%s similar=False", kind)
        return False, None

    def replacement(self, kind: str) -> str:
        pool = ROAST_FALLBACKS if kind == "roast" else JOKE_FALLBACKS
        index = self._fallback_index.get(kind, 0) % len(pool)
        self._fallback_index[kind] = index + 1
        return pool[index]

    def recent(self, kind: str) -> list[str]:
        return list(self._recent.setdefault(kind, deque(maxlen=self.limit)))

    @staticmethod
    def normalize(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9']+", str(text).lower()))

    @staticmethod
    def digest(text: str) -> str:
        return hashlib.sha256(RepetitionGuard.normalize(text).encode("utf-8")).hexdigest()

    @staticmethod
    def jaccard(left: str, right: str) -> float:
        left_tokens = set(left.split())
        right_tokens = set(right.split())
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
