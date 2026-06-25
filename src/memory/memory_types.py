"""Small typed records for CASE memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time


@dataclass(frozen=True)
class TurnRecord:
    user_text: str
    assistant_text: str
    timestamp: float = field(default_factory=time)


@dataclass(frozen=True)
class MemoryItem:
    kind: str
    text: str
    key: str = ""
    timestamp: float = field(default_factory=time)
