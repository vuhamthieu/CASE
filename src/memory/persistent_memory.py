"""Tiny SQLite-backed summary memory for CASE."""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

from .memory_types import MemoryItem, TurnRecord


class PersistentMemory:
    """Persist concise memories without storing raw full conversations forever."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        if not self.path.is_absolute():
            self.path = Path(__file__).resolve().parents[2] / self.path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init_db(self) -> None:
        with self._connect() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    key TEXT NOT NULL DEFAULT '',
                    text TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS turn_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_summary TEXT NOT NULL,
                    assistant_summary TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    def add_item(self, item: MemoryItem) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO memory_items(kind, key, text, created_at) VALUES (?, ?, ?, ?)",
                (item.kind, item.key, item.text, item.timestamp),
            )

    def add_turn_summary(self, record: TurnRecord) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO turn_summaries(user_summary, assistant_summary, created_at) "
                "VALUES (?, ?, ?)",
                (
                    self._summarize(record.user_text),
                    self._summarize(record.assistant_text),
                    record.timestamp,
                ),
            )

    def recent_items(self, kind: str, limit: int = 10) -> list[MemoryItem]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT kind, key, text, created_at FROM memory_items "
                "WHERE kind = ? ORDER BY created_at DESC LIMIT ?",
                (kind, int(limit)),
            ).fetchall()
        return [
            MemoryItem(kind=row[0], key=row[1], text=row[2], timestamp=float(row[3]))
            for row in rows
        ]

    def recent_turn_summaries(self, limit: int = 8) -> list[TurnRecord]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT user_summary, assistant_summary, created_at FROM turn_summaries "
                "ORDER BY created_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [
            TurnRecord(user_text=row[0], assistant_text=row[1], timestamp=float(row[2]))
            for row in rows
        ]

    @staticmethod
    def _summarize(text: str, max_chars: int = 240) -> str:
        cleaned = " ".join(str(text).split())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[: max_chars - 1].rstrip() + "..."
