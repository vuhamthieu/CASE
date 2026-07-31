"""JSON-backed Core Memory system for CASE."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CoreMemory:
    """Manages persistent core memory facts stored in a root-level JSON file."""

    def __init__(self, filepath: str | Path | None = None) -> None:
        self.file_path = Path(filepath) if filepath is not None else Path(__file__).resolve().parents[2] / "core_memory.json"
        self._ensure_file()

    def _ensure_file(self) -> None:
        """Create a blank JSON file if it does not exist, safely wrapping file IO."""
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.file_path.exists():
                logger.info("Initializing new core memory file at %s", self.file_path)
                with open(self.file_path, "w", encoding="utf-8") as f:
                    json.dump({}, f)
        except Exception as exc:
            logger.error("Failed to ensure core memory file exists: %s", exc, exc_info=True)

    def get_all(self) -> str:
        """Read core memory JSON and format key-value pairs as a bulleted string."""
        try:
            if not self.file_path.exists():
                return "- No core memories stored yet."
            
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if not isinstance(data, dict) or not data:
                return "- No core memories stored yet."
            
            return "\n".join(f"- {key}: {value}" for key, value in data.items())
        except Exception as exc:
            logger.error("Failed to read core memories: %s", exc, exc_info=True)
            return "- Memory temporarily unavailable due to a read error."

    def update_memory(self, key: str, value: str) -> str:
        """Update a key-value pair in core memory JSON and save with indentation."""
        try:
            self._ensure_file()
            
            data = {}
            if self.file_path.exists():
                try:
                    with open(self.file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    logger.warning("Core memory file was corrupted or empty. Resetting.")
                    data = {}
            
            data[key] = value
            
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            logger.info("Updated core memory: %s = %s", key, value)
            return f"Successfully updated core memory: '{key}' set to '{value}'."
        except Exception as exc:
            logger.error("Failed to update core memory for key '%s': %s", key, exc, exc_info=True)
            return f"Failed to update core memory: {exc}"


case_memory = CoreMemory()
