"""Small ANSI console transcript and dual-destination logging setup."""

from __future__ import annotations

import logging
import sys
import threading
from datetime import datetime
from pathlib import Path

from src.config import defaults
from src.config.env import get_bool, get_str


CASE_CONSOLE_MODE = get_str("CASE_CONSOLE_MODE", defaults.CASE_CONSOLE_MODE).lower()
CASE_CONSOLE_CLEAN_TRANSCRIPT = get_bool(
    "CASE_CONSOLE_CLEAN_TRANSCRIPT", defaults.CASE_CONSOLE_CLEAN_TRANSCRIPT
)
CASE_DEBUG_LOG_FILE = get_str("CASE_DEBUG_LOG_FILE", defaults.CASE_DEBUG_LOG_FILE)
CASE_CONSOLE_LOG_LEVEL = get_str(
    "CASE_CONSOLE_LOG_LEVEL", defaults.CASE_CONSOLE_LOG_LEVEL
)
CASE_FILE_LOG_LEVEL = get_str("CASE_FILE_LOG_LEVEL", defaults.CASE_FILE_LOG_LEVEL)


COLORS = {
    "YOU": "\033[96m",
    "CASE": "\033[92m",
    "SYS": "\033[2;37m",
    "VIS": "\033[95m",
    "WARN": "\033[93m",
    "ERR": "\033[91m",
}
RESET = "\033[0m"


class ConsoleTranscript:
    """Print stable, one-line human-readable CASE events."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def write(self, channel: str, text: str) -> None:
        if not CASE_CONSOLE_CLEAN_TRANSCRIPT or CASE_CONSOLE_MODE == "debug":
            return
        channel = channel.upper()
        clean = " ".join(str(text).split())
        if not clean:
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = COLORS.get(channel, COLORS["SYS"])
        with self._lock:
            print(
                f"{color}[{timestamp}] {channel:<4} > {clean}{RESET}",
                file=sys.stdout,
                flush=True,
            )

    def you(self, text: str) -> None:
        self.write("YOU", text)

    def case(self, text: str) -> None:
        self.write("CASE", text)

    def system(self, text: str) -> None:
        self.write("SYS", text)

    def vision(self, text: str) -> None:
        self.write("VIS", text)

    def warning(self, text: str) -> None:
        self.write("WARN", text)

    def error(self, text: str) -> None:
        self.write("ERR", text)


console = ConsoleTranscript()


class _LevelColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if record.levelno >= logging.ERROR:
            color = COLORS["ERR"]
        elif record.levelno >= logging.WARNING:
            color = COLORS["WARN"]
        else:
            color = COLORS["SYS"]
        return f"{color}{super().format(record)}{RESET}"


def configure_case_logging(project_root: Path) -> Path:
    """Send full diagnostics to a file and selected records to stderr."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    debug_path = Path(CASE_DEBUG_LOG_FILE).expanduser()
    if not debug_path.is_absolute():
        debug_path = project_root / debug_path
    debug_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(debug_path, encoding="utf-8")
    file_handler.setLevel(getattr(logging, CASE_FILE_LOG_LEVEL.upper(), logging.DEBUG))
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(threadName)s %(message)s"
        )
    )
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    if CASE_CONSOLE_MODE == "debug":
        console_level = logging.INFO
    else:
        console_level = getattr(
            logging,
            CASE_CONSOLE_LOG_LEVEL.upper(),
            logging.WARNING,
        )
    console_handler.setLevel(console_level)
    console_handler.setFormatter(
        _LevelColorFormatter("%(asctime)s %(levelname)s %(message)s")
    )
    root.addHandler(console_handler)
    return debug_path
