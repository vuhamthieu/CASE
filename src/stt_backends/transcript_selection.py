"""Select one final STT candidate and remove accidental repeated phrases."""

from __future__ import annotations

import re
from typing import Any


_WORD_RE = re.compile(r"\w+(?:['’]\w+)?", flags=re.UNICODE)


def normalize_transcript(text: str) -> str:
    """Normalize spacing while preserving readable case and punctuation."""
    return " ".join(str(text or "").strip().replace("’", "'").split())


def _words(text: str) -> list[str]:
    return [match.group(0).lower().replace("’", "'") for match in _WORD_RE.finditer(text)]


def _is_usable(text: str) -> bool:
    return bool(_words(text))


def dedupe_repeated_transcript(text: str) -> str:
    """Prefer the fuller side when a transcript contains a repeated candidate."""
    cleaned = normalize_transcript(text)
    matches = list(_WORD_RE.finditer(cleaned))
    if len(matches) < 2:
        return cleaned

    # Try every word boundary. This catches identical halves as well as a
    # shorter Vosk phrase followed by a fuller SenseVoice phrase.
    for split in range(1, len(matches)):
        right_start = matches[split].start()
        left_text = cleaned[:right_start].strip()
        right_text = cleaned[right_start:].strip()
        left_words = _words(left_text)
        right_words = _words(right_text)
        if not left_words or not right_words:
            continue
        if left_words == right_words:
            return normalize_transcript(right_text)
        if len(right_words) >= len(left_words) and right_words[-len(left_words) :] == left_words:
            return normalize_transcript(right_text)
        if len(left_words) > len(right_words) and left_words[-len(right_words) :] == right_words:
            return normalize_transcript(left_text)
    return cleaned


def choose_final_transcript(
    vosk_candidate: str,
    sensevoice_candidate: str,
    backend_status: dict[str, Any],
) -> str:
    """Choose SenseVoice final text, falling back to Vosk when necessary.

    ``backend_status`` is updated with ``selected_source`` for structured logs.
    """
    sensevoice_ok = (
        backend_status.get("sensevoice_available", True)
        and not backend_status.get("sensevoice_error")
    )
    sensevoice_text = dedupe_repeated_transcript(sensevoice_candidate)
    vosk_text = dedupe_repeated_transcript(vosk_candidate)

    if sensevoice_ok and _is_usable(sensevoice_text):
        backend_status["selected_source"] = "sensevoice"
        return sensevoice_text
    if _is_usable(vosk_text):
        backend_status["selected_source"] = "vosk_fallback"
        return vosk_text
    backend_status["selected_source"] = "none"
    return ""
