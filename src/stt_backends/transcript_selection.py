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
    words = _words(text)
    if not words:
        return False
    alpha_count = sum(ch.isalpha() for ch in text)
    if alpha_count < 2:
        return False
    if len(words) >= 3 and len(set(words)) == 1:
        return False
    return True


def is_usable_transcript(text: str) -> bool:
    return _is_usable(text)


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
    *,
    lgraph_candidate: str = "",
) -> str:
    """Choose a final transcript in profile order, falling back when necessary.

    ``backend_status`` is updated with ``selected_source`` for structured logs.
    """
    final_chain = tuple(backend_status.get("final_chain") or ())
    if not final_chain:
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

    candidates = {
        "sensevoice": dedupe_repeated_transcript(sensevoice_candidate),
        "vosk_lgraph": dedupe_repeated_transcript(lgraph_candidate),
        "vosk_small": dedupe_repeated_transcript(vosk_candidate),
    }
    available = {
        "sensevoice": (
            backend_status.get("sensevoice_available", True)
            and not backend_status.get("sensevoice_error")
        ),
        "vosk_lgraph": (
            backend_status.get("vosk_lgraph_available", True)
            and not backend_status.get("vosk_lgraph_error")
        ),
        "vosk_small": True,
    }
    for source in final_chain:
        text = candidates.get(source, "")
        if available.get(source, False) and _is_usable(text):
            backend_status["selected_source"] = source
            return text
    backend_status["selected_source"] = "none"
    return ""
