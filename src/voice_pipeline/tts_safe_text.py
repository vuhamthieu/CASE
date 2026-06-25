"""Safe text boundaries for streamed Piper speech."""

from __future__ import annotations

import re


WEAK_TRAILING_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "to",
        "of",
        "for",
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
)


def clean_tts_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def trailing_word(text: str) -> str:
    words = re.findall(r"[A-Za-z']+", clean_tts_text(text).lower())
    return words[-1] if words else ""


def has_safe_sentence_boundary(text: str) -> bool:
    cleaned = clean_tts_text(text)
    return bool(cleaned and re.search(r"[.!?][\"']?$", cleaned)) and (
        trailing_word(cleaned) not in WEAK_TRAILING_WORDS
    )


def first_complete_sentence(text: str, max_chars: int | None = None) -> str:
    cleaned = clean_tts_text(text)
    for match in re.finditer(r"[.!?](?=[\"']?(?:\s|$))", cleaned):
        candidate = cleaned[: match.end()].strip()
        if max_chars is not None and len(candidate) > max_chars:
            break
        if has_safe_sentence_boundary(candidate):
            return candidate
    return ""


def trim_to_safe_boundary(
    text: str,
    *,
    max_chars: int,
    min_clause_chars: int,
) -> str:
    """Return a complete sentence or a comma-delimited clause ending in a period."""
    cleaned = clean_tts_text(text)
    sentence = first_complete_sentence(cleaned, max_chars=max_chars)
    if sentence:
        return sentence

    prefix = cleaned[:max_chars].rstrip()
    boundaries = [match.start() for match in re.finditer(r"[,;:]", prefix)]
    for boundary in reversed(boundaries):
        candidate = prefix[:boundary].strip()
        if (
            len(candidate) >= min_clause_chars
            and trailing_word(candidate) not in WEAK_TRAILING_WORDS
        ):
            return candidate.rstrip(" ,;:") + "."
    return ""


def safe_tts_text(
    text: str,
    *,
    max_chars: int,
    min_clause_chars: int,
    fallback: str,
) -> str:
    cleaned = clean_tts_text(text)
    cleaned_fallback = clean_tts_text(fallback)
    if (
        cleaned == cleaned_fallback
        and len(cleaned) <= max_chars
        and has_safe_sentence_boundary(cleaned)
    ):
        return cleaned
    return trim_to_safe_boundary(
        text,
        max_chars=max_chars,
        min_clause_chars=min_clause_chars,
    ) or cleaned_fallback
