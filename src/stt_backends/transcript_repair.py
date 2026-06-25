"""Small conservative transcript repairs for common CASE command phrases."""

from __future__ import annotations

import re


COMMON_TRANSCRIPT_REPAIRS = {
    "k roasted me": "Can you roast me?",
    "k roast me": "Can you roast me?",
    "can roasted me": "Can you roast me?",
    "can roast me": "Can you roast me?",
    "a you doing": "What are you doing?",
    "the are you doing": "What are you doing?",
    "are you doing": "What are you doing?",
}

MALFORMED_UNREPAIRED_PREFIXES = (
    "k ",
)


def normalize_for_repair(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9 ]+", " ", str(text).lower())
    return " ".join(normalized.split())


def repair_common_transcript(
    text: str,
    *,
    recent_context: str = "",
) -> tuple[str, str | None]:
    cleaned = normalize_for_repair(text)
    repaired = COMMON_TRANSCRIPT_REPAIRS.get(cleaned)
    if repaired:
        return repaired, "common_phrase"
    if "real a longer joke" in cleaned:
        repaired_text = re.sub(
            r"\breal\s+a\s+longer\s+joke\b",
            "real longer joke",
            str(text).strip(),
            flags=re.IGNORECASE,
        )
        return repaired_text, "common_phrase"
    context = normalize_for_repair(recent_context)
    joke_context = any(
        word in context for word in {"joke", "funny", "roast", "laugh", "punchline"}
    )
    if joke_context and (
        "tell me a longer job" in cleaned or "tell me your longer job" in cleaned
    ):
        if "too short" in cleaned:
            return (
                "That's too short. Tell me a longer joke.",
                "context_joke_job_to_joke",
            )
        return "Tell me a longer joke.", "context_joke_job_to_joke"
    if joke_context and "longer job" in cleaned:
        repaired_text = re.sub(
            r"\blonger job\b",
            "longer joke",
            str(text).strip(),
            flags=re.IGNORECASE,
        )
        return repaired_text, "context_joke_job_to_joke"
    return str(text).strip(), None


def malformed_transcript_reason(text: str) -> str | None:
    cleaned = normalize_for_repair(text)
    if any(cleaned.startswith(prefix) for prefix in MALFORMED_UNREPAIRED_PREFIXES):
        return "malformed_unrepaired"
    return None
