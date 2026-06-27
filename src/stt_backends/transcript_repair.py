"""Small conservative transcript repairs for common CASE command phrases."""

from __future__ import annotations

import re


COMMON_TRANSCRIPT_REPAIRS = {
    "k roasted me": "Can you roast me?",
    "k roast me": "Can you roast me?",
    "can roasted me": "Can you roast me?",
    "can roast me": "Can you roast me?",
    "can you roasts me": "Can you roast me?",
    "movinging me something funny": "Tell me something funny.",
    "boring to me something funny": "Tell me something funny.",
    "a you doing": "What are you doing?",
    "the are you doing": "What are you doing?",
    "are you doing": "What are you doing?",
}

EMBEDDED_FOLLOWUP_COMMAND_REPAIRS = {
    "can you tell me something funny": "can you tell me something funny",
    "tell me something funny": "tell me something funny",
    "tell me another joke": "tell me another joke",
    "tell me a joke": "tell me a joke",
    "one more": "tell me another one",
    "again": "tell me another one",
    "do another one": "do another one",
    "make it funnier": "make it funnier",
    "tell me something longer": "tell me something longer",
    "continue": "continue",
    "go on": "go on",
    "can you roast me": "can you roast me",
    "roast me": "roast me",
}

FOLLOWUP_PHRASE_REPAIRS = {
    "yeah can you tell me long": "can you tell me something longer",
    "can you tell me long": "can you tell me something longer",
    "make it longer": "tell me a longer version",
    "make it shorter": "tell me a shorter version",
    "funnier": "tell me something funnier",
    "more funny": "tell me something funnier",
}

PHONETIC_FOLLOWUP_REPAIRS = {
    "which tusk do require you": "which task do you require",
    "which task do require you": "which task do you require",
    "what task do require you": "what task do you require",
}

BANTER_PHONETIC_REPAIRS = {
    "the here you should move out": "yeah you should move out",
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
    repaired = FOLLOWUP_PHRASE_REPAIRS.get(cleaned)
    if repaired:
        return repaired, "followup_phrase"
    repaired = PHONETIC_FOLLOWUP_REPAIRS.get(cleaned)
    if repaired:
        return repaired, "phonetic_followup_repair"
    repaired = BANTER_PHONETIC_REPAIRS.get(cleaned)
    if repaired:
        return repaired, "banter_phonetic_repair"
    if "tusk" in cleaned or "do require you" in cleaned:
        repaired_text = re.sub(r"\btusk\b", "task", cleaned)
        repaired_text = re.sub(r"\bdo require you\b", "do you require", repaired_text)
        if repaired_text != cleaned and (
            repaired_text.startswith("which task ")
            or repaired_text.startswith("what task ")
        ):
            return repaired_text, "phonetic_followup_repair"
    for phrase, replacement in EMBEDDED_FOLLOWUP_COMMAND_REPAIRS.items():
        if cleaned == phrase:
            return replacement, "embedded_known_command"
        marker = f" {phrase}"
        if marker in cleaned:
            return replacement, "embedded_known_command"
    if cleaned.startswith("sorry i mean "):
        remainder = cleaned.removeprefix("sorry i mean ").strip()
        if remainder in EMBEDDED_FOLLOWUP_COMMAND_REPAIRS:
            return EMBEDDED_FOLLOWUP_COMMAND_REPAIRS[remainder], "embedded_known_command"
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
    if joke_context and cleaned == "tell me up":
        return "Tell me a joke.", "context_joke_phrase"
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
