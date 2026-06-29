"""Deterministic STT domain-term repair for CASE transcripts."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def normalize_for_glossary(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9 ]+", " ", str(text).lower())
    return " ".join(normalized.split())


@dataclass(frozen=True)
class DomainGlossaryPattern:
    canonical: str
    pattern: str
    negative_context: tuple[str, ...]

    @property
    def normalized_pattern(self) -> str:
        return normalize_for_glossary(self.pattern)


@dataclass(frozen=True)
class DomainGlossaryMatch:
    canonical: str
    pattern: str
    original: str


class DomainGlossary:
    def __init__(self, patterns: Iterable[DomainGlossaryPattern]):
        self.patterns = sorted(
            patterns,
            key=lambda item: (
                len(item.normalized_pattern.split()),
                len(item.normalized_pattern),
            ),
            reverse=True,
        )

    @property
    def size(self) -> int:
        return len(self.patterns)

    @classmethod
    def from_file(cls, path: str | Path) -> "DomainGlossary":
        source = Path(path)
        with source.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        patterns: list[DomainGlossaryPattern] = []
        for entry in data.get("terms", []):
            canonical = str(entry.get("canonical", "")).strip()
            if not canonical:
                continue
            negatives = tuple(
                normalize_for_glossary(item)
                for item in entry.get("negative_context", [])
                if str(item).strip()
            )
            for pattern in entry.get("patterns", []):
                pattern_text = str(pattern).strip()
                if pattern_text:
                    patterns.append(
                        DomainGlossaryPattern(
                            canonical=canonical,
                            pattern=pattern_text,
                            negative_context=negatives,
                        )
                    )
        return cls(patterns)

    def repair(self, text: str) -> tuple[str, DomainGlossaryMatch | None]:
        original_text = str(text).strip()
        normalized_text = normalize_for_glossary(original_text)
        if not original_text or not normalized_text:
            return original_text, None

        for item in self.patterns:
            if any(context and context in normalized_text for context in item.negative_context):
                continue
            pattern = item.normalized_pattern
            if not pattern:
                continue
            expression = re.compile(
                r"(?<!\w)" + r"\s+".join(re.escape(part) for part in pattern.split()) + r"(?!\w)",
                re.IGNORECASE,
            )
            match = expression.search(original_text)
            if not match:
                # Normalized matching catches punctuation variants, but only when
                # the original spacing also makes a safe replacement possible.
                normalized_expression = re.compile(
                    r"(?<!\w)"
                    + r"\s+".join(re.escape(part) for part in pattern.split())
                    + r"(?!\w)"
                )
                if not normalized_expression.search(normalized_text):
                    continue
                continue
            repaired = (
                original_text[: match.start()]
                + item.canonical
                + original_text[match.end() :]
            ).strip()
            return repaired, DomainGlossaryMatch(
                canonical=item.canonical,
                pattern=item.pattern,
                original=match.group(0),
            )
        return original_text, None


def load_domain_glossary(path: str | Path) -> DomainGlossary | None:
    source = Path(path)
    try:
        glossary = DomainGlossary.from_file(source)
    except FileNotFoundError:
        logging.warning("STT_GLOSSARY_REPAIR: missing path=%s", source)
        return None
    except Exception as exc:
        logging.warning("STT_GLOSSARY_REPAIR: failed path=%s error=%s", source, exc)
        return None
    logging.info("STT_GLOSSARY_REPAIR: loaded path=%s patterns=%d", source, glossary.size)
    return glossary
