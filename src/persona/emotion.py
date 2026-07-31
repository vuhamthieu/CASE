"""Hybrid emotion routing for CASE voice/personality."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Callable


logger = logging.getLogger(__name__)

VALID_EMOTIONS = {
    "neutral",
    "deadpan",
    "amused",
    "sarcastic",
    "annoyed",
    "angry",
    "sad",
    "excited",
}

VALID_REASONS = {
    "default_personality",
    "user_rejection",
    "requested_emotion_style",
    "user_praise",
    "user_sadness",
    "humor_request",
    "emotion_meta_question",
    "emotion_deescalation",
    "sarcastic_followup",
    "ambiguous",
}

MEMORY_EMOTIONS = {
    "angry",
    "annoyed",
    "sad",
    "amused",
    "sarcastic",
    "excited",
}


@dataclass(frozen=True)
class EmotionState:
    emotion: str = "deadpan"
    intensity: float = 0.35
    reason: str = "default_personality"
    confidence: float = 1.0
    source: str = "rules"
    match: str = ""

    def __post_init__(self) -> None:
        emotion = self.emotion if self.emotion in VALID_EMOTIONS else "deadpan"
        reason = self.reason if self.reason in VALID_REASONS else "ambiguous"
        object.__setattr__(self, "emotion", emotion)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "intensity", clamp_unit(self.intensity, default=0.35))
        object.__setattr__(self, "confidence", clamp_unit(self.confidence, default=0.0))


@dataclass(frozen=True)
class TtsEmotionProfile:
    length_scale: float = 1.0
    gain_db: float = 0.0


@dataclass(frozen=True)
class UtteranceSignals:
    original_text: str
    normalized: str
    tokens: list[str]
    targets_case: bool
    targets_user_self: bool
    targets_other_object: bool
    is_question: bool
    is_command: bool
    is_deescalation: bool
    is_meta_emotion_question: bool
    requested_emotion: str | None
    negative_terms: list[str]
    positive_terms: list[str]
    rejection_terms: list[str]
    praise_terms: list[str]
    target_negative_score: float
    target_positive_score: float
    generic_negative_score: float


@dataclass
class EmotionMemory:
    last_emotion: str | None = None
    last_intensity: float = 0.0
    last_reason: str = ""
    last_confidence: float = 0.0
    last_source: str = ""
    updated_turn_id: int = 0
    updated_at_monotonic: float = 0.0

    def clear(self, *, reason: str) -> None:
        self.last_emotion = None
        self.last_intensity = 0.0
        self.last_reason = ""
        self.last_confidence = 0.0
        self.last_source = ""
        self.updated_turn_id = 0
        self.updated_at_monotonic = 0.0
        logger.info("EMOTION_MEMORY_CLEAR: reason=%s", reason)

    def update_from_state(
        self,
        state: EmotionState,
        *,
        turn_id: int,
        now: float | None = None,
        min_confidence: float = 0.75,
    ) -> None:
        if state.reason == "emotion_deescalation":
            self.clear(reason="deescalation")
            return
        if state.emotion not in MEMORY_EMOTIONS:
            logger.info("EMOTION_MEMORY_SKIP: reason=default_emotion")
            return
        if state.confidence < min_confidence:
            logger.info("EMOTION_MEMORY_SKIP: reason=low_confidence")
            return

        self.last_emotion = state.emotion
        self.last_intensity = state.intensity
        self.last_reason = state.reason
        self.last_confidence = state.confidence
        self.last_source = state.source
        self.updated_turn_id = int(turn_id)
        self.updated_at_monotonic = time.monotonic() if now is None else float(now)
        logger.info(
            "EMOTION_MEMORY_UPDATE: emotion=%s intensity=%.2f reason=%s "
            "confidence=%.2f source=%s turn=%s",
            self.last_emotion,
            self.last_intensity,
            self.last_reason,
            self.last_confidence,
            self.last_source,
            self.updated_turn_id,
        )

    def is_valid(
        self,
        *,
        turn_id: int,
        now: float | None = None,
        ttl_turns: int = 2,
        ttl_sec: float = 45.0,
    ) -> bool:
        if self.last_emotion not in MEMORY_EMOTIONS:
            return False
        current = time.monotonic() if now is None else float(now)
        if int(turn_id) - self.updated_turn_id > int(ttl_turns):
            logger.info("EMOTION_MEMORY_EXPIRE: reason=ttl_turns")
            self.clear(reason="ttl_turns")
            return False
        if current - self.updated_at_monotonic > float(ttl_sec):
            logger.info("EMOTION_MEMORY_EXPIRE: reason=ttl_seconds")
            self.clear(reason="ttl_seconds")
            return False
        return True


TTS_EMOTION_PROFILES = {
    "neutral": TtsEmotionProfile(length_scale=1.00, gain_db=0.0),
    "deadpan": TtsEmotionProfile(length_scale=1.05, gain_db=-0.5),
    "amused": TtsEmotionProfile(length_scale=0.98, gain_db=1.0),
    "sarcastic": TtsEmotionProfile(length_scale=1.03, gain_db=0.5),
    "annoyed": TtsEmotionProfile(length_scale=0.92, gain_db=2.0),
    "angry": TtsEmotionProfile(length_scale=0.85, gain_db=4.0),
    "sad": TtsEmotionProfile(length_scale=1.15, gain_db=-1.5),
    "excited": TtsEmotionProfile(length_scale=0.88, gain_db=3.0),
}


def clamp_unit(value: float, *, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def clamp_intensity(value: float) -> float:
    return clamp_unit(value, default=0.35)


NEGATIVE_TERMS = {
    "boring",
    "bored",
    "useless",
    "stupid",
    "dumb",
    "annoying",
    "bad",
    "terrible",
    "awful",
    "lame",
    "cringe",
    "slow",
    "broken",
    "trash",
    "garbage",
    "mid",
    "pathetic",
    "chán",
    "nhạt",
    "vô dụng",
    "ngu",
    "dở",
    "tệ",
    "phiền",
    "rác",
    "xàm",
    "phế",
}
POSITIVE_TERMS = {
    "good",
    "great",
    "nice",
    "amazing",
    "smart",
    "useful",
    "helpful",
    "impressive",
    "well",
    "proud",
    "funny",
    "tốt",
    "giỏi",
    "hay",
    "đỉnh",
    "xịn",
    "hữu ích",
    "tự hào",
}
REJECTION_TERMS = {
    "hate",
    "dislike",
    "do not like",
    "bored of",
    "tired of",
    "sick of",
    "done with",
    "annoyed by",
    "shut up",
    "stop talking",
    "nobody asked",
    "im đi",
    "câm đi",
    "ghét",
    "không thích",
    "chán",
    "mệt với",
}
PRAISE_TERMS = {
    "good job",
    "nice work",
    "well done",
    "did well",
    "did good",
    "proud of",
    "làm tốt",
    "tự hào",
}
CASE_TARGET_TERMS = {"you", "your", "case", "robot", "bot", "mày", "bạn", "cậu", "m"}
SELF_TARGET_TERMS = {"i", "me", "my", "mình", "tao", "tớ", "tui"}
OBJECT_TERMS = {"movie", "joke", "task", "weather", "code", "board"}
OBJECT_DETERMINERS = {"this", "that", "the"}
EMOTION_REQUEST_TERMS = {
    "angry": "angry",
    "mad": "angry",
    "giận": "angry",
    "gắt": "angry",
    "tức giận": "angry",
    "sarcastic": "sarcastic",
    "sad": "sad",
    "buồn": "sad",
    "happy": "excited",
    "excited": "excited",
    "annoyed": "annoyed",
    "angrily": "angry",
    "louder": "excited",
    "to hơn": "excited",
}
HUMOR_TERMS = {"joke", "funny", "roast", "laugh", "khịa", "chuyện cười"}
COMMAND_TERMS = {"speak", "act", "sound", "talk", "say", "nói", "be", "get"}
SELF_SADNESS_TERMS = {"sad", "tired", "stressed", "buồn", "mệt"}
SHORT_PRAISE_UTTERANCES = {"nice", "good job", "well done", "nice work", "hay đấy", "tốt đấy"}
DISMISSIVE_REJECTION_UTTERANCES = {"shut up", "stop talking", "nobody asked you", "im đi", "i am đi", "câm đi"}

_CONTRACTIONS = (
    (r"\bi'm\b", "i am"),
    (r"\bim\b", "i am"),
    (r"\bi've\b", "i have"),
    (r"\bive\b", "i have"),
    (r"\byou're\b", "you are"),
    (r"\byoure\b", "you are"),
    (r"\bdon't\b", "do not"),
    (r"\bdont\b", "do not"),
    (r"\bcan't\b", "cannot"),
    (r"\bcant\b", "cannot"),
)
_START_FILLERS = (
    ("sorry", "i", "mean"),
    ("i", "mean"),
    ("ý", "là"),
    ("actually",),
    ("like",),
    ("yea",),
    ("yeah",),
    ("uh",),
    ("um",),
    ("so",),
    ("à",),
)


def normalize_emotion_text(text: str, *, strip_start_fillers: bool = True) -> str:
    raw = str(text or "")
    normalized = raw.lower()
    for pattern, replacement in _CONTRACTIONS:
        normalized = re.sub(pattern, replacement, normalized)
    normalized = re.sub(r"[^\w\s]+", " ", normalized, flags=re.UNICODE)
    words = normalized.split()
    if strip_start_fillers:
        changed = True
        while changed and words:
            changed = False
            for filler in _START_FILLERS:
                if tuple(words[: len(filler)]) == filler:
                    del words[: len(filler)]
                    changed = True
                    break
    normalized = " ".join(words)
    logger.debug("EMOTION_NORMALIZED: raw=%r normalized=%r", raw, normalized)
    return normalized


def _contains_phrase(normalized: str, phrase: str) -> bool:
    return bool(re.search(rf"\b{re.escape(phrase)}\b", normalized))


def _find_terms(normalized: str, tokens: list[str], terms: set[str]) -> list[str]:
    found: list[str] = []
    token_set = set(tokens)
    for term in sorted(terms):
        if " " in term:
            if _contains_phrase(normalized, term):
                found.append(term)
        elif term in token_set:
            found.append(term)
    return found


def _has_any_phrase(normalized: str, phrases: tuple[str, ...]) -> bool:
    return any(_contains_phrase(normalized, phrase) for phrase in phrases)


def _term_indexes(tokens: list[str], terms: set[str]) -> list[int]:
    return [index for index, token in enumerate(tokens) if token in terms]


def _has_proximity(tokens: list[str], left_terms: set[str], right_terms: set[str], *, window: int = 4) -> bool:
    left = _term_indexes(tokens, left_terms)
    right = _term_indexes(tokens, right_terms)
    return any(abs(a - b) <= window for a in left for b in right)


def _has_direct_case_target(tokens: list[str]) -> bool:
    return any(token in {"you", "your", "mày", "bạn", "cậu", "m"} for token in tokens)


def _case_named_before_sentiment(tokens: list[str], sentiment_terms: set[str]) -> bool:
    case_indexes = _term_indexes(tokens, {"case", "robot", "bot"})
    sentiment_indexes = _term_indexes(tokens, sentiment_terms)
    return any(case_index < sentiment_index for case_index in case_indexes for sentiment_index in sentiment_indexes)


def _targets_case(normalized: str, tokens: list[str]) -> bool:
    if _has_direct_case_target(tokens):
        return True
    if _contains_phrase(normalized, "hey case"):
        return True
    return any(token in {"case", "robot", "bot"} for token in tokens)


def _targets_other_object(tokens: list[str]) -> bool:
    if any(token in OBJECT_TERMS for token in tokens):
        return True
    return bool(tokens and tokens[0] in OBJECT_DETERMINERS)


def _has_case_rejection_relation(normalized: str, tokens: list[str]) -> bool:
    relation_phrases = (
        "bored of you",
        "tired of you",
        "sick of you",
        "done with you",
        "annoyed by you",
        "bored of case",
        "tired of case",
        "sick of case",
        "hate you",
        "dislike you",
        "do not like you",
        "hate case",
        "dislike case",
        "ghét mày",
        "ghét bạn",
        "không thích mày",
        "không thích bạn",
        "chán mày",
        "chán bạn",
        "mệt với mày",
        "mệt với bạn",
    )
    if _has_any_phrase(normalized, relation_phrases):
        return True
    if _has_proximity(tokens, {"ghét", "chán", "mệt"}, {"mày", "bạn", "cậu", "m"}, window=4):
        return True
    return False


def _is_dismissive_rejection(normalized: str) -> bool:
    return any(_contains_phrase(normalized, phrase) for phrase in DISMISSIVE_REJECTION_UTTERANCES)


def _is_self_sadness(signals: UtteranceSignals) -> bool:
    if signals.target_negative_score > 0.0:
        return False
    if "feel bad" in signals.normalized:
        return True
    return signals.targets_user_self and any(term in signals.tokens for term in SELF_SADNESS_TERMS)


def _has_case_negative_construction(normalized: str, tokens: list[str], negative_terms: list[str], rejection_terms: list[str]) -> bool:
    if rejection_terms and _has_case_rejection_relation(normalized, tokens):
        return True
    negative_set = set(negative_terms)
    if not negative_set:
        return False
    if _has_direct_case_target(tokens) and any(" " in term and _contains_phrase(normalized, term) for term in negative_terms):
        return True
    if _has_direct_case_target(tokens) and _has_proximity(tokens, CASE_TARGET_TERMS, negative_set, window=5):
        return True
    if _case_named_before_sentiment(tokens, negative_set):
        return True
    return False


def _has_case_positive_construction(normalized: str, tokens: list[str], positive_terms: list[str], praise_terms: list[str]) -> bool:
    positive_set = set(positive_terms) | set(praise_terms)
    if not positive_set:
        return False
    if _has_direct_case_target(tokens) and any(
        " " in term and _contains_phrase(normalized, term)
        for term in list(positive_terms) + list(praise_terms)
    ):
        return True
    if _has_direct_case_target(tokens) and _has_proximity(tokens, CASE_TARGET_TERMS, positive_set, window=5):
        return True
    if _case_named_before_sentiment(tokens, positive_set):
        return True
    if _has_any_phrase(normalized, ("good job case", "proud of you", "tự hào về mày", "tự hào về bạn")):
        return True
    if normalized in SHORT_PRAISE_UTTERANCES:
        return True
    return False


def _detect_requested_emotion(normalized: str, tokens: list[str]) -> str | None:
    if _detect_deescalation(normalized):
        return None
    requested = None
    for term, emotion in EMOTION_REQUEST_TERMS.items():
        if _contains_phrase(normalized, term):
            requested = emotion
            break
    if requested is None:
        return None
    if any(token in COMMAND_TERMS for token in tokens):
        return requested
    if _has_any_phrase(normalized, ("can you", "for a moment", "or moment", "like you are")):
        return requested
    return None


def _is_humor_request(normalized: str, tokens: list[str]) -> bool:
    if "roast" in tokens or "khịa" in tokens:
        return True
    if "joke" in tokens and any(token in {"tell", "make", "say"} for token in tokens):
        return True
    return _has_any_phrase(
        normalized,
        ("tell me a joke", "make me laugh", "say something funny", "make fun of me", "kể chuyện cười"),
    )


def analyze_utterance_signals(text: str) -> UtteranceSignals:
    normalized = normalize_emotion_text(text)
    tokens = normalized.split()
    negative_terms = _find_terms(normalized, tokens, NEGATIVE_TERMS)
    positive_terms = _find_terms(normalized, tokens, POSITIVE_TERMS)
    rejection_terms = _find_terms(normalized, tokens, REJECTION_TERMS)
    praise_terms = _find_terms(normalized, tokens, PRAISE_TERMS)
    targets_case = _targets_case(normalized, tokens)
    targets_user_self = any(token in SELF_TARGET_TERMS for token in tokens)
    targets_other_object = _targets_other_object(tokens) and not targets_case
    target_negative = 0.0
    if targets_case and _has_case_negative_construction(normalized, tokens, negative_terms, rejection_terms):
        target_negative = 0.90 if rejection_terms else 0.86
    target_positive = 0.0
    if (targets_case or normalized in SHORT_PRAISE_UTTERANCES) and _has_case_positive_construction(normalized, tokens, positive_terms, praise_terms):
        target_positive = 0.82 if praise_terms else 0.78
    generic_negative = 0.55 if negative_terms else 0.0
    signals = UtteranceSignals(
        original_text=str(text or ""),
        normalized=normalized,
        tokens=tokens,
        targets_case=targets_case,
        targets_user_self=targets_user_self,
        targets_other_object=targets_other_object,
        is_question="?" in str(text or "") or bool(tokens and tokens[0] in {"what", "why", "how", "are", "do", "did", "can", "could", "would", "is"}),
        is_command=any(token in COMMAND_TERMS for token in tokens),
        is_deescalation=_detect_deescalation(normalized) is not None,
        is_meta_emotion_question=_detect_emotion_meta_question(normalized) is not None,
        requested_emotion=_detect_requested_emotion(normalized, tokens),
        negative_terms=negative_terms,
        positive_terms=positive_terms,
        rejection_terms=rejection_terms,
        praise_terms=praise_terms,
        target_negative_score=target_negative,
        target_positive_score=target_positive,
        generic_negative_score=generic_negative,
    )
    logger.debug(
        "EMOTION_SIGNALS: targets_case=%s self_target=%s negative=%s positive=%s "
        "rejection=%s requested=%s meta=%s deescalation=%s",
        signals.targets_case,
        signals.targets_user_self,
        signals.negative_terms,
        signals.positive_terms,
        signals.rejection_terms,
        signals.requested_emotion,
        signals.is_meta_emotion_question,
        signals.is_deescalation,
    )
    return signals


def _state(
    emotion: str,
    intensity: float,
    reason: str,
    *,
    confidence: float,
    match: str,
    source: str = "rules",
) -> EmotionState:
    return EmotionState(
        emotion=emotion,
        intensity=intensity,
        reason=reason,
        confidence=confidence,
        source=source,
        match=match,
    )


def default_emotion_state(
    *,
    emotion: str = "deadpan",
    intensity: float = 0.35,
    reason: str = "default_personality",
    confidence: float = 0.0,
    source: str = "rules",
    match: str = "no_rule_match",
) -> EmotionState:
    return EmotionState(
        emotion=emotion,
        intensity=intensity,
        reason=reason,
        confidence=confidence,
        source=source,
        match=match,
    )


def _route_from_signals(signals: UtteranceSignals) -> EmotionState:
    if signals.requested_emotion:
        emotion = signals.requested_emotion
        intensity = 0.70 if emotion == "angry" else 0.65
        logger.debug(
            "EMOTION_SCORE: candidate=%s reason=requested_emotion_style score=0.80 match=requested_emotion_style",
            emotion,
        )
        return _state(
            emotion,
            intensity,
            "requested_emotion_style",
            confidence=0.80,
            match="requested_emotion_style",
        )

    if _is_dismissive_rejection(signals.normalized):
        logger.debug(
            "EMOTION_SCORE: candidate=angry reason=user_rejection score=0.88 match=dismissive_command"
        )
        return _state(
            "angry",
            0.82,
            "user_rejection",
            confidence=0.88,
            match="dismissive_command",
        )

    if signals.target_negative_score >= 0.80:
        logger.debug(
            "EMOTION_SCORE: candidate=angry reason=user_rejection score=%.2f match=targeted_negative_sentiment",
            signals.target_negative_score,
        )
        return _state(
            "angry",
            0.85,
            "user_rejection",
            confidence=signals.target_negative_score,
            match="targeted_negative_sentiment",
        )

    if _is_self_sadness(signals):
        logger.debug(
            "EMOTION_SCORE: candidate=sad reason=user_sadness score=0.82 match=self_sadness"
        )
        return _state(
            "sad",
            0.65,
            "user_sadness",
            confidence=0.82,
            match="self_sadness",
        )

    if signals.target_positive_score >= 0.70:
        logger.debug(
            "EMOTION_SCORE: candidate=amused reason=user_praise score=%.2f match=targeted_positive_sentiment",
            signals.target_positive_score,
        )
        return _state(
            "amused",
            0.65,
            "user_praise",
            confidence=signals.target_positive_score,
            match="targeted_positive_sentiment",
        )

    if _is_humor_request(signals.normalized, signals.tokens):
        logger.debug(
            "EMOTION_SCORE: candidate=sarcastic reason=humor_request score=0.82 match=humor_request"
        )
        return _state(
            "sarcastic",
            0.65,
            "humor_request",
            confidence=0.82,
            match="humor_request",
        )

    if signals.generic_negative_score > 0.0:
        logger.debug("EMOTION_SCORE_SKIP: reason=negative_not_targeted")
    return default_emotion_state()


def detect_emotion(text: str) -> EmotionState:
    signals = analyze_utterance_signals(text)
    if not signals.normalized:
        return default_emotion_state()
    return _route_from_signals(signals)


def _detect_deescalation(cleaned: str) -> str | None:
    patterns = (
        (r"\bcalm down\b", "calm_down"),
        (r"\bdo not be angry\b", "do_not_be_angry"),
        (r"\bdon t be angry\b", "dont_be_angry"),
        (r"\bdo not angry\b", "do_not_angry"),
        (r"\bdont angry\b", "dont_angry"),
        (r"\bdon t angry\b", "dont_angry"),
        (r"\bđừng giận\b", "vietnamese_dont_be_angry"),
        (r"\bbình tĩnh đi\b", "vietnamese_calm_down"),
    )
    for pattern, name in patterns:
        if re.search(pattern, cleaned):
            return name
    return None


def _detect_apology(cleaned: str) -> str | None:
    patterns = (
        (r"^sorry(?:\s+case)?$", "apology"),
        (r"\bmy bad\b", "apology"),
        (r"\bmy fault\b", "apology"),
        (r"\bi apologize\b", "apology"),
        (r"^xin lỗi(?:\s+case)?$", "apology"),
        (r"\blỗi của tao\b", "apology"),
        (r"\blỗi của mình\b", "apology"),
    )
    for pattern, name in patterns:
        if re.search(pattern, cleaned):
            return name
    return None


def _detect_emotion_meta_question(cleaned: str) -> str | None:
    patterns = (
        (r"\bare you (?:not )?(?:mad|angry) at me\b", "soft_mad_at_me"),
        (r"\bare you upset with me\b", "soft_upset_with_me"),
        (r"\bdid i make you (?:angry|mad)\b", "soft_user_caused_emotion"),
        (r"\bare you (?:mad|angry) because of me\b", "soft_user_caused_emotion"),
        (r"\bwhy are you (?:angry|mad|upset)\b", "why_are_you_angry"),
        (r"\bare you still (?:angry|mad|upset)\b", "are_you_still_angry"),
        (r"\bare you not (?:angry|mad|upset)\b", "are_you_not_angry"),
        (r"\byou are not (?:angry|mad|upset)\b", "you_are_not_angry"),
        (r"\byou not (?:angry|mad|upset)\b", "you_not_angry"),
        (r"^not (?:angry|mad|upset)$", "not_angry"),
        (r"\bare you (?:angry|mad|upset)\b", "are_you_angry"),
        (r"\byou (?:angry|mad|upset)\b", "you_angry"),
        (r"\bmày không giận à\b", "vietnamese_you_not_angry"),
        (r"\bbạn không giận à\b", "vietnamese_you_not_angry"),
        (r"\bvẫn giận à\b", "vietnamese_still_angry"),
        (r"\bsao mày giận\b", "vietnamese_why_angry"),
        (r"\bmày giận à\b", "vietnamese_you_angry"),
        (r"\bbạn giận à\b", "vietnamese_you_angry"),
        (r"\bcase giận à\b", "vietnamese_case_angry"),
    )
    for pattern, name in patterns:
        if re.search(pattern, cleaned):
            return name
    return None


def _detect_sarcastic_followup(cleaned: str) -> str | None:
    patterns = (
        (r"\bha ha\s+very funny\b", "ha_ha_very_funny"),
        (r"\bhaha\s+very funny\b", "haha_very_funny"),
        (r"\bvery funny\b", "very_funny"),
        (r"^funny$", "funny"),
    )
    for pattern, name in patterns:
        if re.search(pattern, cleaned):
            return name
    return None


def _memory_carry_state(
    memory: EmotionMemory,
    *,
    match: str,
    decay: float,
) -> EmotionState:
    previous = memory.last_emotion or "deadpan"
    intensity = clamp_intensity(memory.last_intensity * decay)
    confidence = clamp_unit(memory.last_confidence * decay, default=0.75)
    if previous in {"angry", "annoyed"}:
        emotion = "annoyed"
        intensity = max(0.45, min(0.75, intensity))
    elif previous == "sad":
        emotion = "sad"
        intensity = max(0.40, min(0.70, intensity))
    elif previous in {"amused", "sarcastic"}:
        emotion = previous
        intensity = max(0.35, min(0.65, intensity))
    elif previous == "excited":
        emotion = "excited"
        intensity = max(0.35, min(0.65, intensity))
    else:
        emotion = "deadpan"
        intensity = 0.35
    logger.info(
        "EMOTION_MEMORY_CARRY: from=%s to=%s intensity=%.2f "
        "reason=emotion_meta_question source=memory",
        previous,
        emotion,
        intensity,
    )
    return _state(
        emotion,
        intensity,
        "emotion_meta_question",
        confidence=max(0.70, confidence),
        source="memory",
        match=match,
    )


def _sarcastic_followup_state(memory: EmotionMemory, *, match: str) -> EmotionState:
    previous = memory.last_emotion or "deadpan"
    intensity = 0.62 if previous in {"angry", "annoyed"} else 0.55
    logger.info(
        "EMOTION_MEMORY_CARRY: from=%s to=sarcastic intensity=%.2f "
        "reason=sarcastic_followup source=memory",
        previous,
        intensity,
    )
    return _state(
        "sarcastic",
        intensity,
        "sarcastic_followup",
        confidence=0.78,
        source="memory",
        match=match,
    )


def _memory_deescalation_state(memory: EmotionMemory, *, match: str) -> EmotionState:
    previous = memory.last_emotion or "deadpan"
    if match == "apology":
        emotion = "deadpan"
        intensity = 0.35
    else:
        emotion = "annoyed" if previous in {"angry", "annoyed"} else "deadpan"
        intensity = 0.45 if emotion == "annoyed" else 0.35
    logger.info(
        "EMOTION_DEESCALATE: reason=%s from=%s to=%s intensity=%.2f",
        match,
        previous,
        emotion,
        intensity,
    )
    return _state(
        emotion,
        intensity,
        "emotion_deescalation",
        confidence=0.85,
        source="memory",
        match=match,
    )


def select_emotion_with_memory(
    text: str,
    memory: EmotionMemory | None,
    *,
    turn_id: int,
    now: float | None = None,
    memory_enabled: bool = True,
    ttl_turns: int = 2,
    ttl_sec: float = 45.0,
    min_confidence: float = 0.75,
    decay: float = 0.70,
    meta_questions_enabled: bool = True,
) -> EmotionState:
    current = detect_emotion(text)
    if memory is None or not memory_enabled:
        return current

    if current.reason != "default_personality":
        memory.update_from_state(
            current,
            turn_id=turn_id,
            now=now,
            min_confidence=min_confidence,
        )
        return current

    cleaned = normalize_emotion_text(text)
    if meta_questions_enabled:
        apology_match = _detect_apology(cleaned)
        if apology_match:
            logger.info("EMOTION_DEESCALATE: detected=true match=%s", apology_match)
            if (
                memory.last_emotion in {"angry", "annoyed"}
                and memory.is_valid(
                    turn_id=turn_id,
                    now=now,
                    ttl_turns=ttl_turns,
                    ttl_sec=ttl_sec,
                )
            ):
                state = _memory_deescalation_state(memory, match=apology_match)
                memory.clear(reason="apology")
                return state

        deescalation_match = _detect_deescalation(cleaned)
        if deescalation_match:
            logger.info("EMOTION_META_QUESTION: detected=true match=%s", deescalation_match)
            if memory.is_valid(
                turn_id=turn_id,
                now=now,
                ttl_turns=ttl_turns,
                ttl_sec=ttl_sec,
            ):
                state = _memory_deescalation_state(memory, match=deescalation_match)
                memory.update_from_state(
                    state,
                    turn_id=turn_id,
                    now=now,
                    min_confidence=min_confidence,
                )
                return state

        meta_match = _detect_emotion_meta_question(cleaned)
        if meta_match:
            logger.info("EMOTION_META_QUESTION: detected=true match=%s", meta_match)
            if memory.is_valid(
                turn_id=turn_id,
                now=now,
                ttl_turns=ttl_turns,
                ttl_sec=ttl_sec,
            ):
                return _memory_carry_state(memory, match=meta_match, decay=decay)

    sarcastic_match = _detect_sarcastic_followup(cleaned)
    if sarcastic_match and memory.is_valid(
        turn_id=turn_id,
        now=now,
        ttl_turns=ttl_turns,
        ttl_sec=ttl_sec,
    ) and memory.last_emotion in {"angry", "annoyed", "sarcastic"}:
        state = _sarcastic_followup_state(memory, match=sarcastic_match)
        memory.update_from_state(
            state,
            turn_id=turn_id,
            now=now,
            min_confidence=min_confidence,
        )
        return state

    memory.update_from_state(
        current,
        turn_id=turn_id,
        now=now,
        min_confidence=min_confidence,
    )
    return current


def emotion_prompt_instruction(state: EmotionState) -> str:
    if state.reason == "emotion_deescalation":
        return (
            "The user is telling CASE to calm down. Respond with dry restraint "
            "and then reduce emotional intensity. Keep it short."
        )
    if state.reason == "emotion_meta_question" and state.emotion in {"angry", "annoyed"}:
        return (
            "The user is asking whether CASE is angry after a previous rejection "
            "or insult. Respond as CASE with restrained offended sarcasm. Keep it "
            "short. Do not be threatening or hateful."
        )
    if state.emotion in {"angry", "annoyed"}:
        return (
            "The user insulted or rejected CASE. Respond as CASE with offended, "
            "sharp, sarcastic energy. Keep it short. Do not be genuinely harmful "
            "or threatening."
        )
    if state.emotion == "amused":
        return "Respond with amused confidence. Short, dry, slightly smug."
    if state.emotion == "sad":
        return "Respond more softly and supportively. Still sound like CASE, but less harsh."
    if state.emotion == "excited":
        return "Respond with brief energized confidence. Stay concise and useful."
    if state.emotion == "sarcastic":
        return "Respond with dry sarcastic timing. Keep it short and harmless."
    return "Respond in CASE's usual dry, concise, deadpan style."


def build_emotion_user_message(user_text: str, state: EmotionState) -> str:
    instruction = emotion_prompt_instruction(state)
    return (
        "Internal response style note for CASE, not to be spoken verbatim: "
        f"{instruction}\n\nUser said: {user_text}"
    )


def blend_tts_emotion_profile(
    state: EmotionState,
    *,
    max_gain_db: float = 5.0,
) -> TtsEmotionProfile:
    neutral = TTS_EMOTION_PROFILES["neutral"]
    target = TTS_EMOTION_PROFILES.get(state.emotion, TTS_EMOTION_PROFILES["deadpan"])
    intensity = clamp_intensity(state.intensity)
    length_scale = neutral.length_scale + (target.length_scale - neutral.length_scale) * intensity
    gain_db = neutral.gain_db + (target.gain_db - neutral.gain_db) * intensity
    gain_db = max(-abs(max_gain_db), min(abs(max_gain_db), gain_db))
    return TtsEmotionProfile(length_scale=length_scale, gain_db=gain_db)


_EMOTION_TAG_RE = re.compile(
    r"^\s*\[emotion=(?P<emotion>[a-zA-Z_]+)(?:\s+intensity=(?P<intensity>[0-9.]+))?\]\s*(?P<text>.*)$",
    re.DOTALL,
)
_MALFORMED_EMOTION_TAG_RE = re.compile(r"^\s*\[emotion=[^\]]*\]\s*(?P<text>.*)$", re.DOTALL)


def parse_leading_emotion_tag(text: str) -> tuple[EmotionState | None, str]:
    source = str(text)
    match = _EMOTION_TAG_RE.match(source)
    if match:
        emotion = match.group("emotion").lower()
        if emotion in VALID_EMOTIONS:
            intensity = match.group("intensity")
            return (
                EmotionState(
                    emotion=emotion,
                    intensity=0.35 if intensity is None else float(intensity),
                    reason="ambiguous",
                    confidence=0.80,
                    source="model_tag",
                    match="leading_emotion_tag",
                ),
                match.group("text").strip(),
            )
        return None, match.group("text").strip()
    malformed = _MALFORMED_EMOTION_TAG_RE.match(source)
    if malformed:
        return None, malformed.group("text").strip()
    return None, source.strip()


def parse_llm_emotion_json(
    payload: str,
    *,
    min_confidence: float = 0.70,
) -> EmotionState | None:
    try:
        data = json.loads(str(payload))
    except json.JSONDecodeError:
        logger.info("EMOTION_LLM_CLASSIFY_FALLBACK: reason=parse_error")
        return None
    emotion = str(data.get("emotion", "")).strip().lower()
    reason = str(data.get("reason", "")).strip().lower()
    confidence = clamp_unit(data.get("confidence", 0.0), default=0.0)
    if emotion not in VALID_EMOTIONS or reason not in VALID_REASONS:
        logger.info("EMOTION_LLM_CLASSIFY_FALLBACK: reason=invalid_label")
        return None
    if confidence < min_confidence:
        logger.info("EMOTION_LLM_CLASSIFY_FALLBACK: reason=default_low_confidence")
        return None
    return EmotionState(
        emotion=emotion,
        intensity=clamp_intensity(data.get("intensity", 0.35)),
        reason=reason,
        confidence=confidence,
        source="llm",
        match="llm_classifier",
    )


def classify_emotion_with_llm(
    text: str,
    classifier: Callable[[str], str],
    *,
    min_confidence: float = 0.70,
) -> EmotionState | None:
    logger.info("EMOTION_LLM_CLASSIFY_START: text=%r", text)
    try:
        payload = classifier(text)
    except Exception as exc:
        logger.info("EMOTION_LLM_CLASSIFY_FALLBACK: reason=error error=%s", exc)
        return None
    state = parse_llm_emotion_json(payload, min_confidence=min_confidence)
    if state is not None:
        logger.info(
            "EMOTION_LLM_CLASSIFY_RESULT: emotion=%s intensity=%.2f "
            "confidence=%.2f reason=%s",
            state.emotion,
            state.intensity,
            state.confidence,
            state.reason,
        )
    return state
