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


_CONTRACTIONS = (
    (r"\bi'm\b", "i am"),
    (r"\bim\b", "i am"),
    (r"\byou're\b", "you are"),
    (r"\byoure\b", "you are"),
    (r"\bdon't\b", "do not"),
    (r"\bcant\b", "can not"),
    (r"\bcan't\b", "can not"),
)
_START_FILLERS = {"yeah", "uh", "um", "like", "so"}


def normalize_emotion_text(text: str, *, strip_start_fillers: bool = True) -> str:
    raw = str(text or "")
    normalized = raw.lower()
    for pattern, replacement in _CONTRACTIONS:
        normalized = re.sub(pattern, replacement, normalized)
    normalized = re.sub(r"[^\w\s]+", " ", normalized, flags=re.UNICODE)
    words = normalized.split()
    if strip_start_fillers:
        while words and words[0] in _START_FILLERS:
            words.pop(0)
    normalized = " ".join(words)
    logger.debug("EMOTION_NORMALIZED: raw=%r normalized=%r", raw, normalized)
    return normalized


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


def _requested_style(cleaned: str) -> EmotionState | None:
    if re.search(r"\b(?:be|act|sound)\s+(?:so\s+|really\s+)?angry\b", cleaned):
        return _state("angry", 0.70, "requested_emotion_style", confidence=0.82, match="english_angry_style")
    if re.search(r"\bspeak\s+(?:angrily|angry|louder)\b", cleaned):
        return _state("angry", 0.70, "requested_emotion_style", confidence=0.80, match="english_angry_speak")
    if re.search(r"\bsay it\s+angry\b", cleaned):
        return _state("angry", 0.70, "requested_emotion_style", confidence=0.80, match="english_angry_say_it")
    if re.search(r"\bsound\s+excited\b", cleaned):
        return _state("excited", 0.68, "requested_emotion_style", confidence=0.78, match="english_excited_style")
    if re.search(r"\bsound\s+sad\b", cleaned):
        return _state("sad", 0.65, "requested_emotion_style", confidence=0.78, match="english_sad_style")
    if re.search(r"\bnói\s+(?:giận|kiểu tức giận|gắt)(?:\s+lên)?\b", cleaned):
        return _state("angry", 0.70, "requested_emotion_style", confidence=0.82, match="vietnamese_angry_style")
    if re.search(r"\bnói\s+buồn\s+hơn\b", cleaned):
        return _state("sad", 0.65, "requested_emotion_style", confidence=0.80, match="vietnamese_sad_style")
    if re.search(r"\bnói\s+vui\s+hơn\b", cleaned):
        return _state("excited", 0.65, "requested_emotion_style", confidence=0.76, match="vietnamese_excited_style")
    if re.search(r"\bnói\s+to\s+hơn\b", cleaned):
        return _state("excited", 0.62, "requested_emotion_style", confidence=0.75, match="vietnamese_louder_style")
    return None


def _user_rejection(cleaned: str) -> EmotionState | None:
    if re.search(r"\b(?:bored|tired|sick)\s+(?:of|with)\s+(?:you|case)\b", cleaned):
        return _state("angry", 0.85, "user_rejection", confidence=0.90, match="bored_of_you")
    if re.search(r"\b(?:i am\s+)?(?:really\s+|so\s+)?(?:bored|tired|sick)\s+(?:of|with)\s+(?:you|case)\b", cleaned):
        return _state("angry", 0.85, "user_rejection", confidence=0.90, match="bored_of_you")
    if re.search(r"\b(?:hate|dislike)\s+(?:you|case)\b", cleaned):
        return _state("angry", 0.86, "user_rejection", confidence=0.90, match="hate_you")
    if re.search(r"\b(?:you|case)\s+(?:are\s+)?(?:boring|useless|stupid|dumb|annoying)\b", cleaned):
        return _state("angry", 0.85, "user_rejection", confidence=0.90, match="you_are_insult")
    if re.search(r"\b(?:shut up|stop talking|nobody asked you)\b", cleaned):
        return _state("angry", 0.82, "user_rejection", confidence=0.88, match="dismissive_command")
    if re.search(r"\b(?:chán|ghét|bực|khó chịu)\b.*\b(?:mày|bạn|case|cậu)\b", cleaned):
        return _state("angry", 0.86, "user_rejection", confidence=0.90, match="vietnamese_rejection")
    if re.search(r"\b(?:mày|bạn|case|cậu)\b.*\b(?:chán|vô dụng|ngu|phiền)\b", cleaned):
        return _state("angry", 0.86, "user_rejection", confidence=0.90, match="vietnamese_you_insult")
    if re.search(r"\b(?:im đi|i am đi|câm đi)\b", cleaned):
        return _state("angry", 0.82, "user_rejection", confidence=0.88, match="vietnamese_shut_up")
    return None


def _user_sadness(cleaned: str) -> EmotionState | None:
    if re.search(r"\bi am\s+(?:sad|tired|stressed)\b", cleaned):
        return _state("sad", 0.65, "user_sadness", confidence=0.82, match="english_user_feeling")
    if re.search(r"\bi feel bad\b", cleaned):
        return _state("sad", 0.65, "user_sadness", confidence=0.82, match="english_feel_bad")
    if re.search(r"\b(?:hôm nay tao buồn|mình buồn quá|tao mệt|mình mệt quá)\b", cleaned):
        return _state("sad", 0.68, "user_sadness", confidence=0.82, match="vietnamese_sadness")
    return None


def _user_praise(cleaned: str) -> EmotionState | None:
    if re.search(r"\b(?:good job|nice|well done)\b", cleaned):
        return _state("amused", 0.62, "user_praise", confidence=0.80, match="short_praise")
    if re.search(r"\byou are\s+(?:funny|smart)\b", cleaned):
        return _state("amused", 0.65, "user_praise", confidence=0.82, match="you_are_praise")
    if re.search(r"\b(?:mày giỏi|hay đấy|tốt đấy)\b", cleaned):
        return _state("amused", 0.65, "user_praise", confidence=0.80, match="vietnamese_praise")
    return None


def _humor_request(cleaned: str) -> EmotionState | None:
    if re.search(r"\b(?:tell me a joke|roast me|say something funny|make fun of me)\b", cleaned):
        return _state("sarcastic", 0.65, "humor_request", confidence=0.82, match="english_humor_request")
    if re.search(r"\b(?:kể chuyện cười|khịa tao đi)\b", cleaned):
        return _state("sarcastic", 0.65, "humor_request", confidence=0.80, match="vietnamese_humor_request")
    return None


_RULES = (
    _requested_style,
    _user_rejection,
    _user_sadness,
    _user_praise,
    _humor_request,
)


def detect_emotion(text: str) -> EmotionState:
    cleaned = normalize_emotion_text(text)
    if not cleaned:
        return default_emotion_state()
    for detector in _RULES:
        state = detector(cleaned)
        if state is not None:
            return state
    return default_emotion_state()


def _detect_deescalation(cleaned: str) -> str | None:
    patterns = (
        (r"\bcalm down\b", "calm_down"),
        (r"\bdo not be angry\b", "do_not_be_angry"),
        (r"\bdon t be angry\b", "dont_be_angry"),
        (r"\bđừng giận\b", "vietnamese_dont_be_angry"),
        (r"\bbình tĩnh đi\b", "vietnamese_calm_down"),
    )
    for pattern, name in patterns:
        if re.search(pattern, cleaned):
            return name
    return None


def _detect_emotion_meta_question(cleaned: str) -> str | None:
    patterns = (
        (r"\bwhy are you (?:angry|mad|upset)\b", "why_are_you_angry"),
        (r"\bare you still (?:angry|mad|upset)\b", "are_you_still_angry"),
        (r"\bare you not (?:angry|mad|upset)\b", "are_you_not_angry"),
        (r"\byou not (?:angry|mad|upset)\b", "you_not_angry"),
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


def _memory_deescalation_state(memory: EmotionMemory, *, match: str) -> EmotionState:
    previous = memory.last_emotion or "deadpan"
    emotion = "annoyed" if previous in {"angry", "annoyed"} else "deadpan"
    intensity = 0.45 if emotion == "annoyed" else 0.35
    logger.info(
        "EMOTION_DEESCALATE: from=%s to=%s intensity=%.2f",
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
