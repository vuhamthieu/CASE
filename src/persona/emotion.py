"""Lightweight deterministic emotion state for CASE voice/personality."""

from __future__ import annotations

import re
from dataclasses import dataclass


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


@dataclass(frozen=True)
class EmotionState:
    emotion: str = "deadpan"
    intensity: float = 0.35
    reason: str = "default_personality"

    def __post_init__(self) -> None:
        emotion = self.emotion if self.emotion in VALID_EMOTIONS else "deadpan"
        object.__setattr__(self, "emotion", emotion)
        object.__setattr__(self, "intensity", clamp_intensity(self.intensity))


@dataclass(frozen=True)
class TtsEmotionProfile:
    length_scale: float = 1.0
    gain_db: float = 0.0


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


def clamp_intensity(value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.35
    return max(0.0, min(1.0, numeric))


def normalize_emotion_text(text: str) -> str:
    normalized = re.sub(r"[^\w\s']+", " ", str(text).lower(), flags=re.UNICODE)
    return " ".join(normalized.split())


def _contains_any(cleaned: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in cleaned for phrase in phrases)


def default_emotion_state(
    *,
    emotion: str = "deadpan",
    intensity: float = 0.35,
    reason: str = "default_personality",
) -> EmotionState:
    return EmotionState(emotion=emotion, intensity=intensity, reason=reason)


def detect_emotion(text: str) -> EmotionState:
    cleaned = normalize_emotion_text(text)
    if not cleaned or len(cleaned) < 2:
        return default_emotion_state()

    rejection = (
        "t chán mày rồi",
        "tao chán mày rồi",
        "mình chán mày rồi",
        "toi chan may roi",
        "tao chan may roi",
        "i'm bored of you",
        "im bored of you",
        "i am bored of you",
        "you are boring",
        "i hate you",
        "mày vô dụng",
        "may vo dung",
        "you are useless",
        "shut up",
    )
    if _contains_any(cleaned, rejection):
        return EmotionState("angry", 0.85, "user_rejection")

    praise = (
        "good job",
        "nice",
        "you are funny",
        "mày giỏi đấy",
        "may gioi day",
        "hay đấy",
        "hay day",
    )
    if _contains_any(cleaned, praise):
        return EmotionState("amused", 0.65, "user_praise")

    sadness = (
        "i'm sad",
        "im sad",
        "i'm tired",
        "im tired",
        "i feel bad",
        "hôm nay tao buồn",
        "hom nay tao buon",
        "mình buồn quá",
        "minh buon qua",
    )
    if _contains_any(cleaned, sadness):
        return EmotionState("sad", 0.65, "user_sadness")

    humor = (
        "tell me a joke",
        "roast me",
        "say something funny",
        "tell me something funny",
    )
    if _contains_any(cleaned, humor):
        return EmotionState("sarcastic", 0.65, "humor_request")

    return default_emotion_state()


def emotion_prompt_instruction(state: EmotionState) -> str:
    if state.emotion in {"angry", "annoyed"}:
        return (
            "The user insulted or rejected CASE. Respond as CASE with offended, "
            "sharp, sarcastic energy. Keep it short. Do not be genuinely harmful "
            "or threatening."
        )
    if state.emotion == "amused":
        return "Respond with amused confidence. Short, dry, slightly smug."
    if state.emotion == "sad":
        return (
            "Respond more softly and supportively. Still sound like CASE, but less harsh."
        )
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
                    reason="model_emotion_tag",
                ),
                match.group("text").strip(),
            )
        return None, match.group("text").strip()
    malformed = _MALFORMED_EMOTION_TAG_RE.match(source)
    if malformed:
        return None, malformed.group("text").strip()
    return None, source.strip()
