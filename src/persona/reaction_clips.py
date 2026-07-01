"""Pre-generated CASE reaction clip manifest and selection."""

from __future__ import annotations

import json
import logging
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.audio.audio_format import convert_channels, load_wav_int16
from src.audio.output_device import play_int16_mono
from src.config import defaults
from src.config.env import get_bool, get_float, get_int, get_str
from src.persona.emotion import EmotionState


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReactionClip:
    clip_id: str
    text: str
    tts_text: str
    emotion: str
    path: Path


@dataclass(frozen=True)
class ReactionSelection:
    clip_id: str
    text: str
    tts_text: str
    path: Path
    reason: str
    emotion: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_runtime_path(path: str | Path, *, root: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (root or repo_root()) / candidate


def disabled_clip_ids(value: str | None = None) -> set[str]:
    raw = defaults.CASE_REACTION_DISABLED_CLIPS if value is None else value
    if value is None:
        raw = get_str("CASE_REACTION_DISABLED_CLIPS", defaults.CASE_REACTION_DISABLED_CLIPS)
    return {
        part.strip().lower()
        for part in str(raw or "").split(",")
        if part.strip()
    }


def wav_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as source:
        frames = source.getnframes()
        rate = source.getframerate()
    if rate <= 0:
        return 0.0
    return frames / float(rate)


def load_reaction_manifest(
    manifest_path: str | Path,
    *,
    root: Path | None = None,
    disabled_clips: set[str] | None = None,
    min_duration_sec: float | None = None,
) -> dict[str, ReactionClip]:
    resolved = resolve_runtime_path(manifest_path, root=root)
    if not resolved.is_file():
        logger.info("REACTION_CLIPS_LOADED: count=0")
        logger.info("REACTION_CLIP_MISSING: clip=manifest path=%s", resolved)
        return {}
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("REACTION_CLIPS_LOADED: count=0 invalid_manifest=%s", exc)
        return {}

    clips: dict[str, ReactionClip] = {}
    raw_clips = data.get("clips", {})
    if not isinstance(raw_clips, dict):
        logger.warning("REACTION_CLIPS_LOADED: count=0 invalid_manifest=clips_not_dict")
        return {}

    disabled = disabled_clip_ids() if disabled_clips is None else {item.lower() for item in disabled_clips}
    min_duration = (
        get_float("CASE_REACTION_MIN_DURATION_SEC", defaults.CASE_REACTION_MIN_DURATION_SEC)
        if min_duration_sec is None
        else float(min_duration_sec)
    )
    for clip_id, item in raw_clips.items():
        if not isinstance(item, dict):
            continue
        normalized_id = str(clip_id).strip().lower()
        if item.get("enabled", True) is False:
            logger.info("REACTION_CLIP_DISABLED: clip=%s", clip_id)
            continue
        if normalized_id in disabled:
            logger.info("REACTION_CLIP_DISABLED: clip=%s", clip_id)
            continue
        text = str(item.get("text", "")).strip()
        tts_text = str(item.get("tts_text", text)).strip()
        emotion = str(item.get("emotion", "")).strip().lower()
        raw_path = str(item.get("path", "")).strip()
        if not clip_id or not text or not raw_path:
            continue
        if normalized_id == "one_sec" or text.casefold().strip(".!? ") == "one sec" or tts_text.casefold().strip(".!? ") == "one sec":
            logger.info("REACTION_CLIP_DISABLED: clip=%s", clip_id)
            continue
        path = resolve_runtime_path(raw_path, root=root)
        if not path.is_file():
            logger.info("REACTION_CLIP_MISSING: clip=%s path=%s", clip_id, path)
            continue
        try:
            duration = wav_duration_sec(path)
        except Exception as exc:
            logger.info("REACTION_SKIP: reason=invalid_wav clip=%s error=%s", clip_id, exc)
            continue
        if duration < min_duration:
            logger.info(
                "REACTION_CLIP_TOO_SHORT: clip=%s duration=%.3f min=%.3f action=skip",
                clip_id,
                duration,
                min_duration,
            )
            continue
        clips[str(clip_id)] = ReactionClip(
            clip_id=str(clip_id),
            text=text,
            tts_text=tts_text or text,
            emotion=emotion,
            path=path,
        )

    logger.info("REACTION_CLIPS_LOADED: count=%s", len(clips))
    return clips


def strip_leading_reaction_duplicate(text: str, reaction_text: str, clip_id: str) -> str:
    source = str(text or "").lstrip()
    reaction = str(reaction_text or "").strip()
    if not source or not reaction:
        return source
    if source.casefold().startswith(reaction.casefold()):
        stripped = source[: len(reaction)]
        rest = source[len(reaction) :].lstrip()
        rest = rest.lstrip("-–—,:; ")
        logger.info(
            "REACTION_DUPLICATE_STRIP: clip=%s stripped_text=%r",
            clip_id,
            stripped,
        )
        return rest
    return source


class ReactionClipSelector:
    def __init__(
        self,
        clips: dict[str, ReactionClip],
        *,
        min_intensity: float = 0.70,
        cooldown_sec: float = 8.0,
        max_per_turn: int = 1,
    ) -> None:
        self.clips = dict(clips)
        self.min_intensity = float(min_intensity)
        self.cooldown_sec = float(cooldown_sec)
        self.max_per_turn = int(max_per_turn)
        self._last_played_at = 0.0
        self._played_by_turn: dict[int, int] = {}

    def choose(
        self,
        state: EmotionState,
        *,
        user_text: str = "",
        turn_id: int,
        now: float | None = None,
    ) -> ReactionSelection | None:
        if self.max_per_turn <= 0:
            logger.info("REACTION_SKIP: reason=disabled")
            return None
        if self._played_by_turn.get(int(turn_id), 0) >= self.max_per_turn:
            logger.info("REACTION_SKIP: reason=max_per_turn")
            return None
        current = time.monotonic() if now is None else float(now)
        if self._last_played_at and current - self._last_played_at < self.cooldown_sec:
            logger.info("REACTION_SKIP: reason=cooldown")
            return None

        selected = self._select_for_state(state)
        if selected is None:
            logger.info("REACTION_SKIP: reason=no_matching_clip")
            return None
        clip_id, reason = selected
        clip = self.clips.get(clip_id)
        if clip is None or not clip.path.is_file():
            logger.info("REACTION_SKIP: reason=clip_missing clip=%s", clip_id)
            return None

        self._played_by_turn[int(turn_id)] = self._played_by_turn.get(int(turn_id), 0) + 1
        self._last_played_at = current
        logger.info(
            "REACTION_SELECT: clip=%s emotion=%s reason=%s turn=%s",
            clip.clip_id,
            state.emotion,
            reason,
            turn_id,
        )
        return ReactionSelection(
            clip_id=clip.clip_id,
            text=clip.text,
            tts_text=clip.tts_text,
            path=clip.path,
            reason=reason,
            emotion=clip.emotion,
        )

    def _select_for_state(self, state: EmotionState) -> tuple[str, str] | None:
        if state.reason == "user_rejection" and state.emotion == "angry":
            if state.intensity < 0.80:
                return None
            return self._first_available(("seriously", "fine"), "angry_user_rejection")
        if state.reason == "requested_emotion_style" and state.emotion == "angry":
            return "fine", "requested_angry_style"
        if (
            state.reason == "emotion_meta_question"
            and state.source == "memory"
            and state.emotion in {"annoyed", "angry"}
        ):
            return ("seriously" if "seriously" in self.clips else "fine"), "anger_meta_question"
        if state.reason == "emotion_deescalation":
            return "fine", "emotion_deescalation"
        if state.reason == "humor_request" and state.emotion == "sarcastic":
            if state.intensity >= 0.65:
                return "wow", "sarcastic_humor"
            return None
        if state.reason == "user_praise" and state.emotion == "amused":
            if state.intensity >= 0.60:
                return "nice", "amused_praise"
            return None
        return None

    def _first_available(
        self,
        clip_ids: tuple[str, ...],
        reason: str,
    ) -> tuple[str, str] | None:
        for clip_id in clip_ids:
            if clip_id in self.clips:
                return clip_id, reason
        return None


def play_reaction_clip_wav(path: str | Path) -> dict:
    resolved = resolve_runtime_path(path)
    audio, sample_rate, source_channels = load_wav_int16(resolved)
    mono = convert_channels(audio, 1)[:, 0]
    mono = np.ascontiguousarray(mono, dtype=np.int16)
    safe_mode = get_bool(
        "CASE_REACTION_SAFE_MODE",
        defaults.CASE_REACTION_SAFE_MODE,
    )
    extra_tail_sec = max(
        0.0,
        get_int(
            "REACTION_CLIP_EXTRA_RUNTIME_TAIL_MS",
            defaults.WAKE_ACK_EXTRA_RUNTIME_TAIL_MS,
        )
        / 1000.0,
    )
    post_guard = get_float(
        "REACTION_CLIP_POST_PLAYBACK_GUARD_SEC",
        defaults.WAKE_ACK_POST_PLAYBACK_GUARD_SEC,
    )
    result = play_int16_mono(
        mono,
        int(sample_rate),
        post_guard_sec=post_guard,
        safe_mode=safe_mode,
        extra_tail_sec=extra_tail_sec,
    )
    logger.info(
        "REACTION_CLIP_AUDIO_FORMAT: path=%s source_rate=%s source_channels=%s "
        "target_rate=%s target_channels=%s duration_in=%.3fs duration_out=%.3fs "
        "resampled=%s",
        resolved,
        sample_rate,
        source_channels,
        result.get("sample_rate"),
        result.get("channels"),
        result.get("duration_in", 0.0),
        result.get("duration_out", 0.0),
        result.get("resampled"),
    )
    return result
