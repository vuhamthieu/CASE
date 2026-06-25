"""Shared mate-style wake acknowledgement selection and cached-WAV mapping."""

from __future__ import annotations

import logging
import random
import re
import threading
from pathlib import Path

from src.audio.audio_format import convert_channels, load_wav_int16
from src.audio.output_device import play_int16_mono
from src.audio.wake_ack_audio import prepare_wake_ack_audio
from src.config import defaults
from src.config.env import get_bool, get_float, get_int, get_str


ACK_WAV_NAMES = {
    "What?": "what.wav",
    "Yeah?": "yeah.wav",
    "I'm listening.": "im_listening.wav",
    "You called?": "you_called.wav",
    "Yes!": "yes.wav",
    "Go on.": "go_on.wav",
    "I'm here.": "im_here.wav",
    "Say that again?": "say_that_again.wav",
    "Say that again.": "say_that_again.wav",
    "Still with you.": "still_with_you.wav",
}
WAKE_ACK_TEXT_BY_KEY = {
    **defaults.WAKE_ACK_TEXTS,
    **defaults.OPTIONAL_SHORT_WAKE_ACKS,
    "im_listening": "I'm listening.",
    "go_on": "Go on.",
    "im_here": "I'm here.",
    "say_that_again": "Say that again?",
    "still_with_you": "Still with you.",
}
WAKE_ACK_KEY_BY_TEXT = {text: key for key, text in WAKE_ACK_TEXT_BY_KEY.items()}
SHORT_WAKE_ACK_TEXTS = set(defaults.OPTIONAL_SHORT_WAKE_ACKS.values())


class WakeAcknowledgementSelector:
    def __init__(self) -> None:
        allow_short = get_bool(
            "WAKE_ACK_ALLOW_SHORT_INTERJECTIONS",
            defaults.WAKE_ACK_ALLOW_SHORT_INTERJECTIONS,
        )
        configured = get_str(
            "WAKE_ACK_POOL",
            get_str(
                "REALTIME_WAKE_ACK_POOL", "|".join(defaults.WAKE_ACK_POOL)
            ),
        )
        configured_items = [
            item.strip()
            for item in re.split(r"[|,]", configured)
            if item.strip()
        ]
        pool = [
            text
            for item in configured_items
            if (text := self._text_for_config_item(item)) is not None
        ]
        if not pool:
            pool = [
                defaults.WAKE_ACK_TEXTS[key]
                for key in defaults.DEFAULT_WAKE_ACK_POOL
            ]
        if not allow_short:
            default_texts = {
                defaults.WAKE_ACK_TEXTS[key]
                for key in defaults.DEFAULT_WAKE_ACK_POOL
            }
            pool = [
                item
                for item in pool
                if item not in SHORT_WAKE_ACK_TEXTS or item in default_texts
            ]
        self.pool = pool or [
            defaults.WAKE_ACK_TEXTS[key] for key in defaults.DEFAULT_WAKE_ACK_POOL
        ]
        logging.getLogger(__name__).info(
            "WAKE_ACK_POOL_ACTIVE: %s",
            list(defaults.DEFAULT_WAKE_ACK_POOL),
        )
        self.random_enabled = get_bool(
            "WAKE_ACK_RANDOM_ENABLED",
            get_bool(
                "REALTIME_WAKE_ACK_RANDOM_ENABLED",
                defaults.WAKE_ACK_RANDOM_ENABLED,
            ),
        )
        self.avoid_repeat = get_bool(
            "WAKE_ACK_AVOID_REPEAT",
            get_bool(
                "REALTIME_WAKE_ACK_AVOID_REPEAT",
                defaults.WAKE_ACK_AVOID_REPEAT,
            ),
        )
        self.default_text = get_str(
            "REALTIME_WAKE_ACK_TEXT", defaults.REALTIME_WAKE_ACK_TEXT
        )
        if not allow_short and self.default_text in SHORT_WAKE_ACK_TEXTS:
            self.default_text = defaults.WAKE_ACK_TEXTS[
                defaults.DEFAULT_WAKE_ACK_POOL[0]
            ]
        self._last: str | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _text_for_config_item(item: str) -> str | None:
        active_texts = set(defaults.WAKE_ACK_TEXTS.values())
        if item in defaults.WAKE_ACK_TEXTS:
            return defaults.WAKE_ACK_TEXTS[item]
        if item in active_texts:
            return item
        return None

    def choose(self) -> str:
        with self._lock:
            if not self.random_enabled or not self.pool:
                selected = self.default_text
            else:
                choices = self.pool
                if self.avoid_repeat and len(choices) > 1 and self._last in choices:
                    choices = [item for item in choices if item != self._last]
                selected = random.choice(choices)
            self._last = selected
            return selected


selector = WakeAcknowledgementSelector()
_missing_wav_warnings: set[Path] = set()
_migration_checked: set[Path] = set()


def choose_wake_ack() -> str:
    return selector.choose()


def wake_ack_wav_name(text: str) -> str:
    text = WAKE_ACK_TEXT_BY_KEY.get(text, text)
    if text in ACK_WAV_NAMES:
        return ACK_WAV_NAMES[text]
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "wake_ack"
    return f"{slug}.wav"


def wake_ack_wav_path(text: str, directory: str) -> Path:
    return Path(directory) / wake_ack_wav_name(text)


def wake_ack_key(text: str) -> str:
    return WAKE_ACK_KEY_BY_TEXT.get(
        WAKE_ACK_TEXT_BY_KEY.get(text, text),
        re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "wake_ack",
    )


def _runtime_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return Path(__file__).resolve().parents[2] / expanded


def migrate_legacy_generated_wavs(cached_directory: str | Path) -> None:
    """Copy old root wake-ack WAVs into the canonical generated directory once."""
    generated_dir = _runtime_path(Path(cached_directory))
    if generated_dir in _migration_checked:
        return
    _migration_checked.add(generated_dir)
    if generated_dir.name != "generated":
        return
    legacy_dir = generated_dir.parent
    generated_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(__name__)
    active_files = {
        wake_ack_wav_name(defaults.WAKE_ACK_TEXTS[key])
        for key in defaults.DEFAULT_WAKE_ACK_POOL
    }
    for filename in sorted(active_files):
        source = legacy_dir / filename
        destination = generated_dir / filename
        if destination.exists() or not source.is_file():
            continue
        try:
            import shutil

            shutil.copy2(source, destination)
            log.info(
                "WAKE_ACK_MIGRATE: copied legacy %s -> generated/%s",
                filename,
                filename,
            )
        except OSError as exc:
            log.warning(
                "WAKE_ACK_MIGRATE: failed legacy %s -> %s: %s",
                source,
                destination,
                exc,
            )


def pad_audio_to_minimum(audio, sample_rate: int, minimum_sec: float = 1.1):
    """Return `(audio, padded)` with enough trailing silence for safe playback."""
    import numpy as np

    minimum_frames = int(round(float(sample_rate) * minimum_sec))
    if len(audio) >= minimum_frames:
        return audio, False
    return np.pad(audio, (0, minimum_frames - len(audio))), True


def _play_wake_ack_path(text: str, path: Path, source: str) -> bool:
    path = _runtime_path(path)
    if not path.is_file():
        if path not in _missing_wav_warnings:
            if source == "cached_wav":
                logging.getLogger(__name__).warning(
                    "WAKE_ACK_MISSING_GENERATED: key=%s path=%s",
                    wake_ack_key(text),
                    path,
                )
            logging.getLogger(__name__).warning(
                "WAKE_ACK: %s missing path=%s",
                source,
                path,
            )
            _missing_wav_warnings.add(path)
        return False
    try:
        import numpy as np

        loaded_audio, sample_rate, source_channels = load_wav_int16(path)
        audio = convert_channels(loaded_audio, 1)[:, 0]
        logging.getLogger(__name__).info(
            "WAKE_ACK_AUDIO: source=%s path=%s",
            source,
            path,
        )

        pad_short = get_bool(
            "WAKE_ACK_PAD_SHORT_AUDIO", defaults.WAKE_ACK_PAD_SHORT_AUDIO
        )
        if pad_short and source != "recorded_wav":
            audio, padded = prepare_wake_ack_audio(audio, int(sample_rate))
        else:
            padded = False
        duration = len(audio) / float(sample_rate)
        post_guard = get_float(
            "WAKE_ACK_POST_PLAYBACK_GUARD_SEC",
            defaults.WAKE_ACK_POST_PLAYBACK_GUARD_SEC,
        )
        safe_mode = get_bool(
            "WAKE_ACK_FORCE_BLOCKING_PLAYBACK",
            defaults.WAKE_ACK_FORCE_BLOCKING_PLAYBACK,
        )
        extra_tail_sec = max(
            0.0,
            get_int(
                "WAKE_ACK_EXTRA_RUNTIME_TAIL_MS",
                defaults.WAKE_ACK_EXTRA_RUNTIME_TAIL_MS,
            )
            / 1000.0,
        )
        logging.getLogger(__name__).info(
            "WAKE_ACK_AUDIO: playing source=%s path=%s duration=%.3fs "
            "padded=%s safe_mode=%s",
            source,
            path,
            duration,
            padded,
            safe_mode,
        )

        result = play_int16_mono(
            np.ascontiguousarray(audio),
            int(sample_rate),
            post_guard_sec=post_guard,
            safe_mode=safe_mode,
            extra_tail_sec=extra_tail_sec,
        )
        duration_delta = abs(result["duration_out"] - result["duration_in"])
        duration_ratio = duration_delta / max(result["duration_in"], 1e-9)
        logging.getLogger(__name__).info(
            "WAKE_ACK_AUDIO_FORMAT: path=%s source_rate=%s source_channels=%s "
            "target_rate=%s target_channels=%s frames_in=%s frames_out=%s "
            "duration_in=%.3fs duration_out=%.3fs resampled=%s",
            path,
            sample_rate,
            source_channels,
            result["sample_rate"],
            result["channels"],
            result["frames_in"],
            result["frames_out"],
            result["duration_in"],
            result["duration_out"],
            result["resampled"],
        )
        if duration_ratio > 0.02:
            logging.getLogger(__name__).warning(
                "WAKE_ACK_AUDIO_FORMAT: duration changed by %.1f%% path=%s",
                duration_ratio * 100.0,
                path,
            )
        logging.getLogger(__name__).info(
            "WAKE_ACK_AUDIO: drained device=%r sample_rate=%s channels=%s "
            "safe_mode=%s underflow=%s",
            result["device_name"],
            result["sample_rate"],
            result["channels"],
            result["safe_mode"],
            result["underflow"],
        )
        return True
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "WAKE_ACK: %s playback failed: %s", source, exc
        )
        return False


def play_wake_ack_asset(
    text: str,
    *,
    mode: str | None = None,
    recorded_directory: str | None = None,
    cached_directory: str | None = None,
    fallback_mode: str | None = None,
) -> bool:
    """Play a selected recorded/generated asset according to fallback policy."""
    selected_mode = (
        mode or get_str("WAKE_ACK_MODE", defaults.WAKE_ACK_MODE)
    ).lower()
    recorded_directory = recorded_directory or get_str(
        "WAKE_ACK_RECORDED_DIR", defaults.WAKE_ACK_RECORDED_DIR
    )
    cached_directory = cached_directory or get_str(
        "WAKE_ACK_WAV_DIR", defaults.WAKE_ACK_WAV_DIR
    )
    fallback_mode = (
        fallback_mode
        or get_str("WAKE_ACK_FALLBACK_MODE", defaults.WAKE_ACK_FALLBACK_MODE)
    ).lower()
    recorded_enabled = get_bool(
        "WAKE_ACK_RECORDED_ENABLED", defaults.WAKE_ACK_RECORDED_ENABLED
    )
    if selected_mode == "recorded_wav" and not recorded_enabled:
        logging.getLogger(__name__).info(
            "WAKE_ACK: recorded_wav disabled; using cached_wav"
        )
        selected_mode = "cached_wav"

    sources: list[tuple[str, str]] = []
    if selected_mode == "recorded_wav":
        sources.append(("recorded_wav", recorded_directory))
        if fallback_mode == "cached_wav":
            sources.append(("cached_wav", cached_directory))
    elif selected_mode == "cached_wav":
        sources.append(("cached_wav", cached_directory))
    elif selected_mode not in {"local_tts", "beep", "none"}:
        logging.getLogger(__name__).warning(
            "WAKE_ACK: unsupported asset mode=%s",
            selected_mode,
        )

    if any(source == "cached_wav" for source, _ in sources):
        migrate_legacy_generated_wavs(cached_directory)

    for index, (source, directory) in enumerate(sources):
        logging.getLogger(__name__).info(
            "WAKE_ACK_SELECT: key=%s source=%s",
            wake_ack_key(text),
            source,
        )
        if _play_wake_ack_path(
            text,
            wake_ack_wav_path(text, directory),
            source,
        ):
            return True
        if source == "recorded_wav" and index + 1 < len(sources):
            fallback_source, fallback_directory = sources[index + 1]
            fallback_path = _runtime_path(
                wake_ack_wav_path(text, fallback_directory)
            )
            logging.getLogger(__name__).warning(
                "WAKE_ACK: recorded_wav missing, falling back to %s path=%s",
                fallback_source,
                fallback_path,
            )
    return False


def play_cached_wake_ack(text: str, directory: str) -> bool:
    """Backward-compatible generated-cache playback wrapper."""
    return _play_wake_ack_path(
        text,
        wake_ack_wav_path(text, directory),
        "cached_wav",
    )


def play_wake_ack_beep() -> bool:
    """Play the final local fallback without involving any speech backend."""
    try:
        import numpy as np

        sample_rate = 24_000
        timeline = np.arange(int(sample_rate * 0.16), dtype=np.float32) / sample_rate
        audio = np.rint(0.12 * 32767.0 * np.sin(2 * np.pi * 660 * timeline)).astype(
            np.int16
        )
        play_int16_mono(audio, sample_rate, safe_mode=True)
        return True
    except Exception as exc:
        logging.getLogger(__name__).warning("WAKE_ACK: beep failed: %s", exc)
        return False
