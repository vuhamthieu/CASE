"""Cached WAV thinking filler selection and playback."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np

from src.audio.audio_format import convert_channels, load_wav_int16
from src.audio.output_device import play_int16_mono
from src.config import defaults
from src.config.env import get_bool, get_float, get_int


logger = logging.getLogger(__name__)


def runtime_path(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded
    return Path(__file__).resolve().parents[2] / expanded


class ThinkingFillerSelector:
    """Choose an existing filler WAV, preferring short reactions."""

    def __init__(
        self,
        directory: str | Path,
        keys: list[str] | tuple[str, ...],
        preferred_keys: list[str] | tuple[str, ...],
    ) -> None:
        self.directory = runtime_path(directory)
        self.keys = list(keys)
        self.preferred_keys = [key for key in preferred_keys if key in self.keys]
        self._last_key: str | None = None

    def existing_assets(self) -> dict[str, Path]:
        assets: dict[str, Path] = {}
        for key in self.keys:
            path = self.directory / f"{key}.wav"
            if path.is_file():
                assets[key] = path
        return assets

    def choose(self) -> tuple[str, Path] | None:
        assets = self.existing_assets()
        if not assets:
            return None

        preferred = [key for key in self.preferred_keys if key in assets]
        pool = preferred or list(assets)
        if self._last_key in pool and len(pool) > 1:
            pool = [key for key in pool if key != self._last_key]
        key = random.choice(pool)
        self._last_key = key
        return key, assets[key]


def play_thinking_filler_wav(path: str | Path) -> dict:
    """Play one cached filler WAV through CASE's serialized local audio path."""
    resolved = runtime_path(path)
    audio, sample_rate, source_channels = load_wav_int16(resolved)
    mono = convert_channels(audio, 1)[:, 0]
    mono = np.ascontiguousarray(mono, dtype=np.int16)
    safe_mode = get_bool(
        "THINKING_FILLER_FORCE_SAFE_PLAYBACK",
        defaults.WAKE_ACK_FORCE_BLOCKING_PLAYBACK,
    )
    extra_tail_sec = max(
        0.0,
        get_int(
            "THINKING_FILLER_EXTRA_RUNTIME_TAIL_MS",
            defaults.WAKE_ACK_EXTRA_RUNTIME_TAIL_MS,
        )
        / 1000.0,
    )
    post_guard = get_float(
        "THINKING_FILLER_POST_PLAYBACK_GUARD_SEC",
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
        "THINKING_FILLER_AUDIO_FORMAT: path=%s source_rate=%s source_channels=%s "
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
