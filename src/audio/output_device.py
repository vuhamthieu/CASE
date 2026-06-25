"""Shared blocking playback helpers for CASE's configured speaker."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config import defaults
from src.config.env import get_str


def configured_output_device() -> int | str | None:
    """Return the configured PortAudio device index/name, or the system default."""
    value = get_str(
        "AUDIO_OUTPUT_DEVICE",
        get_str("CASE_AUDIO_OUTPUT_DEVICE", defaults.AUDIO_OUTPUT_DEVICE),
    ).strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def query_output_device() -> tuple[int | str | None, dict[str, Any]]:
    """Resolve the configured/default PortAudio output and its capabilities."""
    import sounddevice as sd

    device = configured_output_device()
    try:
        info = sd.query_devices(device, "output")
    except Exception as exc:
        raise RuntimeError(
            f"audio output {device!r} is unavailable or has no output channels; "
            "run `python3 scripts/test_audio_output.py --list` and choose a "
            "device with output channels"
        ) from exc
    return device, info


def play_int16_mono(
    audio: bytes | np.ndarray,
    sample_rate: int,
    *,
    post_guard_sec: float = 0.03,
    safe_mode: bool = False,
    extra_tail_sec: float = 0.0,
) -> dict[str, Any]:
    """Compatibility wrapper around the process-wide playback manager."""
    from src.audio.playback_manager import get_playback_manager

    return get_playback_manager().play(
        audio,
        sample_rate,
        tail_guard_sec=post_guard_sec,
        safe_mode=safe_mode,
        extra_tail_sec=extra_tail_sec,
    )
