"""Shared input-device selection for CASE microphone consumers."""

from __future__ import annotations

from typing import Any

from src.config import defaults
from src.config.env import get_str


def configured_input_device() -> int | str | None:
    """Return the configured PortAudio input index/name, or the system default."""
    value = get_str(
        "CASE_AUDIO_INPUT_DEVICE",
        get_str("CASE_MIC_DEVICE", defaults.CASE_AUDIO_INPUT_DEVICE),
    ).strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def query_input_device() -> tuple[int | str | None, dict[str, Any]]:
    import sounddevice as sd

    device = configured_input_device()
    return device, sd.query_devices(device, "input")
