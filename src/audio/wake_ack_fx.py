"""Lightweight offline-only performance shaping for wake acknowledgements."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np


logger = logging.getLogger(__name__)
TARGET_PEAK = 32767.0 * (10.0 ** (-3.0 / 20.0))
_fallback_logged = False


def _resample_to_length(audio: np.ndarray, output_frames: int) -> np.ndarray:
    global _fallback_logged
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    output_frames = max(1, int(output_frames))
    if len(samples) == output_frames:
        return samples.copy()
    try:
        from scipy.signal import resample

        return np.asarray(resample(samples, output_frames), dtype=np.float32)
    except ImportError:
        if not _fallback_logged:
            logger.warning(
                "WAKE_ACK_FX: scipy unavailable; using linear interpolation fallback"
            )
            _fallback_logged = True
        source_positions = np.arange(len(samples), dtype=np.float64)
        target_positions = np.linspace(
            0.0,
            max(0.0, len(samples) - 1.0),
            output_frames,
        )
        return np.interp(target_positions, source_positions, samples).astype(
            np.float32
        )


def pitch_shift_light(
    audio: np.ndarray,
    sample_rate: int,
    semitones: float,
) -> np.ndarray:
    """Apply a small resampling-based pitch lift suitable for short reactions."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not samples.size or abs(semitones) < 1e-6:
        return samples.copy()
    ratio = 2.0 ** (float(semitones) / 12.0)
    return _resample_to_length(samples, round(len(samples) / ratio))


def tempo_scale_light(audio: np.ndarray, factor: float) -> np.ndarray:
    """Make a short acknowledgement faster when ``factor`` is above one."""
    if factor <= 0:
        raise ValueError("tempo factor must be positive")
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not samples.size or abs(factor - 1.0) < 1e-6:
        return samples.copy()
    return _resample_to_length(samples, round(len(samples) / float(factor)))


def apply_wake_ack_fx(
    audio: np.ndarray,
    sample_rate: int,
    style: Mapping[str, float],
) -> np.ndarray:
    """Apply pitch, tempo, gain, fades, and a -3 dBFS peak normalization."""
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not samples.size:
        raise ValueError("wake acknowledgement audio is empty")
    samples = pitch_shift_light(
        samples,
        sample_rate,
        float(style.get("pitch_shift_semitones", 0.0)),
    )
    samples = tempo_scale_light(samples, float(style.get("tempo", 1.0)))
    samples *= 10.0 ** (float(style.get("gain_db", 0.0)) / 20.0)

    fade_in = min(len(samples), int(round(sample_rate * 0.012)))
    fade_out = min(len(samples), int(round(sample_rate * 0.050)))
    if fade_in:
        samples[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    if fade_out:
        samples[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)

    peak = float(np.max(np.abs(samples)))
    if peak > 0:
        samples *= TARGET_PEAK / peak
    return np.ascontiguousarray(
        np.clip(np.rint(samples), -32768, 32767).astype("<i2")
    )
