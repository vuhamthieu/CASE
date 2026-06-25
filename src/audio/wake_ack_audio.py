"""Padding, fades, and inspection for cached CASE wake acknowledgements."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.config import defaults


SILENCE_THRESHOLD = 64
SAFE_PEAK = int(32767 * 0.90)


@dataclass(frozen=True)
class WakeAckStats:
    sample_rate: int
    duration_sec: float
    voiced_ms: float
    leading_silence_ms: float
    trailing_silence_ms: float
    peak: int
    peak_dbfs: float
    clipped: bool
    passed: bool


def silence_frame_counts(samples: np.ndarray) -> tuple[int, int]:
    absolute = np.abs(samples.astype(np.int32))
    audible = np.flatnonzero(absolute > SILENCE_THRESHOLD)
    if not audible.size:
        return len(samples), len(samples)
    leading = int(audible[0])
    trailing = int(len(samples) - audible[-1] - 1)
    return leading, trailing


def prepare_wake_ack_audio(
    audio: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, bool]:
    """Return safe int16 mono audio with required edge padding and fades."""
    samples = np.asarray(audio)
    if samples.ndim != 1:
        raise ValueError(f"expected mono wake acknowledgement, got {samples.shape}")
    if samples.dtype != np.int16:
        samples = np.clip(np.rint(samples), -32768, 32767).astype(np.int16)
    else:
        samples = samples.copy()
    if not samples.size:
        raise ValueError("wake acknowledgement is empty")

    leading, trailing = silence_frame_counts(samples)
    speech_start = min(leading, len(samples))
    speech_end = max(speech_start, len(samples) - trailing)
    speech = samples[speech_start:speech_end].astype(np.float32)
    if not speech.size:
        raise ValueError("wake acknowledgement contains only silence")

    fade_in = min(
        len(speech),
        int(round(sample_rate * defaults.WAKE_ACK_FADE_IN_MS / 1000)),
    )
    fade_out = min(
        len(speech),
        int(round(sample_rate * defaults.WAKE_ACK_FADE_OUT_MS / 1000)),
    )
    if fade_in:
        speech[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    if fade_out:
        speech[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)
    peak = float(np.max(np.abs(speech)))
    if peak > SAFE_PEAK:
        speech *= SAFE_PEAK / peak
    speech = np.clip(np.rint(speech), -32768, 32767).astype(np.int16)

    required_leading = int(
        round(sample_rate * defaults.WAKE_ACK_PRE_SILENCE_MS / 1000)
    )
    required_trailing = int(
        round(sample_rate * defaults.WAKE_ACK_POST_SILENCE_MS / 1000)
    )
    output = np.concatenate(
        (
            np.zeros(max(required_leading, leading), dtype=np.int16),
            speech,
            np.zeros(max(required_trailing, trailing), dtype=np.int16),
        )
    )
    minimum = int(round(sample_rate * defaults.WAKE_ACK_MIN_DURATION_MS / 1000))
    if len(output) < minimum:
        output = np.pad(output, (0, minimum - len(output)))

    modified = not np.array_equal(output, samples)
    return np.ascontiguousarray(output, dtype=np.int16), modified


def inspect_wake_ack(audio: np.ndarray, sample_rate: int) -> WakeAckStats:
    samples = np.asarray(audio, dtype=np.int16).reshape(-1)
    leading, trailing = silence_frame_counts(samples)
    peak = int(np.max(np.abs(samples.astype(np.int32)))) if samples.size else 0
    peak_dbfs = 20.0 * math.log10(peak / 32767.0) if peak else float("-inf")
    duration = len(samples) / float(sample_rate) if sample_rate else 0.0
    leading_ms = leading * 1000.0 / sample_rate if sample_rate else 0.0
    trailing_ms = trailing * 1000.0 / sample_rate if sample_rate else 0.0
    voiced_samples = max(0, len(samples) - leading - trailing)
    voiced_ms = voiced_samples * 1000.0 / sample_rate if sample_rate else 0.0
    clipped = peak >= 32767
    passed = (
        sample_rate > 0
        and duration >= defaults.WAKE_ACK_MIN_DURATION_MS / 1000.0
        and voiced_ms + 0.1 >= defaults.WAKE_ACK_MIN_VOICED_MS
        and leading_ms + 0.1 >= defaults.WAKE_ACK_PRE_SILENCE_MS
        and trailing_ms + 0.1 >= defaults.WAKE_ACK_POST_SILENCE_MS
        and peak > 0
        and not clipped
    )
    return WakeAckStats(
        sample_rate=sample_rate,
        duration_sec=duration,
        voiced_ms=voiced_ms,
        leading_silence_ms=leading_ms,
        trailing_silence_ms=trailing_ms,
        peak=peak,
        peak_dbfs=peak_dbfs,
        clipped=clipped,
        passed=passed,
    )
