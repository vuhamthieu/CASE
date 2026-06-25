"""Offline cleanup and CASE-style processing for recorded wake acknowledgements."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping

import numpy as np

from src.audio.wake_ack_fx import pitch_shift_light, tempo_scale_light


logger = logging.getLogger(__name__)
_pitch_skip_logged = False

WAKE_ACK_PROCESSING_PRESETS = {
    "clean": {
        "highpass_hz": 80,
        "compressor_threshold_db": -20,
        "compressor_ratio": 2.5,
        "makeup_gain_db": 1.5,
        "low_mid_gain_db": 1.0,
        "presence_gain_db": -0.5,
        "saturation_drive": 1.05,
        "pitch_shift_semitones": 0.0,
        "tempo": 1.0,
    },
    "case_robot": {
        "highpass_hz": 75,
        "compressor_threshold_db": -22,
        "compressor_ratio": 3.0,
        "makeup_gain_db": 2.0,
        "low_mid_gain_db": 2.5,
        "presence_gain_db": -1.2,
        "saturation_drive": 1.18,
        "pitch_shift_semitones": -1.0,
        "tempo": 1.0,
    },
    "surprised_ack": {
        "highpass_hz": 90,
        "compressor_threshold_db": -18,
        "compressor_ratio": 2.2,
        "makeup_gain_db": 2.0,
        "low_mid_gain_db": 1.0,
        "presence_gain_db": 0.5,
        "saturation_drive": 1.10,
        "pitch_shift_semitones": 1.5,
        "tempo": 1.05,
    },
}

LEADING_SILENCE_MS = 120
TRAILING_SILENCE_MS = 350
FADE_IN_MS = 8
FADE_OUT_MS = 30
TARGET_PEAK = 10.0 ** (-3.0 / 20.0)


def _to_float_mono(audio: np.ndarray) -> np.ndarray:
    samples = np.asarray(audio)
    if samples.ndim == 2:
        samples = samples.astype(np.float32).mean(axis=1)
    elif samples.ndim != 1:
        raise ValueError(f"expected mono/stereo audio, got shape {samples.shape}")
    if samples.dtype == np.int16:
        return samples.astype(np.float32) / 32768.0
    values = samples.astype(np.float32)
    if values.size and float(np.max(np.abs(values))) > 1.5:
        values /= 32768.0
    return np.clip(values, -1.0, 1.0)


def _light_trim(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak <= 0:
        raise ValueError("recording contains only silence")
    edge = max(1, int(round(sample_rate * 0.15)))
    noise = np.concatenate((samples[:edge], samples[-edge:]))
    noise_rms = float(np.sqrt(np.mean(noise * noise))) if noise.size else 0.0
    threshold = max(0.004, noise_rms * 2.5, peak * 0.018)
    audible = np.flatnonzero(np.abs(samples) >= threshold)
    if not audible.size:
        raise ValueError("no clear speech found in recording")
    context = int(round(sample_rate * 0.05))
    start = max(0, int(audible[0]) - context)
    end = min(len(samples), int(audible[-1]) + context + 1)
    return samples[start:end].copy()


def _highpass(samples: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0:
        return samples.copy()
    dt = 1.0 / sample_rate
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    alpha = rc / (rc + dt)
    output = np.empty_like(samples)
    output[0] = samples[0]
    for index in range(1, len(samples)):
        output[index] = alpha * (
            output[index - 1] + samples[index] - samples[index - 1]
        )
    return output


def _lowpass(samples: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    dt = 1.0 / sample_rate
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    alpha = dt / (rc + dt)
    output = np.empty_like(samples)
    output[0] = samples[0]
    for index in range(1, len(samples)):
        output[index] = output[index - 1] + alpha * (
            samples[index] - output[index - 1]
        )
    return output


def _gentle_noise_gate(samples: np.ndarray) -> np.ndarray:
    level = np.abs(samples)
    threshold = max(10.0 ** (-52.0 / 20.0), float(np.percentile(level, 15)) * 2.0)
    gain = np.clip(level / max(threshold, 1e-6), 0.20, 1.0)
    return samples * gain.astype(np.float32)


def _compress(samples: np.ndarray, threshold_db: float, ratio: float) -> np.ndarray:
    magnitude = np.maximum(np.abs(samples), 1e-7)
    input_db = 20.0 * np.log10(magnitude)
    output_db = np.where(
        input_db > threshold_db,
        threshold_db + (input_db - threshold_db) / max(ratio, 1.0),
        input_db,
    )
    gain = 10.0 ** ((output_db - input_db) / 20.0)
    return samples * gain.astype(np.float32)


def _eq(
    samples: np.ndarray,
    sample_rate: int,
    low_mid_gain_db: float,
    presence_gain_db: float,
) -> np.ndarray:
    low_180 = _lowpass(samples, sample_rate, 180.0)
    low_900 = _lowpass(samples, sample_rate, 900.0)
    low_mid = low_900 - low_180
    low_1800 = _lowpass(samples, sample_rate, 1800.0)
    low_4800 = _lowpass(samples, sample_rate, 4800.0)
    presence = low_4800 - low_1800
    low_mid_mix = 10.0 ** (low_mid_gain_db / 20.0) - 1.0
    presence_mix = 10.0 ** (presence_gain_db / 20.0) - 1.0
    return samples + low_mid * low_mid_mix + presence * presence_mix


def _optional_pitch_shift(
    samples: np.ndarray,
    sample_rate: int,
    semitones: float,
) -> np.ndarray:
    global _pitch_skip_logged
    if abs(semitones) < 1e-6:
        return samples
    try:
        import scipy.signal  # noqa: F401
    except ImportError:
        if not _pitch_skip_logged:
            logger.warning(
                "WAKE_ACK_PROCESS: scipy unavailable; skipping optional pitch shift"
            )
            _pitch_skip_logged = True
        return samples
    return pitch_shift_light(samples, sample_rate, semitones)


def process_recorded_wake_ack(
    audio: np.ndarray,
    sample_rate: int,
    preset: Mapping[str, float],
) -> np.ndarray:
    """Return final mono int16 audio with cleanup, character, and safe padding."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    samples = _to_float_mono(audio)
    samples = samples - float(np.mean(samples))
    samples = _light_trim(samples, sample_rate)
    samples = _highpass(samples, sample_rate, float(preset["highpass_hz"]))
    samples = _gentle_noise_gate(samples)
    samples = _compress(
        samples,
        float(preset["compressor_threshold_db"]),
        float(preset["compressor_ratio"]),
    )
    samples *= 10.0 ** (float(preset["makeup_gain_db"]) / 20.0)
    samples = _eq(
        samples,
        sample_rate,
        float(preset["low_mid_gain_db"]),
        float(preset["presence_gain_db"]),
    )
    drive = max(1.0, float(preset["saturation_drive"]))
    samples = np.tanh(samples * drive) / math.tanh(drive)
    samples = _optional_pitch_shift(
        samples,
        sample_rate,
        float(preset["pitch_shift_semitones"]),
    )
    samples = tempo_scale_light(samples, float(preset["tempo"]))

    peak = float(np.max(np.abs(samples)))
    if peak > 0:
        samples *= TARGET_PEAK / peak
    fade_in = min(len(samples), int(round(sample_rate * FADE_IN_MS / 1000)))
    fade_out = min(len(samples), int(round(sample_rate * FADE_OUT_MS / 1000)))
    if fade_in:
        samples[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    if fade_out:
        samples[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)

    leading = np.zeros(int(round(sample_rate * LEADING_SILENCE_MS / 1000)), np.float32)
    trailing = np.zeros(int(round(sample_rate * TRAILING_SILENCE_MS / 1000)), np.float32)
    output = np.concatenate((leading, samples, trailing))
    return np.ascontiguousarray(
        np.clip(np.rint(output * 32767.0), -32768, 32767).astype("<i2")
    )
