"""Central PCM loading and playback-format normalization for CASE."""

from __future__ import annotations

import logging
import math
import wave
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)


def _as_int16(audio: np.ndarray) -> np.ndarray:
    samples = np.asarray(audio)
    if samples.dtype == np.int16:
        return samples.astype("<i2", copy=True)
    if np.issubdtype(samples.dtype, np.floating):
        values = np.nan_to_num(samples.astype(np.float64))
        if values.size and float(np.max(np.abs(values))) <= 1.5:
            values *= 32767.0
        return np.clip(np.rint(values), -32768, 32767).astype("<i2")
    if samples.dtype == np.uint8:
        return ((samples.astype(np.int16) - 128) << 8).astype("<i2")
    if np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)
        scale = max(abs(info.min), abs(info.max)) / 32768.0
        values = samples.astype(np.float64) / max(scale, 1.0)
        return np.clip(np.rint(values), -32768, 32767).astype("<i2")
    raise ValueError(f"unsupported audio dtype: {samples.dtype}")


def ensure_2d_audio(audio: np.ndarray) -> np.ndarray:
    """Return audio shaped as ``(frames, channels)``."""
    samples = np.asarray(audio)
    if samples.ndim == 1:
        return samples[:, None]
    if samples.ndim == 2:
        return samples
    raise ValueError(f"expected 1-D or 2-D audio, got shape {samples.shape}")


def load_wav_int16(path: Path) -> tuple[np.ndarray, int, int]:
    """Load a WAV as C-contiguous int16 ``(frames, channels)`` audio."""
    try:
        from scipy.io import wavfile

        sample_rate, audio = wavfile.read(Path(path))
    except ImportError:
        with wave.open(str(Path(path)), "rb") as source:
            sample_rate = source.getframerate()
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            payload = source.readframes(source.getnframes())
        if sample_width == 1:
            audio = np.frombuffer(payload, dtype=np.uint8)
        elif sample_width == 2:
            audio = np.frombuffer(payload, dtype="<i2")
        elif sample_width == 4:
            audio = np.frombuffer(payload, dtype="<i4")
        else:
            raise ValueError(f"unsupported WAV sample width: {sample_width * 8}-bit")
        audio = audio.reshape(-1, channels)
    samples = ensure_2d_audio(_as_int16(audio))
    return np.ascontiguousarray(samples, dtype="<i2"), int(sample_rate), samples.shape[1]


def convert_channels(audio: np.ndarray, target_channels: int) -> np.ndarray:
    """Convert channel count, duplicating mono or averaging multi-channel audio."""
    if target_channels < 1:
        raise ValueError("target_channels must be at least 1")
    samples = ensure_2d_audio(_as_int16(audio))
    if samples.shape[1] == target_channels:
        return np.ascontiguousarray(samples, dtype="<i2")
    mono = np.rint(samples.astype(np.float32).mean(axis=1)).astype("<i2")
    if target_channels == 1:
        return np.ascontiguousarray(mono[:, None], dtype="<i2")
    return np.ascontiguousarray(
        np.repeat(mono[:, None], target_channels, axis=1),
        dtype="<i2",
    )


def resample_audio(
    audio: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    """Resample along the frame axis while preserving channels and int16 PCM."""
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("audio sample rates must be positive")
    samples = ensure_2d_audio(_as_int16(audio))
    if source_rate == target_rate:
        return np.ascontiguousarray(samples, dtype="<i2")
    try:
        from scipy.signal import resample_poly

        divisor = math.gcd(source_rate, target_rate)
        converted = resample_poly(
            samples.astype(np.float32),
            target_rate // divisor,
            source_rate // divisor,
            axis=0,
        )
    except ImportError:
        logger.warning(
            "AUDIO_FORMAT: scipy unavailable; using linear resampling fallback"
        )
        output_frames = max(1, int(round(len(samples) * target_rate / source_rate)))
        source_positions = np.arange(len(samples), dtype=np.float64)
        target_positions = np.linspace(
            0.0,
            max(0.0, len(samples) - 1.0),
            output_frames,
        )
        converted = np.column_stack(
            [
                np.interp(target_positions, source_positions, samples[:, channel])
                for channel in range(samples.shape[1])
            ]
        )
    return np.ascontiguousarray(
        np.clip(np.rint(converted), -32768, 32767).astype("<i2")
    )


def normalize_for_playback(
    audio: np.ndarray,
    source_rate: int,
    target_rate: int,
    target_channels: int,
) -> np.ndarray:
    """Return C-contiguous int16 audio matching an output stream exactly."""
    resampled = resample_audio(audio, source_rate, target_rate)
    return np.ascontiguousarray(
        convert_channels(resampled, target_channels),
        dtype="<i2",
    )
