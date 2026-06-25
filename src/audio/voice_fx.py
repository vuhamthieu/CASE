"""Pi-friendly, stateful PCM effects for CASE model speech.

Pitch/formant conversion and neural voice conversion are intentionally future
offline laptop/server research, not Raspberry Pi realtime v1 dependencies.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.config import defaults
from src.config.env import get_bool, get_str


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceFXPreset:
    highpass_hz: float
    lowpass_hz: float
    warmth_gain_db: float
    presence_gain_db: float
    compressor_threshold_db: float
    compressor_ratio: float
    makeup_gain_db: float
    saturation_drive: float
    wet: float = 1.0


VOICE_FX_PRESETS = {
    "bypass": None,
    "cinematic_robot_v1": VoiceFXPreset(70, 9000, 2.5, -1.0, -18, 2.5, 1.5, 1.15),
    "dry_computer_v1": VoiceFXPreset(90, 7500, 1.0, -2.0, -20, 3.0, 1.0, 1.05),
    "dark_robot_v1": VoiceFXPreset(60, 6500, 4.0, -2.5, -22, 3.5, 1.5, 1.25),
}


class VoiceFX:
    def __init__(self, sample_rate: int = 24_000, preset: str = "cinematic_robot_v1"):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if preset not in VOICE_FX_PRESETS:
            raise ValueError(f"unknown voice FX preset: {preset}")
        self.sample_rate = sample_rate
        self.preset_name = preset
        self.preset = VOICE_FX_PRESETS[preset]
        self._hp_input = self._hp_output = 0.0
        self._warm_low = self._warm_high = 0.0
        self._presence_low = self._presence_high = 0.0
        self._lowpass = self._envelope = 0.0

    def process_int16_mono(self, pcm_bytes: bytes) -> bytes:
        if self.preset is None or not pcm_bytes:
            return pcm_bytes
        if len(pcm_bytes) % 2:
            logger.warning("CASE_VOICE_FX: odd PCM byte count; bypassing chunk")
            return pcm_bytes
        try:
            dry = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
            wet = self._process_float(dry)
            mixed = dry * (1.0 - self.preset.wet) + wet * self.preset.wet
            return np.clip(np.rint(mixed * 32767.0), -32768, 32767).astype(
                "<i2"
            ).tobytes()
        except Exception as exc:
            logger.warning("CASE_VOICE_FX: processing failed; bypassing: %s", exc)
            return pcm_bytes

    def _process_float(self, samples: np.ndarray) -> np.ndarray:
        p = self.preset
        assert p is not None
        output = np.empty_like(samples)
        hp_alpha = self._alpha(p.highpass_hz)
        warm_lo_a, warm_hi_a = self._alpha(180), self._alpha(260)
        presence_lo_a, presence_hi_a = self._alpha(2000), self._alpha(4000)
        lowpass_a = self._alpha(p.lowpass_hz)
        warmth = 10 ** (p.warmth_gain_db / 20) - 1
        presence = 10 ** (p.presence_gain_db / 20) - 1
        threshold = 10 ** (p.compressor_threshold_db / 20)
        makeup = 10 ** (p.makeup_gain_db / 20)
        attack = math.exp(-1 / (0.010 * self.sample_rate))
        release = math.exp(-1 / (0.120 * self.sample_rate))

        for index, sample in enumerate(samples):
            value = float(sample)
            hp = hp_alpha * (self._hp_output + value - self._hp_input)
            self._hp_input, self._hp_output = value, hp
            self._warm_low = warm_lo_a * self._warm_low + (1 - warm_lo_a) * hp
            self._warm_high = warm_hi_a * self._warm_high + (1 - warm_hi_a) * hp
            self._presence_low = presence_lo_a * self._presence_low + (1 - presence_lo_a) * hp
            self._presence_high = presence_hi_a * self._presence_high + (1 - presence_hi_a) * hp
            shaped = (
                hp
                + warmth * (self._warm_high - self._warm_low)
                + presence * (self._presence_high - self._presence_low)
            )
            self._lowpass = lowpass_a * self._lowpass + (1 - lowpass_a) * shaped
            level = abs(self._lowpass)
            coefficient = attack if level > self._envelope else release
            self._envelope = coefficient * self._envelope + (1 - coefficient) * level
            gain = 1.0
            if self._envelope > threshold:
                compressed = threshold + (self._envelope - threshold) / p.compressor_ratio
                gain = compressed / max(self._envelope, 1e-8)
            driven = self._lowpass * gain * makeup
            output[index] = math.tanh(driven * p.saturation_drive) / math.tanh(
                p.saturation_drive
            )
        return output

    def _alpha(self, frequency: float) -> float:
        return math.exp(-2 * math.pi * frequency / self.sample_rate)


def create_voice_fx_from_config(sample_rate: int) -> Optional[VoiceFX]:
    if not get_bool("CASE_VOICE_FX_ENABLED", defaults.CASE_VOICE_FX_ENABLED):
        logger.info("CASE_VOICE_FX: disabled")
        return None
    preset = get_str("CASE_VOICE_FX_PRESET", defaults.CASE_VOICE_FX_PRESET)
    try:
        result = VoiceFX(sample_rate, preset)
    except Exception as exc:
        logger.warning("CASE_VOICE_FX: invalid configuration; disabled: %s", exc)
        return None
    logger.info("CASE_VOICE_FX: enabled preset=%s sample_rate=%s", preset, sample_rate)
    return result
