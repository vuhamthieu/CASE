"""Reusable local audio playback for CASE TTS and cached acknowledgements."""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from typing import Any

import numpy as np

from src.audio.audio_format import ensure_2d_audio, normalize_for_playback
from src.audio.output_device import configured_output_device, query_output_device
from src.config import defaults
from src.config.env import get_bool, get_float, get_int, get_str


logger = logging.getLogger(__name__)


class AudioPlaybackManager:
    """Own one reusable blocking output stream and serialize local playback."""

    def __init__(self) -> None:
        self.backend = get_str(
            "AUDIO_PLAYBACK_BACKEND", defaults.AUDIO_PLAYBACK_BACKEND
        ).lower()
        self.requested_rate = get_int(
            "AUDIO_OUTPUT_SAMPLE_RATE", defaults.AUDIO_OUTPUT_SAMPLE_RATE
        )
        self.requested_channels = get_int(
            "AUDIO_OUTPUT_CHANNELS", defaults.AUDIO_OUTPUT_CHANNELS
        )
        self.latency = get_str(
            "AUDIO_PLAYBACK_LATENCY", defaults.AUDIO_PLAYBACK_LATENCY
        )
        self.blocksize = get_int(
            "AUDIO_PLAYBACK_BLOCKSIZE", defaults.AUDIO_PLAYBACK_BLOCKSIZE
        )
        self.keep_stream_open = get_bool(
            "AUDIO_PLAYBACK_KEEP_STREAM_OPEN",
            defaults.AUDIO_PLAYBACK_KEEP_STREAM_OPEN,
        )
        self.tail_guard_sec = get_float(
            "AUDIO_PLAYBACK_TAIL_GUARD_SEC",
            defaults.AUDIO_PLAYBACK_TAIL_GUARD_SEC,
        )
        self.short_sound_safe_mode = get_bool(
            "AUDIO_SHORT_SOUND_SAFE_MODE", defaults.AUDIO_SHORT_SOUND_SAFE_MODE
        )
        self.short_sound_threshold_sec = get_float(
            "AUDIO_SHORT_SOUND_THRESHOLD_SEC",
            defaults.AUDIO_SHORT_SOUND_THRESHOLD_SEC,
        )
        self.short_sound_extra_tail_sec = max(
            0.0,
            get_int(
                "AUDIO_SHORT_SOUND_EXTRA_TAIL_MS",
                defaults.AUDIO_SHORT_SOUND_EXTRA_TAIL_MS,
            )
            / 1000.0,
        )
        self.safe_latency = get_str(
            "WAKE_ACK_PLAYBACK_LATENCY", defaults.WAKE_ACK_PLAYBACK_LATENCY
        )
        self.retry_on_underflow = get_bool(
            "AUDIO_PLAYBACK_RETRY_ON_UNDERFLOW",
            defaults.AUDIO_PLAYBACK_RETRY_ON_UNDERFLOW,
        )
        self._lock = threading.RLock()
        self._stream = None
        self._device = None
        self._device_name = None
        self._sample_rate = self.requested_rate
        self._channels = max(1, self.requested_channels)
        self._stream_latency = None
        self._force_safe_mode = False
        self._underflow_warned = False
        self._closed = False

    def start(self) -> None:
        with self._lock:
            self._closed = False
            if self.backend == "aplay":
                logger.info(
                    "AUDIO_PLAYBACK: backend=aplay device=default keep_open=False"
                )
                return
            if self.backend != "sounddevice":
                raise ValueError(f"unsupported audio playback backend: {self.backend}")
            self._ensure_sounddevice_stream()

    def play(
        self,
        audio: bytes | np.ndarray,
        sample_rate: int,
        *,
        tail_guard_sec: float | None = None,
        safe_mode: bool = False,
        extra_tail_sec: float = 0.0,
    ) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                self._closed = False
            if self.backend == "aplay":
                return self._play_aplay(audio, sample_rate)
            if self.backend != "sounddevice":
                raise ValueError(f"unsupported audio playback backend: {self.backend}")

            source = self._source_array(audio)
            source_2d = ensure_2d_audio(source)
            source_channels = source_2d.shape[1]
            frames_in = len(source_2d)
            if sample_rate <= 0:
                raise ValueError("source sample rate must be positive")
            duration_in = frames_in / float(sample_rate)
            is_short_sound = duration_in <= self.short_sound_threshold_sec
            effective_safe_mode = (
                safe_mode
                or self._force_safe_mode
                or (self.short_sound_safe_mode and is_short_sound)
            )
            if is_short_sound and extra_tail_sec <= 0:
                extra_tail_sec = self.short_sound_extra_tail_sec
            self._ensure_sounddevice_stream(safe_mode=effective_safe_mode)
            payload = normalize_for_playback(
                source_2d,
                sample_rate,
                self._sample_rate,
                self._channels,
            )
            frames_out = len(payload)
            duration_out = frames_out / float(self._sample_rate)
            if extra_tail_sec > 0:
                tail_frames = int(round(self._sample_rate * extra_tail_sec))
                payload = np.pad(payload, ((0, max(0, tail_frames)), (0, 0)))
            payload = np.ascontiguousarray(payload, dtype="<i2")

            duration = len(payload) / float(self._sample_rate)
            guard = self.tail_guard_sec if tail_guard_sec is None else tail_guard_sec
            if duration >= 1.2 and tail_guard_sec is None:
                guard = 0.0

            try:
                self._stream.start()
                underflowed = bool(
                    self._stream.write(payload.tobytes())
                )
                if guard > 0:
                    time.sleep(guard)
                self._stream.stop()
            except Exception:
                self._discard_stream()
                raise

            result = {
                "configured_device": self._device,
                "device_name": self._device_name,
                "sample_rate": self._sample_rate,
                "channels": self._channels,
                "duration": duration,
                "underflow": underflowed,
                "safe_mode": effective_safe_mode,
                "source_rate": sample_rate,
                "source_channels": source_channels,
                "frames_in": frames_in,
                "frames_out": frames_out,
                "duration_in": duration_in,
                "duration_out": duration_out,
                "resampled": sample_rate != self._sample_rate,
                "short_sound_safe_mode": (
                    is_short_sound and effective_safe_mode
                ),
            }
            logger.info(
                "AUDIO_PLAYBACK: short_sound_safe_mode=%s underflow=%s "
                "safe_mode=%s drained_duration=%.3fs",
                result["short_sound_safe_mode"],
                underflowed,
                effective_safe_mode,
                duration,
            )
            if underflowed:
                if not self._underflow_warned:
                    logger.warning(
                        "AUDIO_PLAYBACK: output underflow detected; future playback "
                        "will use safe mode"
                    )
                    self._underflow_warned = True
                if self.retry_on_underflow:
                    self._force_safe_mode = True
                    self._discard_stream()
            if not self.keep_stream_open:
                self._discard_stream()
            return result

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._discard_stream()

    def _ensure_sounddevice_stream(self, *, safe_mode: bool = False) -> None:
        requested_latency = self.safe_latency if safe_mode else self.latency
        if self._stream is not None and self._stream_latency == requested_latency:
            return
        if self._stream is not None:
            self._discard_stream()
        import sounddevice as sd

        device, info = query_output_device()
        default_rate = int(round(float(info["default_samplerate"])))
        max_channels = int(info.get("max_output_channels", 0))
        if max_channels < 1:
            raise RuntimeError(f"output has no channels: {info!r}")

        rates = [self.requested_rate]
        if default_rate not in rates:
            rates.append(default_rate)
        channels = [min(max(1, self.requested_channels), max_channels)]
        if "MAX98357A" in str(info.get("name", "")).upper() and max_channels >= 2:
            channels.insert(0, 2)
        elif max_channels >= 2 and 2 not in channels:
            channels.append(2)
        if 1 not in channels:
            channels.append(1)

        last_error = None
        for rate in rates:
            for channel_count in channels:
                try:
                    stream = sd.RawOutputStream(
                        device=device,
                        samplerate=rate,
                        channels=channel_count,
                        dtype="int16",
                        latency=requested_latency,
                        blocksize=max(0, self.blocksize),
                    )
                    self._stream = stream
                    self._device = configured_output_device()
                    self._device_name = info.get("name", device)
                    self._sample_rate = rate
                    self._channels = channel_count
                    self._stream_latency = requested_latency
                    logger.info(
                        "AUDIO_PLAYBACK: backend=sounddevice device=%r "
                        "sample_rate=%s channels=%s latency=%s keep_open=%s",
                        self._device_name,
                        self._sample_rate,
                        self._channels,
                        requested_latency,
                        self.keep_stream_open,
                    )
                    return
                except Exception as exc:
                    last_error = exc
        raise RuntimeError(
            f"could not open sounddevice output {info.get('name', device)!r}: "
            f"{last_error}"
        )

    @staticmethod
    def _source_array(audio: bytes | np.ndarray) -> np.ndarray:
        if isinstance(audio, bytes):
            samples = np.frombuffer(audio, dtype="<i2").copy()
        else:
            samples = np.asarray(audio)
        if not samples.size:
            raise ValueError("audio is empty")
        return samples

    def _play_aplay(
        self,
        audio: bytes | np.ndarray,
        sample_rate: int,
    ) -> dict[str, Any]:
        samples = normalize_for_playback(
            self._source_array(audio), sample_rate, sample_rate, 1
        )[:, 0]
        result = subprocess.run(
            [
                "aplay",
                "-D",
                "default",
                "-r",
                str(sample_rate),
                "-f",
                "S16_LE",
                "-c",
                "1",
                "-t",
                "raw",
                "-",
            ],
            input=samples.tobytes(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            error = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"aplay failed with code {result.returncode}: {error}")
        duration = len(samples) / float(sample_rate)
        return {
            "configured_device": "default",
            "device_name": "aplay:default",
            "sample_rate": sample_rate,
            "channels": 1,
            "duration": duration,
            "underflow": False,
        }

    def _discard_stream(self) -> None:
        stream = self._stream
        self._stream = None
        self._stream_latency = None
        if stream is None:
            return
        try:
            if stream.active:
                stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass


_manager: AudioPlaybackManager | None = None
_manager_lock = threading.Lock()


def get_playback_manager() -> AudioPlaybackManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = AudioPlaybackManager()
        return _manager


def close_playback_manager() -> None:
    global _manager
    with _manager_lock:
        manager = _manager
        _manager = None
    if manager is not None:
        manager.close()
