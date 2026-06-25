"""Shared-mic capture and raw PCM playback for Gemini Live."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from math import gcd
from typing import Callable, Optional

import numpy as np
from scipy.signal import resample_poly

from .realtime_config import (
    REALTIME_AUDIO_CHUNK_MS,
    REALTIME_INPUT_SAMPLE_RATE,
    REALTIME_OUTPUT_SAMPLE_RATE,
)


logger = logging.getLogger(__name__)


class RealtimeAudioInput:
    """Provide exact 20 ms, 16 kHz mono int16 chunks.

    Main runtime passes STTEngine.audio_queue so no second microphone is opened.
    Tests may omit shared_queue to open the default input device directly.
    """

    def __init__(
        self,
        shared_queue: Optional["queue.Queue[bytes]"] = None,
        device=None,
    ) -> None:
        self.shared_queue = shared_queue
        self.device = device
        self._queue: "queue.Queue[bytes]" = shared_queue or queue.Queue(maxsize=64)
        self._stream = None
        self._buffer = bytearray()
        self._running = False
        self._source_rate = REALTIME_INPUT_SAMPLE_RATE
        self._channels = 1
        self._up = 1
        self._down = 1
        self.chunk_bytes = int(
            REALTIME_INPUT_SAMPLE_RATE * REALTIME_AUDIO_CHUNK_MS / 1000
        ) * 2

    async def start(self) -> None:
        self._running = True
        if self.shared_queue is not None:
            logger.info("REALTIME: reusing classic STT microphone stream")
            return

        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("sounddevice is required for realtime audio") from exc

        device_info = sd.query_devices(self.device, "input")
        self._source_rate = int(round(device_info["default_samplerate"]))
        self._channels = 1
        divisor = gcd(self._source_rate, REALTIME_INPUT_SAMPLE_RATE)
        self._up = REALTIME_INPUT_SAMPLE_RATE // divisor
        self._down = self._source_rate // divisor
        blocksize = max(
            1,
            int(self._source_rate * REALTIME_AUDIO_CHUNK_MS / 1000),
        )

        def callback(indata, frames, time_info, status) -> None:
            if status:
                logger.debug("REALTIME input status: %s", status)
            raw = np.frombuffer(indata, dtype=np.int16)
            if self._channels > 1:
                raw = raw.reshape(-1, self._channels).mean(axis=1)
            if self._source_rate != REALTIME_INPUT_SAMPLE_RATE:
                raw = resample_poly(raw.astype(np.float32), self._up, self._down)
            pcm = np.clip(np.rint(raw), -32768, 32767).astype(np.int16).tobytes()
            try:
                self._queue.put_nowait(pcm)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(pcm)
                except queue.Empty:
                    pass

        self._stream = sd.RawInputStream(
            device=self.device,
            samplerate=self._source_rate,
            blocksize=blocksize,
            channels=self._channels,
            dtype="int16",
            callback=callback,
            latency="high",
        )
        await asyncio.to_thread(self._stream.start)
        logger.info(
            "REALTIME: standalone mic started rate=%s resample_to=%s",
            self._source_rate,
            REALTIME_INPUT_SAMPLE_RATE,
        )

    async def read_chunk(self, timeout: float = 0.25) -> Optional[bytes]:
        return await asyncio.to_thread(self._read_chunk_blocking, timeout)

    def _read_chunk_blocking(self, timeout: float) -> Optional[bytes]:
        while self._running and len(self._buffer) < self.chunk_bytes:
            try:
                self._buffer.extend(self._queue.get(timeout=timeout))
            except queue.Empty:
                return None
        if len(self._buffer) < self.chunk_bytes:
            return None
        chunk = bytes(self._buffer[: self.chunk_bytes])
        del self._buffer[: self.chunk_bytes]
        return chunk

    def drain(self) -> None:
        self._buffer.clear()
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    async def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            await asyncio.to_thread(self._stream.stop)
            await asyncio.to_thread(self._stream.close)
            self._stream = None


class RealtimeAudioOutput:
    """Continuous callback-driven PCM output with an ordered byte buffer."""

    def __init__(
        self,
        device=None,
        on_playback_start: Optional[Callable[[], None]] = None,
        on_playback_drained: Optional[Callable[[], None]] = None,
    ) -> None:
        self.device = device
        self.on_playback_start = on_playback_start
        self.on_playback_drained = on_playback_drained
        self._stream = None
        self._running = False
        self._playing = threading.Event()
        self._lock = threading.Lock()
        self._buffer = bytearray()
        self._stopping = False
        self._drained_reported = True
        self._last_underrun_log_at = 0.0
        self._first_playback_reported = False
        self.playback_started_at = 0.0
        self._output_rate = REALTIME_OUTPUT_SAMPLE_RATE
        self._up = 1
        self._down = 1
        self._model_chunk_bytes = int(
            REALTIME_OUTPUT_SAMPLE_RATE * REALTIME_AUDIO_CHUNK_MS / 1000
        ) * 2

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._playing.is_set() or bool(self._buffer)

    async def start(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("sounddevice is required for realtime playback") from exc

        device_info = sd.query_devices(self.device, "output")
        device_rate = int(round(device_info["default_samplerate"]))
        rates = [REALTIME_OUTPUT_SAMPLE_RATE]
        if device_rate != REALTIME_OUTPUT_SAMPLE_RATE:
            rates.append(device_rate)

        def callback(outdata, frames, time_info, status) -> None:
            requested = frames * 2
            now = time.monotonic()
            with self._lock:
                available = min(requested, len(self._buffer))
                payload = bytes(self._buffer[:available])
                del self._buffer[:available]
                remaining = len(self._buffer)
            if available < requested:
                payload += b"\x00" * (requested - available)
                if self._playing.is_set() and now - self._last_underrun_log_at >= 1.0:
                    logger.info("REALTIME_AUDIO: playback callback underrun")
                    self._last_underrun_log_at = now
            outdata[:] = payload
            if available and not self._playing.is_set():
                self._playing.set()
                self.playback_started_at = now
                logger.info("REALTIME_AUDIO: playback started")
                if self.on_playback_start is not None:
                    self.on_playback_start()
                self._first_playback_reported = True
            if self._playing.is_set() and remaining == 0 and available < requested:
                self._playing.clear()
                with self._lock:
                    if not self._drained_reported:
                        self._drained_reported = True
                        logger.info("REALTIME_AUDIO: playback drained")
                        if self.on_playback_drained is not None:
                            self.on_playback_drained()

        last_error = None
        for rate in rates:
            try:
                self._stream = sd.RawOutputStream(
                    device=self.device,
                    samplerate=rate,
                    channels=1,
                    dtype="int16",
                    latency="high",
                    blocksize=max(1, int(rate * REALTIME_AUDIO_CHUNK_MS / 1000)),
                    callback=callback,
                )
                await asyncio.to_thread(self._stream.start)
                self._output_rate = rate
                break
            except Exception as exc:
                last_error = exc
                self._stream = None
        if self._stream is None:
            raise RuntimeError(f"could not open realtime audio output: {last_error}")

        divisor = gcd(REALTIME_OUTPUT_SAMPLE_RATE, self._output_rate)
        self._up = self._output_rate // divisor
        self._down = REALTIME_OUTPUT_SAMPLE_RATE // divisor
        self._running = True
        logger.info(
            "REALTIME_AUDIO: output stream opened sample_rate=%s channels=1 dtype=int16",
            self._output_rate,
        )

    def enqueue(self, audio: bytes) -> None:
        if not audio:
            return
        if self._output_rate != REALTIME_OUTPUT_SAMPLE_RATE:
            samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
            samples = resample_poly(samples, self._up, self._down)
            audio = np.clip(np.rint(samples), -32768, 32767).astype(np.int16).tobytes()
        with self._lock:
            self._buffer.extend(audio)
            self._stopping = False
            self._drained_reported = False
        logger.debug("REALTIME_AUDIO: queued model audio bytes=%s", len(audio))

    def request_stop(self) -> bool:
        with self._lock:
            if self._stopping or (not self._buffer and not self._playing.is_set()):
                logger.debug(
                    "REALTIME: playback already stopped; ignoring duplicate stop"
                )
                return False
            self._stopping = True
            self._buffer.clear()
            self._drained_reported = True
        self._playing.clear()
        logger.info("REALTIME: playback stop requested")
        if self.on_playback_drained is not None:
            self.on_playback_drained()
        return True

    def clear(self) -> bool:
        return self.request_stop()

    async def wait_until_drained(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_playing:
                return True
            await asyncio.sleep(0.02)
        return False

    async def stop(self) -> None:
        self._running = False
        self.request_stop()
        if self._stream is not None:
            await asyncio.to_thread(self._stream.stop)
            await asyncio.to_thread(self._stream.close)
            self._stream = None
