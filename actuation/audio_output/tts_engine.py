import asyncio
import hashlib
import logging
import os
import re
import subprocess
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from itertools import count
from typing import TYPE_CHECKING, Optional

import numpy as np

from src.audio.playback_manager import get_playback_manager
from src.voice_pipeline.piper_onnx_backend import PiperOnnxSynthesizer
from src.utils.console_transcript import CASE_CONSOLE_MODE, console
from src.realtime.turn_manager import TurnManager
from src.realtime.realtime_config import (
    CASE_CHECK_PI_THROTTLE,
    CASE_SHOW_LATENCY_SUMMARY,
    CASE_TTS_CACHE_DIR,
    CASE_TTS_CACHE_ENABLED,
    CASE_TTS_DEBUG_DIR,
    CASE_TTS_DUMP_LAST_UTTERANCE,
    CASE_TTS_DROP_OVERFLOW_IN_REALTIME,
    CASE_TTS_FADE_IN_MS,
    CASE_TTS_FADE_OUT_MS,
    CASE_TTS_MIN_UTTERANCE_MS,
    CASE_TTS_POST_SILENCE_MS,
    CASE_TTS_PRE_SILENCE_MS,
    CASE_TTS_REALTIME_MAX_CHUNKS,
    CASE_TTS_PLAYBACK_CONCURRENCY,
    CASE_TTS_SYNTH_CONCURRENCY,
    CASE_TTS_TRIM_KEEP_MS,
    CASE_TTS_TRIM_SILENCE,
    CASE_TTS_TRIM_THRESHOLD_DB,
    CASE_TTS_EMOTION_ENABLED,
    CASE_TTS_DEFAULT_EMOTION,
    CASE_TTS_EMOTION_INTENSITY_DEFAULT,
    CASE_TTS_EMOTION_MAX_GAIN_DB,
    CASE_TTS_EMOTION_USE_GAIN,
    CASE_TTS_EMOTION_USE_LENGTH_SCALE,
    CASE_TTS_EMOTION_ENABLE_PITCH_SHIFT,
    CASE_THINKING_FILLER_DIR,
    CASE_THINKING_FILLER_KEYS,
    CASE_THINKING_FILLER_PREFERRED_KEYS,
    PIPER_CONFIG_PATH,
    PIPER_LENGTH_SCALE,
    PIPER_MODEL_PATH,
    PIPER_NOISE_SCALE,
    PIPER_NOISE_W,
    VOICE_OUTPUT_BACKEND,
)
from src.voice_pipeline.thinking_filler import (
    ThinkingFillerSelector,
    play_thinking_filler_wav,
)
from src.persona.emotion import EmotionState, blend_tts_emotion_profile, parse_leading_emotion_tag


if TYPE_CHECKING:
    from middleware.message_bus import AsyncMessageBus


logger = logging.getLogger(__name__)

PIPER_SAMPLE_RATE = 22_050
ENABLE_TTS_PIPELINE = True
TTS_PREFETCH_NEXT_CHUNK = True


def pad_and_fade_tts_pcm(
    raw_audio: bytes,
    sample_rate: int = PIPER_SAMPLE_RATE,
) -> bytes:
    """Protect utterance edges and enforce a safe minimum playback duration."""
    samples = np.frombuffer(raw_audio, dtype="<i2").astype(np.float32)
    if samples.size == 0:
        return raw_audio
    fade_in = min(samples.size, int(round(sample_rate * CASE_TTS_FADE_IN_MS / 1000)))
    fade_out = min(samples.size, int(round(sample_rate * CASE_TTS_FADE_OUT_MS / 1000)))
    if fade_in:
        samples[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    if fade_out:
        samples[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)
    speech = np.clip(np.rint(samples), -32768, 32767).astype("<i2")
    pre = np.zeros(int(round(sample_rate * CASE_TTS_PRE_SILENCE_MS / 1000)), dtype="<i2")
    post = np.zeros(int(round(sample_rate * CASE_TTS_POST_SILENCE_MS / 1000)), dtype="<i2")
    padded = np.concatenate((pre, speech, post))
    minimum = int(round(sample_rate * CASE_TTS_MIN_UTTERANCE_MS / 1000))
    if len(padded) < minimum:
        padded = np.pad(padded, (0, minimum - len(padded)))
    return padded.tobytes()


def fade_tts_pcm(raw_audio: bytes, sample_rate: int = PIPER_SAMPLE_RATE) -> bytes:
    samples = np.frombuffer(raw_audio, dtype="<i2").astype(np.float32)
    if samples.size == 0:
        return raw_audio
    fade_in = min(samples.size, int(round(sample_rate * CASE_TTS_FADE_IN_MS / 1000)))
    fade_out = min(samples.size, int(round(sample_rate * CASE_TTS_FADE_OUT_MS / 1000)))
    if fade_in:
        samples[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    if fade_out:
        samples[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)
    return np.clip(np.rint(samples), -32768, 32767).astype("<i2").tobytes()


def trim_tts_silence_pcm(
    raw_audio: bytes,
    sample_rate: int,
    *,
    threshold_db: float = -45.0,
    keep_ms: int = 35,
) -> tuple[bytes, dict[str, float]]:
    samples = np.frombuffer(raw_audio, dtype="<i2")
    if samples.size < max(1, int(sample_rate * 0.08)):
        return raw_audio, {"trimmed": 0.0}
    threshold = max(1.0, 32767.0 * (10.0 ** (float(threshold_db) / 20.0)))
    active = np.flatnonzero(np.abs(samples.astype(np.float32)) >= threshold)
    if active.size == 0:
        return raw_audio, {"trimmed": 0.0}
    keep = int(round(sample_rate * max(0, int(keep_ms)) / 1000.0))
    start = max(0, int(active[0]) - keep)
    end = min(samples.size, int(active[-1]) + keep + 1)
    if end <= start or end - start < int(sample_rate * 0.08):
        return raw_audio, {"trimmed": 0.0}
    trimmed = samples[start:end].astype("<i2").tobytes()
    return trimmed, {
        "trimmed": 1.0,
        "lead_ms": start * 1000.0 / sample_rate,
        "tail_ms": (samples.size - end) * 1000.0 / sample_rate,
        "kept_ms": (end - start) * 1000.0 / sample_rate,
    }


def apply_gain_limited_pcm(raw_audio: bytes, gain_db: float) -> tuple[bytes, dict[str, float]]:
    samples = np.frombuffer(raw_audio, dtype="<i2").astype(np.float32)
    if samples.size == 0 or abs(float(gain_db)) < 0.01:
        return raw_audio, {"limited": 0.0, "peak_before": 0.0, "peak_after": 0.0}
    peak_before = float(np.max(np.abs(samples)))
    multiplier = 10.0 ** (float(gain_db) / 20.0)
    boosted = samples * multiplier
    boosted_peak = float(np.max(np.abs(boosted))) if boosted.size else 0.0
    limited = 0.0
    if boosted_peak > 32767.0:
        boosted *= 32767.0 / boosted_peak
        limited = 1.0
    output = np.clip(np.rint(boosted), -32768, 32767).astype("<i2")
    peak_after = float(np.max(np.abs(output.astype(np.float32)))) if output.size else 0.0
    return output.tobytes(), {
        "limited": limited,
        "peak_before": peak_before,
        "peak_after": peak_after,
    }


class CASEVoice:
    """Ordered streaming Piper synthesis and sounddevice playback."""

    def __init__(self, bus: "AsyncMessageBus"):
        self.bus = bus
        self.base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        self.piper_bin = os.path.join(self.base_dir, "ai/tts/piper/piper")
        self.model = os.path.join(self.base_dir, "ai/tts/en_US-ryan-medium.onnx")
        self.voice_backend = VOICE_OUTPUT_BACKEND
        self.piper_onnx: Optional[PiperOnnxSynthesizer] = None
        if self.voice_backend == "piper_onnx":
            model_path = self._resolve_runtime_path(PIPER_MODEL_PATH)
            config_path = self._resolve_runtime_path(PIPER_CONFIG_PATH)
            self.piper_onnx = PiperOnnxSynthesizer(
                model_path,
                config_path,
                length_scale=PIPER_LENGTH_SCALE,
                noise_scale=PIPER_NOISE_SCALE,
                noise_w=PIPER_NOISE_W,
            )
        self.cache_dir = os.path.join(self.base_dir, CASE_TTS_CACHE_DIR)

        self.tts_text_queue: Optional[asyncio.Queue] = None
        self.audio_playback_queue: Optional[asyncio.Queue] = None
        self._synthesis_task: Optional[asyncio.Task] = None
        self._playback_task: Optional[asyncio.Task] = None
        self._full_turn_numbers = count(1_000_000)
        self._synthesis_executor = ThreadPoolExecutor(
            max_workers=max(1, CASE_TTS_SYNTH_CONCURRENCY),
            thread_name_prefix="case-tts-synth",
        )
        self._playback_executor = ThreadPoolExecutor(
            max_workers=max(1, CASE_TTS_PLAYBACK_CONCURRENCY),
            thread_name_prefix="case-audio-playback",
        )
        self._stream_pending_starts: dict[int, dict] = {}
        self._stream_started_turns: set[int] = set()
        self._answer_audio_active = False
        self._thinking_filler_selector = ThinkingFillerSelector(
            CASE_THINKING_FILLER_DIR,
            CASE_THINKING_FILLER_KEYS,
            CASE_THINKING_FILLER_PREFERRED_KEYS,
        )
        self.playback_manager = get_playback_manager()
        self._playback_backend = self.playback_manager.backend

        self.bus.subscribe("AI_SPEAK", self.handle_speak_request)
        self.bus.subscribe("AI_SPEAK_STREAM_START", self.handle_stream_start)
        self.bus.subscribe("AI_SPEAK_STREAM_CHUNK", self.handle_stream_chunk)
        self.bus.subscribe("AI_SPEAK_STREAM_END", self.handle_stream_end)
        self.bus.subscribe("THINKING_FILLER_PLAY", self.handle_thinking_filler)

    def _ensure_workers(self) -> None:
        if self.tts_text_queue is None:
            self.tts_text_queue = asyncio.Queue()
        if self.audio_playback_queue is None:
            self.audio_playback_queue = asyncio.Queue()

        if self._synthesis_task is None or self._synthesis_task.done():
            self._synthesis_task = asyncio.create_task(self._synthesis_worker())
        if self._playback_task is None or self._playback_task.done():
            self._playback_task = asyncio.create_task(self._playback_worker())

    async def prewarm(self) -> None:
        """Start queue workers and warm Piper/model filesystem state once."""
        self._ensure_workers()
        loop = asyncio.get_running_loop()
        if self.piper_onnx is not None:
            try:
                await loop.run_in_executor(
                    self._synthesis_executor,
                    self.piper_onnx.load,
                )
            except Exception as exc:
                logger.warning(
                    "PIPER_ONNX: unavailable; falling back to local_case_tts: %s",
                    exc,
                )
                self.piper_onnx = None
                self.voice_backend = "local_case_tts"
        try:
            await loop.run_in_executor(
                self._synthesis_executor,
                self._synthesize_raw_audio,
                "Ready.",
            )
            logger.info("CASE_TTS: prewarmed")
        except Exception as exc:
            logger.warning("CASE_TTS: prewarm skipped: %s", exc)
        try:
            await loop.run_in_executor(
                self._playback_executor,
                self.playback_manager.start,
            )
        except Exception as exc:
            logger.warning("AUDIO_OUTPUT: device query failed: %s", exc)
        if CASE_CHECK_PI_THROTTLE:
            await loop.run_in_executor(
                self._synthesis_executor,
                self._check_pi_throttle,
            )

    async def handle_speak_request(self, text: str) -> None:
        """Queue a non-streamed response as a single ordered TTS turn."""
        if not isinstance(text, str) or not text.strip():
            return

        tag_state, clean_text = parse_leading_emotion_tag(text)
        text = clean_text
        logger.info("CASE_TTS: speaking text=%r", text.strip())
        self._ensure_workers()
        turn_id = next(self._full_turn_numbers)
        now = time.monotonic()
        emotion_state = tag_state or EmotionState(
            emotion=CASE_TTS_DEFAULT_EMOTION,
            intensity=CASE_TTS_EMOTION_INTENSITY_DEFAULT,
            reason="default_personality",
        )
        metrics = {
            "turn_id": turn_id,
            "transcript_final_at": now,
            "llm_stream_start_at": now,
            "first_llm_chunk_at": now,
            "full_response_done_at": now,
            "emotion": emotion_state.emotion,
            "emotion_intensity": emotion_state.intensity,
            "emotion_reason": emotion_state.reason,
        }
        await self.tts_text_queue.put(
            {"kind": "start", "turn_id": turn_id, "metrics": metrics}
        )
        await self.tts_text_queue.put(
            {
                "kind": "chunk",
                "turn_id": turn_id,
                "sequence": 0,
                "text": text.strip(),
                "queued_at": time.monotonic(),
                "metrics": metrics,
            }
        )
        await self.tts_text_queue.put(
            {"kind": "end", "turn_id": turn_id, "metrics": metrics}
        )

    async def handle_stream_start(self, payload: dict) -> None:
        self._ensure_workers()
        turn_id = int(payload["turn_id"])
        self._stream_pending_starts[turn_id] = {
            "kind": "start",
            "turn_id": turn_id,
            "metrics": payload["metrics"],
        }
        logger.info("CASE_TTS: stream pending turn=%s waiting_for_first_chunk", turn_id)

    async def handle_stream_chunk(self, payload: dict) -> None:
        text = str(payload.get("text", "")).strip()
        if not text:
            return

        metrics = payload.get("metrics", {})
        if (
            metrics.get("realtime_hybrid")
            and CASE_TTS_DROP_OVERFLOW_IN_REALTIME
            and not metrics.get("allow_long_answer")
            and int(payload.get("sequence", 0)) >= CASE_TTS_REALTIME_MAX_CHUNKS
        ):
            if not metrics.get("tts_backend_truncation_logged"):
                logger.info("CASE_TTS: realtime response truncated for latency")
                metrics["tts_backend_truncation_logged"] = True
            return

        logger.info("CASE_TTS: speaking stream chunk=%r", text)
        self._ensure_workers()
        turn_id = int(payload["turn_id"])
        if turn_id not in self._stream_started_turns:
            start_item = self._stream_pending_starts.pop(
                turn_id,
                {"kind": "start", "turn_id": turn_id, "metrics": metrics},
            )
            await self.tts_text_queue.put(start_item)
            self._stream_started_turns.add(turn_id)
            logger.info("CASE_TTS: stream start released turn=%s", turn_id)
        await self.tts_text_queue.put(
            {
                "kind": "chunk",
                "turn_id": turn_id,
                "sequence": payload["sequence"],
                "text": text,
                "queued_at": payload.get("queued_at", time.monotonic()),
                "metrics": payload["metrics"],
                "stream_response": True,
            }
        )

    async def handle_stream_end(self, payload: dict) -> None:
        self._ensure_workers()
        turn_id = int(payload["turn_id"])
        if turn_id not in self._stream_started_turns:
            self._stream_pending_starts.pop(turn_id, None)
            logger.info("CASE_TTS: stream ended without speakable chunks turn=%s", turn_id)
            return
        await self.tts_text_queue.put(
            {
                "kind": "end",
                "turn_id": turn_id,
                "metrics": payload["metrics"],
            }
        )
        self._stream_started_turns.discard(turn_id)

    async def handle_thinking_filler(self, payload: dict) -> None:
        """Play a cached thinking filler without emitting normal TTS state events."""
        if bool(getattr(self, "_answer_audio_active", False)):
            logger.info("THINKING_FILLER_SKIP: reason=audio_active")
            return

        selector = getattr(self, "_thinking_filler_selector", None)
        if selector is None:
            selector = ThinkingFillerSelector(
                CASE_THINKING_FILLER_DIR,
                CASE_THINKING_FILLER_KEYS,
                CASE_THINKING_FILLER_PREFERRED_KEYS,
            )
            self._thinking_filler_selector = selector

        selected = selector.choose()
        if selected is None:
            logger.info("THINKING_FILLER_SKIP: reason=no_assets")
            return

        turn_id = int(payload.get("turn_id", 0) or 0)
        key, path = selected
        logger.info("THINKING_FILLER_PLAY: turn=%s key=%s path=%s", turn_id, key, path)
        started = time.monotonic()
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                self._playback_executor,
                play_thinking_filler_wav,
                path,
            )
            logger.info(
                "THINKING_FILLER_DONE: turn=%s key=%s duration=%.3fs",
                turn_id,
                key,
                time.monotonic() - started,
            )
        except Exception as exc:
            logger.warning(
                "THINKING_FILLER_SKIP: reason=playback_failed error=%s",
                exc,
            )

    async def _synthesis_worker(self) -> None:
        assert self.tts_text_queue is not None
        assert self.audio_playback_queue is not None

        while True:
            item = await self.tts_text_queue.get()
            try:
                kind = item["kind"]
                if kind in {"start", "end"}:
                    await self.audio_playback_queue.put(item)
                    continue

                text = item["text"]
                metrics = item["metrics"]
                item["synth_start_at"] = time.monotonic()
                if "first_tts_chunk_start_at" not in metrics:
                    metrics["first_tts_chunk_start_at"] = item["synth_start_at"]

                logger.info(
                    "Synthesizing TTS chunk: turn=%s sequence=%s queued_for=%.3fs text=%r",
                    item["turn_id"],
                    item["sequence"],
                    item["synth_start_at"] - item["queued_at"],
                    text,
                )

                if not ENABLE_TTS_PIPELINE:
                    raise RuntimeError("Queue-based TTS pipeline is disabled")

                loop = asyncio.get_running_loop()
                raw_audio, audio, sample_rate, cache_hit = await loop.run_in_executor(
                    self._synthesis_executor,
                    self._prepare_tts_audio,
                    text,
                    bool(item.get("stream_response")),
                    item.get("turn_id"),
                    item.get("sequence"),
                    str(metrics.get("emotion", CASE_TTS_DEFAULT_EMOTION)),
                    float(metrics.get("emotion_intensity", CASE_TTS_EMOTION_INTENSITY_DEFAULT)),
                    str(metrics.get("emotion_reason", "default_personality")),
                )
                item["cache_hit"] = cache_hit
                if CASE_TTS_DUMP_LAST_UTTERANCE and raw_audio is not None:
                    self._dump_utterance_debug(raw_audio, audio, sample_rate)
                item["synth_done_at"] = time.monotonic()

                if "first_tts_chunk_done_at" not in metrics:
                    metrics["first_tts_chunk_done_at"] = item["synth_done_at"]

                await self.audio_playback_queue.put(
                    {
                        **item,
                        "kind": "audio",
                        "audio": audio,
                        "sample_rate": sample_rate,
                    }
                )

                if not TTS_PREFETCH_NEXT_CHUNK:
                    await self.audio_playback_queue.join()

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(
                    "Queue-based TTS failed; using direct Piper/sounddevice fallback for "
                    "turn=%s sequence=%s",
                    item.get("turn_id"),
                    item.get("sequence"),
                )
                item["synth_done_at"] = time.monotonic()
                item["pipeline_error"] = str(exc)
                metrics = item.get("metrics", {})
                if "first_tts_chunk_done_at" not in metrics:
                    metrics["first_tts_chunk_done_at"] = item["synth_done_at"]
                await self.audio_playback_queue.put(
                    {
                        **item,
                        "kind": "direct",
                    }
                )
            finally:
                self.tts_text_queue.task_done()

    async def _playback_worker(self) -> None:
        assert self.audio_playback_queue is not None
        active_turn: Optional[int] = None
        active_metrics: Optional[dict] = None
        active_text_parts: list[str] = []

        while True:
            item = await self.audio_playback_queue.get()
            try:
                kind = item["kind"]
                if kind == "start":
                    active_turn = item["turn_id"]
                    active_metrics = item["metrics"]
                    active_text_parts = []
                    await self.bus.publish(
                        "TTS_START",
                        {"turn_id": active_turn, "reason": "CASE speaking"},
                    )
                    await asyncio.sleep(0)
                    continue

                if kind in {"audio", "direct"}:
                    metrics = item["metrics"]
                    item["playback_start_at"] = time.monotonic()
                    if "first_audio_play_start_at" not in metrics:
                        metrics["first_audio_play_start_at"] = item["playback_start_at"]
                    metrics["chunks_played"] = int(metrics.get("chunks_played", 0)) + 1

                    logger.info(
                        "TTS_PLAYBACK_ORDER: turn=%s seq=%s",
                        item.get("turn_id"),
                        item.get("sequence"),
                    )
                    active_text_parts.append(item["text"])
                    if CASE_CONSOLE_MODE != "clean":
                        print(f"\033[96m[CASE]: {item['text']}\033[0m")
                    loop = asyncio.get_running_loop()
                    self._answer_audio_active = True
                    try:
                        if kind == "audio":
                            await loop.run_in_executor(
                                self._playback_executor,
                                self._play_raw_audio,
                                item["audio"],
                                item["sample_rate"],
                            )
                        else:
                            logger.warning(
                                "Playing TTS chunk through direct fallback: turn=%s seq=%s",
                                item["turn_id"],
                                item["sequence"],
                            )
                            await loop.run_in_executor(
                                self._playback_executor,
                                self._run_direct_pipeline,
                                item["text"],
                            )
                    finally:
                        self._answer_audio_active = False

                    item["playback_done_at"] = time.monotonic()
                    self._log_chunk_latency(item)
                    continue

                if kind == "end":
                    metrics = item["metrics"]
                    metrics["full_audio_done_at"] = time.monotonic()
                    await self.bus.publish(
                        "TTS_END",
                        {"turn_id": item["turn_id"], "reason": "CASE finished"},
                    )
                    await asyncio.sleep(0)
                    if CASE_CONSOLE_MODE == "clean" and active_text_parts:
                        console.case(" ".join(active_text_parts))
                    self._log_latency(metrics)
                    self._log_latency_budget(metrics)
                    self._log_compact_latency(metrics)
                    active_turn = None
                    active_metrics = None
                    active_text_parts = []

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                audio = item.get("audio")
                audio_bytes = len(audio) if isinstance(audio, bytes) else 0
                logger.exception(
                    "AUDIO_PLAYBACK: worker failed turn=%r kind=%r seq=%r "
                    "text=%r audio_bytes=%s error=%s",
                    active_turn,
                    item.get("kind"),
                    item.get("sequence"),
                    item.get("text"),
                    audio_bytes,
                    exc,
                )
                if active_turn is not None:
                    await self.bus.publish(
                        "TTS_END",
                        {"turn_id": active_turn, "reason": "playback error"},
                    )
                    await asyncio.sleep(0)
                    active_turn = None
                    active_metrics = None
            finally:
                self.audio_playback_queue.task_done()

    @staticmethod
    def _log_compact_latency(metrics: dict) -> None:
        if not CASE_SHOW_LATENCY_SUMMARY:
            return
        started = metrics.get("transcript_final_at")
        if not started:
            return
        llm_first = metrics.get("first_llm_chunk_at")
        first_audio = metrics.get("first_audio_play_start_at")
        finished = metrics.get("full_audio_done_at")
        summary = (
            "LATENCY_SUMMARY "
            f"llm_first={llm_first - started:.2f}s " if llm_first else "LATENCY_SUMMARY llm_first=n/a "
        )
        summary += (
            f"tts_first={first_audio - started:.2f}s " if first_audio else "tts_first=n/a "
        )
        summary += f"full_turn={finished - started:.2f}s" if finished else "full_turn=n/a"
        logger.info(summary)
        console.system(summary)

    def _synthesize_raw_audio(
        self,
        text: str,
        *,
        length_scale: float | None = None,
    ) -> tuple[bytes, int]:
        if self.piper_onnx is not None:
            try:
                audio, sample_rate = self.piper_onnx.synthesize(
                    text,
                    length_scale=length_scale,
                )
                return audio, sample_rate
            except Exception as exc:
                logger.warning(
                    "PIPER_ONNX: synthesis failed; switching to local_case_tts: %s",
                    exc,
                )
                self.piper_onnx = None
                self.voice_backend = "local_case_tts"

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        result = subprocess.run(
            [self.piper_bin, "--model", self.model, "--output_raw"],
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            check=False,
        )
        if result.returncode != 0:
            error = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Piper failed with code {result.returncode}: {error}")
        if not result.stdout:
            raise RuntimeError("Piper produced no audio")
        return result.stdout, PIPER_SAMPLE_RATE

    def _prepare_tts_audio(
        self,
        text: str,
        response_chunk: bool = False,
        turn_id: int | None = None,
        sequence: int | None = None,
        emotion: str = CASE_TTS_DEFAULT_EMOTION,
        intensity: float = CASE_TTS_EMOTION_INTENSITY_DEFAULT,
        emotion_reason: str = "default_personality",
    ) -> tuple[bytes | None, bytes, int, bool]:
        emotion_state = EmotionState(
            emotion=str(emotion or CASE_TTS_DEFAULT_EMOTION),
            intensity=float(intensity),
            reason=str(emotion_reason or "default_personality"),
        )
        emotion_profile = blend_tts_emotion_profile(
            emotion_state,
            max_gain_db=CASE_TTS_EMOTION_MAX_GAIN_DB,
        )
        length_scale = (
            emotion_profile.length_scale
            if CASE_TTS_EMOTION_ENABLED and CASE_TTS_EMOTION_USE_LENGTH_SCALE
            else None
        )
        gain_db = (
            emotion_profile.gain_db
            if CASE_TTS_EMOTION_ENABLED and CASE_TTS_EMOTION_USE_GAIN
            else 0.0
        )
        cache_path = self._cache_path(
            text,
            response_chunk=response_chunk,
            emotion=emotion_state.emotion,
            intensity=emotion_state.intensity,
            length_scale=length_scale,
            gain_db=gain_db,
        )
        if CASE_TTS_EMOTION_ENABLE_PITCH_SHIFT:
            logger.info("TTS_EMOTION: pitch_shift_requested_but_not_implemented")
        logger.info(
            "TTS_EMOTION_APPLY: turn=%s seq=%s emotion=%s intensity=%.2f "
            "length_scale=%.2f gain_db=%.1f",
            turn_id,
            sequence,
            emotion_state.emotion,
            emotion_state.intensity,
            length_scale if length_scale is not None else PIPER_LENGTH_SCALE,
            gain_db,
        )
        if cache_path and os.path.isfile(cache_path):
            try:
                with wave.open(cache_path, "rb") as source:
                    if (
                        source.getnchannels() != 1
                        or source.getsampwidth() != 2
                    ):
                        raise ValueError("cache format mismatch")
                    sample_rate = source.getframerate()
                    audio = source.readframes(source.getnframes())
                if audio:
                    logger.info("CASE_TTS_CACHE: hit text=%r path=%s", text, cache_path)
                    if response_chunk:
                        logger.info("TTS_RESPONSE_PADDING: mode=normal padding_ms=0")
                    return None, audio, sample_rate, True
            except Exception as exc:
                logger.warning("CASE_TTS_CACHE: ignored invalid entry %s: %s", cache_path, exc)

        raw_audio, sample_rate = self._synthesize_raw_audio(
            text,
            length_scale=length_scale,
        )
        if CASE_TTS_EMOTION_ENABLED and CASE_TTS_EMOTION_USE_GAIN:
            raw_audio, gain_stats = apply_gain_limited_pcm(raw_audio, gain_db)
            if gain_stats.get("limited"):
                logger.info(
                    "TTS_EMOTION_LIMITER: peak_before=%.1f peak_after=%.1f",
                    gain_stats["peak_before"],
                    gain_stats["peak_after"],
                )
        if response_chunk:
            processed = raw_audio
            if CASE_TTS_TRIM_SILENCE:
                processed, trim_stats = trim_tts_silence_pcm(
                    raw_audio,
                    sample_rate,
                    threshold_db=CASE_TTS_TRIM_THRESHOLD_DB,
                    keep_ms=CASE_TTS_TRIM_KEEP_MS,
                )
                if trim_stats.get("trimmed"):
                    logger.info(
                        "TTS_SILENCE_TRIM: turn=%s seq=%s lead_ms=%.1f "
                        "tail_ms=%.1f kept_ms=%.1f",
                        turn_id,
                        sequence,
                        trim_stats["lead_ms"],
                        trim_stats["tail_ms"],
                        trim_stats["kept_ms"],
                    )
            audio = fade_tts_pcm(processed, sample_rate=sample_rate)
            logger.info("TTS_RESPONSE_PADDING: mode=normal padding_ms=0")
        else:
            audio = pad_and_fade_tts_pcm(raw_audio, sample_rate=sample_rate)
        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                temporary = f"{cache_path}.tmp"
                with wave.open(temporary, "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(sample_rate)
                    output.writeframes(audio)
                os.replace(temporary, cache_path)
                logger.info("CASE_TTS_CACHE: stored text=%r path=%s", text, cache_path)
            except Exception as exc:
                logger.warning("CASE_TTS_CACHE: write failed path=%s: %s", cache_path, exc)
        return raw_audio, audio, sample_rate, False

    def _cache_path(
        self,
        text: str,
        *,
        response_chunk: bool = False,
        emotion: str = CASE_TTS_DEFAULT_EMOTION,
        intensity: float = CASE_TTS_EMOTION_INTENSITY_DEFAULT,
        length_scale: float | None = None,
        gain_db: float = 0.0,
    ) -> str | None:
        if not CASE_TTS_CACHE_ENABLED:
            return None
        normalized = " ".join(text.strip().lower().split())
        identity = "|".join(
            (
                normalized,
                self.voice_backend,
                os.path.basename(
                    str(self.piper_onnx.model_path)
                    if self.piper_onnx is not None
                    else self.model
                ),
                str(
                    self.piper_onnx.sample_rate
                    if self.piper_onnx is not None
                    else PIPER_SAMPLE_RATE
                ),
                str(CASE_TTS_PRE_SILENCE_MS),
                str(CASE_TTS_POST_SILENCE_MS),
                str(CASE_TTS_FADE_IN_MS),
                str(CASE_TTS_FADE_OUT_MS),
                str(CASE_TTS_MIN_UTTERANCE_MS),
                "response_chunk" if response_chunk else "safe_utterance",
                str(CASE_TTS_TRIM_SILENCE),
                str(CASE_TTS_TRIM_THRESHOLD_DB),
                str(CASE_TTS_TRIM_KEEP_MS),
                str(CASE_TTS_EMOTION_ENABLED),
                str(emotion),
                f"{float(intensity):.3f}",
                f"{float(length_scale or PIPER_LENGTH_SCALE):.3f}",
                f"{float(gain_db):.3f}",
            )
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        return os.path.join(self.cache_dir, f"{digest}.wav")

    def _resolve_runtime_path(self, path: str) -> str:
        expanded = os.path.expanduser(path)
        if not os.path.isabs(expanded):
            expanded = os.path.join(self.base_dir, expanded)
        return os.path.abspath(expanded)

    @staticmethod
    def _check_pi_throttle() -> None:
        try:
            result = subprocess.run(
                ["vcgencmd", "get_throttled"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=2.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            logger.debug("PI_POWER: vcgencmd unavailable")
            return
        output = result.stdout.strip()
        match = re.search(r"0x([0-9a-fA-F]+)", output)
        if result.returncode == 0 and match:
            flags = int(match.group(1), 16)
            if flags:
                logger.warning("PI_POWER: throttling/undervoltage flags=%s", output)
            else:
                logger.info("PI_POWER: %s", output)
        else:
            logger.debug("PI_POWER: unable to read throttle status: %s", output)

    def _play_raw_audio(self, audio: bytes, sample_rate: int) -> None:
        self._play_raw_audio_sounddevice(audio, sample_rate)

    def _play_raw_audio_sounddevice(self, audio: bytes, sample_rate: int) -> None:
        """Play Piper PCM at the output device's native rate and drain it."""
        try:
            result = self.playback_manager.play(
                audio,
                sample_rate,
            )
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice and scipy are required for the playback fallback"
            ) from exc
        logger.info(
            "PIPER_ONNX_AUDIO_FORMAT: source_rate=%s target_rate=%s "
            "channels=%s->%s duration=%.3fs",
            sample_rate,
            result["sample_rate"],
            result["source_channels"],
            result["channels"],
            result["duration_out"],
        )
        logger.info(
            "AUDIO_PLAYBACK: drained duration=%.3fs path=sounddevice "
            "device=%r sample_rate=%s channels=%s underflow=%s",
            result["duration"],
            result["device_name"],
            result["sample_rate"],
            result["channels"],
            result["underflow"],
        )

    def _dump_utterance_debug(
        self,
        raw_audio: bytes,
        padded_audio: bytes,
        sample_rate: int,
    ) -> None:
        directory = os.path.join(self.base_dir, CASE_TTS_DEBUG_DIR)
        os.makedirs(directory, exist_ok=True)
        for filename, audio in (
            ("last_tts_raw.wav", raw_audio),
            ("last_tts_padded.wav", padded_audio),
        ):
            path = os.path.join(directory, filename)
            with wave.open(path, "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(sample_rate)
                output.writeframes(audio)
        logger.info("CASE_TTS: debug WAVs written directory=%s", directory)

    def _run_direct_pipeline(self, text: str) -> None:
        """Synthesize and play directly if the queue pipeline fails."""
        _, audio, sample_rate, _ = self._prepare_tts_audio(text)
        self._play_raw_audio_sounddevice(audio, sample_rate)

    @staticmethod
    def _log_chunk_latency(item: dict) -> None:
        text = item.get("text", "")
        logger.info(
            "TTS_CHUNK_LATENCY turn=%s seq=%s\n"
            "  queued_at=%s\n"
            "  synth_start_at=%s\n"
            "  synth_done_at=%s\n"
            "  playback_start_at=%s\n"
            "  playback_done_at=%s\n"
            "  chars=%s\n"
            "  words=%s",
            item.get("turn_id"),
            item.get("sequence"),
            CASEVoice._format_timestamp(item.get("queued_at")),
            CASEVoice._format_timestamp(item.get("synth_start_at")),
            CASEVoice._format_timestamp(item.get("synth_done_at")),
            CASEVoice._format_timestamp(item.get("playback_start_at")),
            CASEVoice._format_timestamp(item.get("playback_done_at")),
            len(text),
            len(text.split()),
        )

    @staticmethod
    def _format_timestamp(value) -> str:
        return f"{value:.6f}" if isinstance(value, (int, float)) else "n/a"

    @staticmethod
    def _log_latency(metrics: dict) -> None:
        names = [
            "transcript_final_at",
            "llm_stream_start_at",
            "first_llm_chunk_at",
            "first_tts_chunk_start_at",
            "first_tts_chunk_done_at",
            "first_audio_play_start_at",
            "full_response_done_at",
            "full_audio_done_at",
        ]
        logger.info("LATENCY:")
        for name in names:
            value = metrics.get(name)
            logger.info("  %s = %s", name, f"{value:.6f}" if value else "n/a")

        start = metrics.get("transcript_final_at")
        if not start:
            return

        def elapsed(name: str) -> str:
            value = metrics.get(name)
            return f"{value - start:.3f}" if value else "n/a"

        first_synth_start = metrics.get("first_tts_chunk_start_at")
        first_synth_done = metrics.get("first_tts_chunk_done_at")
        if first_synth_start and first_synth_done:
            first_chunk_synth = f"{first_synth_done - first_synth_start:.3f}"
        else:
            first_chunk_synth = "n/a"

        first_audio = metrics.get("first_audio_play_start_at")
        full_audio = metrics.get("full_audio_done_at")
        if first_audio and full_audio:
            total_audio = f"{full_audio - first_audio:.3f}"
        else:
            total_audio = "n/a"

        logger.info(
            "LATENCY llm_first_token=%ss first_audio=%ss first_chunk_synth=%ss "
            "full_llm=%ss total_audio=%ss full_tts_playback=%ss",
            elapsed("first_llm_chunk_at"),
            elapsed("first_audio_play_start_at"),
            first_chunk_synth,
            elapsed("full_response_done_at"),
            total_audio,
            elapsed("full_audio_done_at"),
        )
        logger.info(
            "TURN_STREAMING_METRICS: llm_first_delta=%ss first_chunk_ready=%ss "
            "first_tts_synth_start=%ss first_tts_synth_done=%ss "
            "first_audio_play_start=%ss chunks_emitted=%s chunks_played=%s "
            "total_response_chars=%s",
            elapsed("llm_first_delta_at"),
            elapsed("first_chunk_ready_at"),
            elapsed("first_tts_chunk_start_at"),
            elapsed("first_tts_chunk_done_at"),
            elapsed("first_audio_play_start_at"),
            metrics.get("chunks_emitted", metrics.get("tts_chunks_accepted", "n/a")),
            metrics.get("chunks_played", "n/a"),
            metrics.get("total_response_chars", metrics.get("tts_spoken_chars", "n/a")),
        )
        TurnManager.log_latency(metrics)

    @staticmethod
    def _log_latency_budget(metrics: dict) -> None:
        start = metrics.get("transcript_final_at")

        def since_start(name: str) -> str:
            value = metrics.get(name)
            if not start or not value:
                return "n/a"
            return f"{value - start:.3f}s"

        synth_start = metrics.get("first_tts_chunk_start_at")
        synth_done = metrics.get("first_tts_chunk_done_at")
        playback_start = metrics.get("first_audio_play_start_at")
        playback_done = metrics.get("full_audio_done_at")
        synth = (
            f"{synth_done - synth_start:.3f}s"
            if synth_start and synth_done
            else "n/a"
        )
        playback = (
            f"{playback_done - playback_start:.3f}s"
            if playback_start and playback_done
            else "n/a"
        )
        logger.info(
            "LATENCY_BUDGET stt_finalize=n/a llm_first=%s text_ready=%s "
            "tts_synth=%s playback=%s total_to_case_audio=%s",
            since_start("first_llm_chunk_at"),
            since_start("text_ready_at"),
            synth,
            playback,
            since_start("first_audio_play_start_at"),
        )
