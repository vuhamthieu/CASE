import asyncio
import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import count
from typing import TYPE_CHECKING, Optional


if TYPE_CHECKING:
    from middleware.message_bus import AsyncMessageBus


logger = logging.getLogger(__name__)

PIPER_SAMPLE_RATE = 22_050
ENABLE_TTS_PIPELINE = True
TTS_PREFETCH_NEXT_CHUNK = True


class CASEVoice:
    """Ordered streaming Piper synthesis and ALSA playback."""

    def __init__(self, bus: "AsyncMessageBus"):
        self.bus = bus
        self.base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        self.piper_bin = os.path.join(self.base_dir, "ai/tts/piper/piper")
        self.model = os.path.join(self.base_dir, "ai/tts/en_US-ryan-medium.onnx")

        self.tts_text_queue: Optional[asyncio.Queue] = None
        self.audio_playback_queue: Optional[asyncio.Queue] = None
        self._synthesis_task: Optional[asyncio.Task] = None
        self._playback_task: Optional[asyncio.Task] = None
        self._full_turn_numbers = count(1_000_000)
        self._synthesis_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="case-tts-synth",
        )
        self._playback_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="case-audio-playback",
        )

        self.bus.subscribe("AI_SPEAK", self.handle_speak_request)
        self.bus.subscribe("AI_SPEAK_STREAM_START", self.handle_stream_start)
        self.bus.subscribe("AI_SPEAK_STREAM_CHUNK", self.handle_stream_chunk)
        self.bus.subscribe("AI_SPEAK_STREAM_END", self.handle_stream_end)

    def _ensure_workers(self) -> None:
        if self.tts_text_queue is None:
            self.tts_text_queue = asyncio.Queue()
        if self.audio_playback_queue is None:
            self.audio_playback_queue = asyncio.Queue()

        if self._synthesis_task is None or self._synthesis_task.done():
            self._synthesis_task = asyncio.create_task(self._synthesis_worker())
        if self._playback_task is None or self._playback_task.done():
            self._playback_task = asyncio.create_task(self._playback_worker())

    async def handle_speak_request(self, text: str) -> None:
        """Queue a non-streamed response as a single ordered TTS turn."""
        if not isinstance(text, str) or not text.strip():
            return

        self._ensure_workers()
        turn_id = next(self._full_turn_numbers)
        now = time.monotonic()
        metrics = {
            "turn_id": turn_id,
            "transcript_final_at": now,
            "llm_stream_start_at": now,
            "first_llm_chunk_at": now,
            "full_response_done_at": now,
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
        await self.tts_text_queue.put(
            {
                "kind": "start",
                "turn_id": payload["turn_id"],
                "metrics": payload["metrics"],
            }
        )

    async def handle_stream_chunk(self, payload: dict) -> None:
        text = str(payload.get("text", "")).strip()
        if not text:
            return

        self._ensure_workers()
        await self.tts_text_queue.put(
            {
                "kind": "chunk",
                "turn_id": payload["turn_id"],
                "sequence": payload["sequence"],
                "text": text,
                "queued_at": payload.get("queued_at", time.monotonic()),
                "metrics": payload["metrics"],
            }
        )

    async def handle_stream_end(self, payload: dict) -> None:
        self._ensure_workers()
        await self.tts_text_queue.put(
            {
                "kind": "end",
                "turn_id": payload["turn_id"],
                "metrics": payload["metrics"],
            }
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
                audio = await loop.run_in_executor(
                    self._synthesis_executor,
                    self._synthesize_raw_audio,
                    text,
                )
                item["synth_done_at"] = time.monotonic()

                if "first_tts_chunk_done_at" not in metrics:
                    metrics["first_tts_chunk_done_at"] = item["synth_done_at"]

                await self.audio_playback_queue.put(
                    {
                        **item,
                        "kind": "audio",
                        "audio": audio,
                    }
                )

                if not TTS_PREFETCH_NEXT_CHUNK:
                    await self.audio_playback_queue.join()

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(
                    "Queue-based TTS failed; using direct Piper/aplay fallback for "
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

        while True:
            item = await self.audio_playback_queue.get()
            try:
                kind = item["kind"]
                if kind == "start":
                    active_turn = item["turn_id"]
                    active_metrics = item["metrics"]
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

                    print(f"\033[96m[CASE]: {item['text']}\033[0m")
                    loop = asyncio.get_running_loop()
                    if kind == "audio":
                        await loop.run_in_executor(
                            self._playback_executor,
                            self._play_raw_audio,
                            item["audio"],
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
                    self._log_latency(metrics)
                    active_turn = None
                    active_metrics = None

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Audio playback worker failed: turn=%r metrics=%r item=%r",
                    active_turn,
                    active_metrics,
                    item,
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

    def _synthesize_raw_audio(self, text: str) -> bytes:
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
        return result.stdout

    def _play_raw_audio(self, audio: bytes) -> None:
        result = subprocess.run(
            [
                "aplay",
                "-r",
                str(PIPER_SAMPLE_RATE),
                "-f",
                "S16_LE",
                "-t",
                "raw",
                "-",
                "-D",
                "default",
            ],
            input=audio,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            error = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"aplay failed with code {result.returncode}: {error}")

    def _run_direct_pipeline(self, text: str) -> None:
        """Legacy Piper-to-aplay streaming path used only as a fallback."""
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        piper_proc = subprocess.Popen(
            [self.piper_bin, "--model", self.model, "--output_raw"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        aplay_proc = subprocess.Popen(
            [
                "aplay",
                "-r",
                str(PIPER_SAMPLE_RATE),
                "-f",
                "S16_LE",
                "-t",
                "raw",
                "-",
                "-D",
                "default",
            ],
            stdin=piper_proc.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if piper_proc.stdout is not None:
            piper_proc.stdout.close()

        piper_proc.communicate(input=text.encode("utf-8"))
        aplay_proc.communicate()
        if piper_proc.returncode != 0 or aplay_proc.returncode != 0:
            raise RuntimeError(
                "Direct TTS fallback failed: "
                f"piper={piper_proc.returncode}, aplay={aplay_proc.returncode}"
            )

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
