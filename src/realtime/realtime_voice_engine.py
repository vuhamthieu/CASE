"""Gemini Live audio-to-audio session engine for CASE."""

from __future__ import annotations

import asyncio
import difflib
import logging
import re
import threading
import time
import wave
from collections import deque
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from src.audio.voice_fx import create_voice_fx_from_config
from .realtime_audio_io import RealtimeAudioInput, RealtimeAudioOutput
from .realtime_config import (
    CASE_VOICE_FX_DEBUG_DIR,
    CASE_VOICE_FX_DUMP_WAV,
    CASE_VOICE_PRESET,
    CASE_HONESTY_PERCENT,
    CASE_HUMOR_PERCENT,
    CASE_SARCASM_LEVEL,
    CASE_STYLE_MAX_SENTENCES,
    CASE_STYLE_SHORT_REPLIES,
    GEMINI_API_KEY,
    GEMINI_LIVE_MODEL,
    GEMINI_LIVE_VOICE_NAME,
    REALTIME_AUDIO_CHUNK_MS,
    REALTIME_AUDIO_CHUNK_LOG_INTERVAL_SEC,
    REALTIME_BARGE_IN_COOLDOWN_SEC,
    REALTIME_BARGE_IN_FRAMES,
    REALTIME_BARGE_IN_IGNORE_AFTER_PLAYBACK_START_SEC,
    REALTIME_BARGE_IN_MIN_SPEECH_MS,
    REALTIME_BARGE_IN_RMS,
    REALTIME_DEBUG_AUDIO_DIR,
    REALTIME_DUMP_MODEL_AUDIO_WAV,
    REALTIME_ENABLE_BARGE_IN,
    REALTIME_ENABLE_TOOLS,
    REALTIME_ENABLE_TRANSCRIPTS,
    REALTIME_ECHO_TEXT_SIMILARITY_THRESHOLD,
    REALTIME_ECHO_TRANSCRIPT_GUARD_SEC,
    REALTIME_HALF_DUPLEX,
    REALTIME_IGNORE_ECHO_TRANSCRIPTS,
    REALTIME_IDLE_END_SEC,
    REALTIME_INPUT_SAMPLE_RATE,
    REALTIME_LOG_AUDIO_LEVELS,
    REALTIME_MIN_MIC_PAUSE_FOR_STREAM_END_SEC,
    REALTIME_MUTE_MIC_DURING_PLAYBACK,
    REALTIME_RESUME_MIC_AFTER_PLAYBACK_DELAY_SEC,
    REALTIME_SEND_AUDIO_STREAM_END_ON_MIC_PAUSE,
    REALTIME_SESSION_TIMEOUT_SEC,
    REALTIME_STATE_LOG_ON_CHANGE_ONLY,
    REALTIME_WAKE_ACK_BLOCKS_MIC,
    REALTIME_WAKE_ACK_CONNECT_IN_PARALLEL,
    REALTIME_WAKE_ACK_ENABLED,
    REALTIME_WAKE_ACK_MODE,
    REALTIME_WAKE_ACK_POST_DELAY_SEC,
    REALTIME_WAKE_ACK_TEXT,
)
from .realtime_persona import build_case_system_instruction
from .realtime_tools import RealtimeToolRouter, update_core_memory
from src.utils.console_transcript import console
from src.voice_pipeline.wake_ack import (
    choose_wake_ack,
    play_wake_ack_asset,
    play_wake_ack_beep,
)


logger = logging.getLogger(__name__)

STATE_REALTIME_CONNECTING = "REALTIME_CONNECTING"
STATE_REALTIME_WAKE_ACK = "WAKE_ACK"
STATE_REALTIME_LISTENING = "REALTIME_LISTENING"
STATE_REALTIME_RESPONDING_RECEIVING_AUDIO = (
    "REALTIME_RESPONDING_RECEIVING_AUDIO"
)
STATE_REALTIME_RESPONDING_PLAYING_AUDIO = "REALTIME_RESPONDING_PLAYING_AUDIO"
STATE_REALTIME_ECHO_GUARD_WAIT = "REALTIME_ECHO_GUARD_WAIT"
STATE_REALTIME_TOOL_EXECUTING = "REALTIME_TOOL_EXECUTING"
STATE_REALTIME_ERROR = "REALTIME_ERROR"


class RealtimeVoiceEngine:
    def __init__(
        self,
        *,
        message_bus: Any,
        shared_audio_queue=None,
        tool_router: Optional[RealtimeToolRouter] = None,
        state_callback: Optional[Callable[[str], None]] = None,
        input_device=None,
        output_device=None,
        enable_barge_in: Optional[bool] = None,
        dump_model_audio_wav: Optional[bool] = None,
        half_duplex: Optional[bool] = None,
        voice_name: Optional[str] = None,
        persona_name: Optional[str] = None,
    ) -> None:
        self.message_bus = message_bus
        self.tool_router = tool_router or RealtimeToolRouter()
        self.state_callback = state_callback
        self.enable_barge_in = (
            REALTIME_ENABLE_BARGE_IN
            if enable_barge_in is None
            else enable_barge_in
        )
        self.dump_model_audio_wav = (
            REALTIME_DUMP_MODEL_AUDIO_WAV
            if dump_model_audio_wav is None
            else dump_model_audio_wav
        )
        self.half_duplex = (
            REALTIME_HALF_DUPLEX if half_duplex is None else half_duplex
        )
        self.mute_mic_during_playback = (
            self.half_duplex and REALTIME_MUTE_MIC_DURING_PLAYBACK
        )
        self.voice_name = (voice_name or GEMINI_LIVE_VOICE_NAME).strip()
        self.persona_name = (persona_name or CASE_VOICE_PRESET).strip()
        self.voice_fx = create_voice_fx_from_config(24_000)
        self.audio_input = RealtimeAudioInput(shared_audio_queue, input_device)
        self.metrics: dict[str, float] = {}
        self._stop_event: Optional[asyncio.Event] = None
        self._last_activity_at = 0.0
        self._barge_in_frames = 0
        self._barge_candidate_at = 0.0
        self._last_barge_in_at = 0.0
        self._last_user_transcript_at = 0.0
        self._last_user_transcript = ""
        self._last_guard_log_at = 0.0
        self._model_audio = bytearray()
        self._model_audio_fx = bytearray()
        self._chunk_log_started_at = 0.0
        self._chunk_log_count = 0
        self._chunk_log_bytes = 0
        self._current_state: Optional[str] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._mic_uplink_muted = threading.Event()
        self.mic_uplink_enabled = False
        self._mic_uplink_reason = "initializing"
        self._mic_pause_started_at = 0.0
        self._audio_stream_end_sent = False
        self._playback_generation = 0
        self._resume_task: Optional[asyncio.Task] = None
        self._echo_guard_until = 0.0
        self._recent_model_transcripts: deque[tuple[float, str]] = deque()
        self._console_user_fragments: list[str] = []
        self._console_model_fragments: list[str] = []
        self._model_response_done = True
        self._playback_drained_before_response_done = False
        self._session_connected = False
        self._shutting_down = False
        self._wake_ack_started = False
        self._wake_ack_tts_count = 0
        self._wake_ack_tts_started: Optional[asyncio.Event] = None
        self._wake_ack_tts_done: Optional[asyncio.Event] = None
        self._wake_ack_text = REALTIME_WAKE_ACK_TEXT
        self._ignored_disabled_transcript_logged = False
        self._session_error: Optional[Exception] = None
        self.audio_output = RealtimeAudioOutput(
            output_device,
            on_playback_start=self._on_playback_start,
            on_playback_drained=self._on_playback_drained,
        )
        if self.message_bus is not None:
            self.message_bus.subscribe("TTS_START", self._on_wake_ack_tts_start)
            self.message_bus.subscribe("TTS_END", self._on_wake_ack_tts_end)

    async def run_session(
        self,
        wake_detected_at: Optional[float] = None,
        max_duration_sec: Optional[float] = None,
    ) -> bool:
        if not GEMINI_API_KEY:
            logger.warning("REALTIME: GEMINI_API_KEY missing")
            return False

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            logger.warning("REALTIME: google-genai is not installed")
            return False

        self.metrics = {
            "wake_detected_at": wake_detected_at or time.monotonic(),
            "session_connect_start_at": time.monotonic(),
        }
        self._last_activity_at = time.monotonic()
        self._session_error = None
        self._stop_event = asyncio.Event()
        self._barge_in_frames = 0
        self._barge_candidate_at = 0.0
        self._last_barge_in_at = 0.0
        self._last_user_transcript_at = 0.0
        self._last_user_transcript = ""
        self._model_audio.clear()
        self._model_audio_fx.clear()
        self._chunk_log_started_at = time.monotonic()
        self._chunk_log_count = 0
        self._chunk_log_bytes = 0
        self._loop = asyncio.get_running_loop()
        self._mic_uplink_muted.set()
        self.mic_uplink_enabled = False
        self._mic_uplink_reason = "wake_ack"
        self._mic_pause_started_at = 0.0
        self._audio_stream_end_sent = False
        self._playback_generation = 0
        self._echo_guard_until = 0.0
        self._recent_model_transcripts.clear()
        self._console_user_fragments.clear()
        self._console_model_fragments.clear()
        self._model_response_done = True
        self._playback_drained_before_response_done = False
        self._session_connected = False
        self._shutting_down = False
        self._wake_ack_started = False
        self._wake_ack_tts_count = 0
        self._wake_ack_tts_started = asyncio.Event()
        self._wake_ack_tts_done = asyncio.Event()
        self._wake_ack_text = choose_wake_ack()
        self._ignored_disabled_transcript_logged = False
        self._set_state(STATE_REALTIME_WAKE_ACK)
        console.system("wake detected")
        logger.info("REALTIME_MIC: uplink disabled reason=wake_ack")

        persona_instruction = build_case_system_instruction(
            self.persona_name,
            short_replies=CASE_STYLE_SHORT_REPLIES,
            max_sentences=CASE_STYLE_MAX_SENTENCES,
            humor_percent=CASE_HUMOR_PERCENT,
            honesty_percent=CASE_HONESTY_PERCENT,
            sarcasm_level=CASE_SARCASM_LEVEL,
        )

        config: dict[str, Any] = {
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {"voice_name": self.voice_name}
                }
            },
            "system_instruction": (
                persona_instruction + " "
                "Never claim to see the user without calling case_vision_see_me or "
                "using a recent case_get_vision_state result. Never claim a picture "
                "was saved without calling case_take_picture successfully. Motor "
                "movement is disabled; case_motion_request only records intent."
            ),
            "realtime_input_config": {
                "automatic_activity_detection": {
                    "disabled": False,
                    "prefix_padding_ms": 20,
                    "silence_duration_ms": 500,
                }
            },
        }
        if not self.enable_barge_in:
            config["realtime_input_config"]["activity_handling"] = "NO_INTERRUPTION"
        if REALTIME_ENABLE_TRANSCRIPTS:
            config["input_audio_transcription"] = {}
            config["output_audio_transcription"] = {}
        # Ensure update_core_memory tool is registered and explicitly passed into the Gemini model configuration
        config["tools"] = [
            {"function_declarations": self.tool_router.declarations}
        ]
        logger.info("LLM_MODE: tool_enabled with update_core_memory explicitly passed")

        client = genai.Client(api_key=GEMINI_API_KEY)
        ack_task: Optional[asyncio.Task] = None
        try:
            self.audio_input.drain()
            await self.audio_input.start()
            if REALTIME_WAKE_ACK_CONNECT_IN_PARALLEL:
                logger.info("REALTIME: connecting in parallel with wake ack")
                ack_task = asyncio.create_task(
                    self._play_wake_ack(),
                    name="realtime-wake-ack",
                )
            else:
                await self._play_wake_ack()
                self._set_state(STATE_REALTIME_CONNECTING)
            async with client.aio.live.connect(
                model=GEMINI_LIVE_MODEL,
                config=config,
            ) as session:
                self._session_connected = True
                self.metrics["session_connected_at"] = time.monotonic()
                self._last_activity_at = time.monotonic()
                logger.info(
                    "REALTIME: connected model=%s",
                    GEMINI_LIVE_MODEL,
                )
                logger.info(
                    "REALTIME: voice=%s persona=%s half_duplex=%s",
                    self.voice_name,
                    self.persona_name,
                    self.half_duplex,
                )
                if ack_task is not None:
                    await ack_task
                    ack_task = None
                console.system(
                    f"realtime connected voice={self.voice_name} "
                    f"mode={'half-duplex' if self.half_duplex else 'full-duplex'}"
                )
                await self.audio_output.start()
                self.audio_input.drain()
                self._set_mic_uplink(True, "listening")
                self._set_state(STATE_REALTIME_LISTENING)
                logger.info(
                    "REALTIME: mic streaming started chunk_ms=%s sample_rate=%s",
                    REALTIME_AUDIO_CHUNK_MS,
                    REALTIME_INPUT_SAMPLE_RATE,
                )
                logger.info(
                    "REALTIME: barge-in %s",
                    "enabled" if self.enable_barge_in else "disabled",
                )

                tasks = [
                    asyncio.create_task(
                        self._guard_task(self._send_audio(session)),
                        name="realtime-audio-send",
                    ),
                    asyncio.create_task(
                        self._guard_task(self._receive(session, types)),
                        name="realtime-receive",
                    ),
                    asyncio.create_task(
                        self._guard_task(self._watchdog(max_duration_sec)),
                        name="realtime-watchdog",
                    ),
                ]
                await self._stop_event.wait()
                try:
                    await session.send_realtime_input(audio_stream_end=True)
                except Exception:
                    pass
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

            if self._session_error is not None:
                raise self._session_error
            return True

        except asyncio.CancelledError:
            logger.info("REALTIME: session cancelled during shutdown")
            console.system("realtime session cancelled")
            raise
        except Exception as exc:
            self._set_state(STATE_REALTIME_ERROR)
            logger.warning("REALTIME: session failed: %s", exc)
            return False
        finally:
            self._shutting_down = True
            self.metrics["session_done_at"] = time.monotonic()
            if ack_task is not None:
                ack_task.cancel()
                await asyncio.gather(ack_task, return_exceptions=True)
            self._set_mic_uplink(False, "shutdown")
            if self._resume_task is not None:
                self._resume_task.cancel()
                await asyncio.gather(self._resume_task, return_exceptions=True)
                self._resume_task = None
            self._flush_console_user()
            self._flush_console_model()
            self._dump_model_audio()
            await self.audio_input.stop()
            await self.audio_output.stop()
            self.audio_input.drain()
            self._log_latency()

    async def _play_wake_ack(self) -> None:
        if not REALTIME_WAKE_ACK_ENABLED or REALTIME_WAKE_ACK_MODE == "none":
            logger.info("REALTIME_WAKE_ACK: disabled")
            if not self._session_connected:
                self._set_state(STATE_REALTIME_CONNECTING)
            return

        mode = REALTIME_WAKE_ACK_MODE
        logger.info(
            'REALTIME_WAKE_ACK: mode=%s text="%s"',
            mode,
            self._wake_ack_text,
        )
        if REALTIME_WAKE_ACK_BLOCKS_MIC:
            self._set_mic_uplink(False, "wake_ack")
            logger.info("REALTIME: mic uplink held until wake ack complete")

        played = False
        if mode in {"recorded_wav", "cached_wav"}:
            played = await asyncio.to_thread(
                play_wake_ack_asset,
                self._wake_ack_text,
                mode=mode,
            )
            if not played:
                logger.warning(
                    "REALTIME_WAKE_ACK: WAV assets unavailable; falling back to local_tts"
                )
                mode = "local_tts"
        if mode == "local_tts":
            played = await self._play_local_tts_wake_ack()
            if not played:
                logger.warning(
                    "REALTIME_WAKE_ACK: local TTS failed; falling back to beep"
                )
                mode = "beep"
        if mode == "beep":
            played = await self._play_beep_wake_ack()
        if mode not in {"recorded_wav", "cached_wav", "local_tts", "beep"}:
            logger.warning(
                "REALTIME_WAKE_ACK: unsupported mode=%s; using beep",
                mode,
            )
            played = await self._play_beep_wake_ack()

        if played:
            console.case(self._wake_ack_text)
        if REALTIME_WAKE_ACK_POST_DELAY_SEC > 0:
            await asyncio.sleep(REALTIME_WAKE_ACK_POST_DELAY_SEC)
        logger.info("REALTIME_WAKE_ACK: complete")
        if not self._session_connected:
            self._set_state(STATE_REALTIME_CONNECTING)

    async def _play_cached_wake_ack(self) -> bool:
        return await asyncio.to_thread(
            play_wake_ack_asset,
            self._wake_ack_text,
            mode="cached_wav",
        )

    async def _play_local_tts_wake_ack(self) -> bool:
        if self.message_bus is None:
            return False
        self._wake_ack_started = True
        self._wake_ack_tts_count = 0
        if self._wake_ack_tts_started is not None:
            self._wake_ack_tts_started.clear()
        if self._wake_ack_tts_done is not None:
            self._wake_ack_tts_done.clear()
        try:
            await self.message_bus.publish("AI_SPEAK", self._wake_ack_text)
            await asyncio.wait_for(self._wake_ack_tts_started.wait(), timeout=5.0)
            await asyncio.wait_for(self._wake_ack_tts_done.wait(), timeout=20.0)
            return True
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning("REALTIME_WAKE_ACK: local TTS error: %s", exc)
            return False
        finally:
            self._wake_ack_started = False

    async def _play_beep_wake_ack(self) -> bool:
        logger.info("REALTIME_WAKE_ACK: playing fallback beep")
        return await asyncio.to_thread(play_wake_ack_beep)

    @staticmethod
    def _wav_to_mono_float32(audio: np.ndarray) -> np.ndarray:
        if audio.ndim > 1:
            audio = audio.astype(np.float32).mean(axis=1)
        if np.issubdtype(audio.dtype, np.integer):
            maximum = float(max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max))
            return audio.astype(np.float32) / maximum
        return np.clip(audio.astype(np.float32), -1.0, 1.0)

    @staticmethod
    async def _play_sounddevice_audio(audio: np.ndarray, sample_rate: int) -> None:
        import sounddevice as sd

        await asyncio.to_thread(sd.play, audio, sample_rate, blocking=True)

    async def _on_wake_ack_tts_start(self, payload: Any) -> None:
        if not self._wake_ack_started:
            return
        self._wake_ack_tts_count += 1
        if self._wake_ack_tts_started is not None:
            self._wake_ack_tts_started.set()

    async def _on_wake_ack_tts_end(self, payload: Any) -> None:
        if not self._wake_ack_started:
            return
        self._wake_ack_tts_count = max(0, self._wake_ack_tts_count - 1)
        if self._wake_ack_tts_count == 0 and self._wake_ack_tts_done is not None:
            self._wake_ack_tts_done.set()

    def _set_mic_uplink(self, enabled: bool, reason: str) -> None:
        if self.mic_uplink_enabled == enabled and self._mic_uplink_reason == reason:
            return
        self.mic_uplink_enabled = enabled
        self._mic_uplink_reason = reason
        if enabled:
            self._mic_uplink_muted.clear()
            self._mic_pause_started_at = 0.0
            self._audio_stream_end_sent = False
            self._last_activity_at = time.monotonic()
            self._ignored_disabled_transcript_logged = False
            logger.info("REALTIME_MIC: uplink enabled reason=%s", reason)
        else:
            self._mic_uplink_muted.set()
            if self._mic_pause_started_at <= 0:
                self._mic_pause_started_at = time.monotonic()
            logger.info("REALTIME_MIC: uplink disabled reason=%s", reason)

    async def _guard_task(self, coroutine) -> None:
        try:
            await coroutine
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._session_error = exc
            logger.exception("REALTIME: background task failed")
            if self._stop_event is not None:
                self._stop_event.set()

    async def _send_audio(self, session) -> None:
        from google.genai import types

        while self._stop_event is not None and not self._stop_event.is_set():
            chunk = await self.audio_input.read_chunk()
            if not chunk:
                continue

            if not self.mic_uplink_enabled:
                await self._maybe_send_audio_stream_end(session)
                continue

            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
            if REALTIME_LOG_AUDIO_LEVELS:
                logger.info("REALTIME: input rms=%.1f", rms)

            if rms >= REALTIME_BARGE_IN_RMS and not self.audio_output.is_playing:
                self._last_activity_at = time.monotonic()

            if self.audio_output.is_playing and self.enable_barge_in:
                self._handle_local_barge_in(rms)

            await session.send_realtime_input(
                audio=types.Blob(
                    data=chunk,
                    mime_type=f"audio/pcm;rate={REALTIME_INPUT_SAMPLE_RATE}",
                )
            )
            if "first_audio_sent_at" not in self.metrics:
                self.metrics["first_audio_sent_at"] = time.monotonic()

    async def _receive(self, session, types) -> None:
        while self._stop_event is not None and not self._stop_event.is_set():
            turn = session.receive()
            async for response in turn:
                await self._handle_response(session, response, types)

    async def _handle_response(self, session, response, types) -> None:
        now = time.monotonic()
        server_content = getattr(response, "server_content", None)
        turn_complete = False
        if server_content is not None:
            input_transcription = getattr(server_content, "input_transcription", None)
            if input_transcription and getattr(input_transcription, "text", None):
                text = input_transcription.text
                if not self.mic_uplink_enabled:
                    if not self._ignored_disabled_transcript_logged:
                        logger.info(
                            "REALTIME_ECHO_GUARD: ignored transcript while mic uplink disabled"
                        )
                        self._ignored_disabled_transcript_logged = True
                elif self._is_echo_transcript(text, now):
                    logger.info(
                        'REALTIME_ECHO_GUARD: ignored echo transcript fragment="%s"',
                        text.strip(),
                    )
                else:
                    self._last_activity_at = now
                    self._last_user_transcript_at = now
                    self._last_user_transcript = text.strip()
                    self.metrics.setdefault("first_user_transcript_at", now)
                    logger.info('REALTIME: transcript user="%s"', text)
                    self._console_user_fragments.append(text)

            output_transcription = getattr(server_content, "output_transcription", None)
            if output_transcription and getattr(output_transcription, "text", None):
                model_text = output_transcription.text
                self._recent_model_transcripts.append((now, model_text))
                self._trim_model_transcripts(now)
                logger.info('REALTIME: transcript model="%s"', model_text)
                self._flush_console_user()
                self._console_model_fragments.append(model_text)

            if getattr(server_content, "interrupted", False):
                if self.enable_barge_in:
                    if self.audio_output.request_stop():
                        self._last_barge_in_at = now
                        logger.info(
                            "REALTIME: server barge-in confirmed, stopping playback"
                        )
                else:
                    logger.debug(
                        "REALTIME: server interruption ignored; barge-in disabled"
                    )

            if getattr(server_content, "turn_complete", False):
                self._last_activity_at = now
                turn_complete = True

        audio_chunks: list[bytes] = []
        direct_data = getattr(response, "data", None)
        if isinstance(direct_data, bytes):
            audio_chunks.append(direct_data)
        elif server_content is not None:
            model_turn = getattr(server_content, "model_turn", None)
            for part in getattr(model_turn, "parts", None) or []:
                inline_data = getattr(part, "inline_data", None)
                data = getattr(inline_data, "data", None)
                if isinstance(data, bytes):
                    audio_chunks.append(data)

        for audio in audio_chunks:
            self.metrics.setdefault("first_model_audio_at", now)
            self._last_activity_at = now
            if self._model_response_done:
                self._model_response_done = False
                self._playback_drained_before_response_done = False
                if self.half_duplex:
                    self._set_mic_uplink(False, "model_response")
                self._set_state(STATE_REALTIME_RESPONDING_RECEIVING_AUDIO)
            self._flush_console_user()
            processed_audio = (
                self.voice_fx.process_int16_mono(audio)
                if self.voice_fx is not None
                else audio
            )
            self.audio_output.enqueue(processed_audio)
            self._model_audio.extend(audio)
            self._model_audio_fx.extend(processed_audio)
            self._log_audio_chunk(len(audio), now)

        if turn_complete:
            self._model_response_done = True
            self._flush_console_user()
            self._flush_console_model()
            self._dump_model_audio()
            logger.info("REALTIME: model response bytes complete")
            if self.half_duplex:
                logger.info(
                    "REALTIME_ECHO_GUARD: waiting for local playback drain"
                )
                if not self.audio_output.is_playing:
                    self._playback_drained_on_loop(self._playback_generation)
            else:
                self._set_mic_uplink(True, "listening")
                self._set_state(STATE_REALTIME_LISTENING)

        tool_call = getattr(response, "tool_call", None)
        if tool_call is not None:
            await self._handle_tool_calls(session, tool_call, types)

    async def _handle_tool_calls(self, session, tool_call, types) -> None:
        function_responses = []
        for function_call in getattr(tool_call, "function_calls", None) or []:
            self.metrics.setdefault("tool_call_at", time.monotonic())
            self._set_mic_uplink(False, "tool_executing")
            self._set_state(STATE_REALTIME_TOOL_EXECUTING)
            name = str(function_call.name)
            arguments = dict(getattr(function_call, "args", None) or {})
            result = await self.tool_router.execute(name, arguments)
            function_responses.append(
                types.FunctionResponse(
                    id=function_call.id,
                    name=name,
                    response={"result": result},
                )
            )

        if function_responses:
            await session.send_tool_response(
                function_responses=function_responses
            )
            self.metrics["tool_result_sent_at"] = time.monotonic()
            self._last_activity_at = time.monotonic()
            logger.info(
                "REALTIME: tool_result sent count=%s",
                len(function_responses),
            )

    async def _watchdog(self, max_duration_sec: Optional[float]) -> None:
        started = time.monotonic()
        session_limit = max_duration_sec or REALTIME_SESSION_TIMEOUT_SEC
        while self._stop_event is not None and not self._stop_event.is_set():
            now = time.monotonic()
            if now - started >= session_limit:
                logger.info("REALTIME: session timeout")
                self._stop_event.set()
                return
            if (
                self.mic_uplink_enabled
                and self._current_state == STATE_REALTIME_LISTENING
                and now - self._last_activity_at >= REALTIME_IDLE_END_SEC
            ):
                logger.info("REALTIME: idle timeout")
                self._stop_event.set()
                return
            await asyncio.sleep(0.25)

    def _on_playback_start(self) -> None:
        if self._shutting_down:
            return
        now = time.monotonic()
        self.metrics.setdefault("first_audio_play_start_at", now)
        if not self.mute_mic_during_playback:
            return
        self._playback_generation += 1
        generation = self._playback_generation
        if self._loop is not None:
            self._loop.call_soon_threadsafe(
                self._playback_started_on_loop,
                generation,
            )

    def _on_playback_drained(self) -> None:
        if self._shutting_down or not self.mute_mic_during_playback:
            return
        generation = self._playback_generation
        if self._loop is not None:
            self._loop.call_soon_threadsafe(
                self._playback_drained_on_loop,
                generation,
            )

    def _playback_started_on_loop(self, generation: int) -> None:
        if generation != self._playback_generation:
            return
        if self._resume_task is not None:
            self._resume_task.cancel()
            self._resume_task = None
        logger.info(
            "REALTIME_ECHO_GUARD: model playback started, muting mic uplink"
        )
        self._set_mic_uplink(False, "model_playback")
        self._set_state(STATE_REALTIME_RESPONDING_PLAYING_AUDIO)
        console.system("echo guard: mic muted during CASE speech")

    def _playback_drained_on_loop(self, generation: int) -> None:
        if generation != self._playback_generation:
            return
        if not self._model_response_done:
            self._playback_drained_before_response_done = True
            logger.debug(
                "REALTIME_ECHO_GUARD: playback drained between incoming chunks"
            )
            return
        self._set_mic_uplink(False, "echo_guard")
        self._set_state(STATE_REALTIME_ECHO_GUARD_WAIT)
        logger.info(
            "REALTIME_ECHO_GUARD: playback drained, waiting %.1fs before mic resume",
            REALTIME_RESUME_MIC_AFTER_PLAYBACK_DELAY_SEC,
        )
        if self._resume_task is not None:
            self._resume_task.cancel()
        self._resume_task = asyncio.create_task(
            self._resume_mic_after_guard(generation),
            name="realtime-mic-resume-guard",
        )

    async def _resume_mic_after_guard(self, generation: int) -> None:
        try:
            await asyncio.sleep(REALTIME_RESUME_MIC_AFTER_PLAYBACK_DELAY_SEC)
            if generation != self._playback_generation or self.audio_output.is_playing:
                return
            self.audio_input.drain()
            self._echo_guard_until = time.monotonic() + REALTIME_ECHO_TRANSCRIPT_GUARD_SEC
            self._set_mic_uplink(True, "listening")
            logger.info("REALTIME_ECHO_GUARD: mic uplink resumed")
            logger.info("REALTIME: idle timer reset after mic resumed")
            self._set_state(STATE_REALTIME_LISTENING)
            console.system("echo guard: mic resumed")
        except asyncio.CancelledError:
            raise

    async def _maybe_send_audio_stream_end(self, session) -> None:
        if (
            not REALTIME_SEND_AUDIO_STREAM_END_ON_MIC_PAUSE
            or self._audio_stream_end_sent
            or self._mic_pause_started_at <= 0
            or time.monotonic() - self._mic_pause_started_at
            < REALTIME_MIN_MIC_PAUSE_FOR_STREAM_END_SEC
        ):
            return
        try:
            await session.send_realtime_input(audio_stream_end=True)
            self._audio_stream_end_sent = True
            logger.info(
                "REALTIME_ECHO_GUARD: audio stream end sent for muted mic uplink"
            )
        except (AttributeError, TypeError):
            self._audio_stream_end_sent = True
            logger.info(
                "REALTIME_ECHO_GUARD: audio stream end not supported by current SDK wrapper"
            )
        except Exception as exc:
            self._audio_stream_end_sent = True
            logger.warning("REALTIME_ECHO_GUARD: audio stream end failed: %s", exc)

    def _is_echo_transcript(self, text: str, now: float) -> bool:
        if not REALTIME_IGNORE_ECHO_TRANSCRIPTS:
            return False
        guarded = self._mic_uplink_muted.is_set() or now <= self._echo_guard_until
        if not guarded:
            return False
        self._trim_model_transcripts(now)
        fragment = self._normalize_transcript(text)
        recent = self._normalize_transcript(
            " ".join(value for _, value in self._recent_model_transcripts)
        )
        if not fragment or not recent:
            return False
        if fragment in recent or recent in fragment:
            return True
        fragment_tokens = set(fragment.split())
        recent_tokens = set(recent.split())
        token_overlap = len(fragment_tokens & recent_tokens) / max(1, len(fragment_tokens))
        similarity = difflib.SequenceMatcher(None, fragment, recent).ratio()
        return max(token_overlap, similarity) >= REALTIME_ECHO_TEXT_SIMILARITY_THRESHOLD

    def _trim_model_transcripts(self, now: float) -> None:
        cutoff = now - max(10.0, REALTIME_ECHO_TRANSCRIPT_GUARD_SEC + 2.0)
        while self._recent_model_transcripts and self._recent_model_transcripts[0][0] < cutoff:
            self._recent_model_transcripts.popleft()

    @staticmethod
    def _normalize_transcript(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", text.lower()))

    @staticmethod
    def _join_transcript_fragments(fragments: list[str]) -> str:
        return " ".join("".join(fragments).split())

    def _flush_console_user(self) -> None:
        text = self._join_transcript_fragments(self._console_user_fragments)
        self._console_user_fragments.clear()
        if text:
            console.you(text)

    def _flush_console_model(self) -> None:
        text = self._join_transcript_fragments(self._console_model_fragments)
        self._console_model_fragments.clear()
        if text:
            console.case(text)

    def _set_state(self, state: str) -> None:
        if REALTIME_STATE_LOG_ON_CHANGE_ONLY and state == self._current_state:
            return
        self._current_state = state
        logger.info("REALTIME_STATE: %s", state)
        if self.state_callback is not None:
            try:
                self.state_callback(state)
            except Exception as exc:
                logger.debug("REALTIME: state callback failed: %s", exc)

    def _handle_local_barge_in(self, rms: float) -> None:
        now = time.monotonic()
        playback_age = now - self.audio_output.playback_started_at
        if playback_age < REALTIME_BARGE_IN_IGNORE_AFTER_PLAYBACK_START_SEC:
            self._barge_in_frames = 0
            if now - self._last_guard_log_at >= 1.0:
                logger.info(
                    "REALTIME: barge-in ignored during playback guard window"
                )
                self._last_guard_log_at = now
            return
        if now - self._last_barge_in_at < REALTIME_BARGE_IN_COOLDOWN_SEC:
            self._barge_in_frames = 0
            if now - self._last_guard_log_at >= 1.0:
                logger.info("REALTIME: barge-in ignored due cooldown")
                self._last_guard_log_at = now
            return

        transcript_is_forming = (
            len(self._last_user_transcript) >= 2
            and now - self._last_user_transcript_at <= 1.5
        )
        if rms < REALTIME_BARGE_IN_RMS or not transcript_is_forming:
            self._barge_in_frames = 0
            self._barge_candidate_at = 0.0
            return

        if self._barge_in_frames == 0:
            self._barge_candidate_at = now
        self._barge_in_frames += 1
        logger.info(
            "REALTIME: barge-in candidate rms=%.0f frames=%s/%s",
            rms,
            self._barge_in_frames,
            REALTIME_BARGE_IN_FRAMES,
        )
        candidate_ms = (now - self._barge_candidate_at) * 1000.0
        if (
            self._barge_in_frames >= REALTIME_BARGE_IN_FRAMES
            and candidate_ms >= REALTIME_BARGE_IN_MIN_SPEECH_MS
            and self.audio_output.request_stop()
        ):
            self._last_barge_in_at = now
            logger.info("REALTIME: barge-in confirmed, stopping playback")
        if not self.audio_output.is_playing:
            self._barge_in_frames = 0
            self._barge_candidate_at = 0.0

    def _log_audio_chunk(self, size: int, now: float) -> None:
        self._chunk_log_count += 1
        self._chunk_log_bytes += size
        elapsed = now - self._chunk_log_started_at
        if elapsed < REALTIME_AUDIO_CHUNK_LOG_INTERVAL_SEC:
            return
        logger.info(
            "REALTIME_AUDIO: received chunks=%s bytes=%s in last %.1fs",
            self._chunk_log_count,
            self._chunk_log_bytes,
            elapsed,
        )
        self._chunk_log_started_at = now
        self._chunk_log_count = 0
        self._chunk_log_bytes = 0

    def _dump_model_audio(self) -> None:
        if not self._model_audio:
            return
        raw_audio = bytes(self._model_audio)
        fx_audio = bytes(self._model_audio_fx)
        self._model_audio.clear()
        self._model_audio_fx.clear()
        root = Path(__file__).resolve().parents[2]
        if self.dump_model_audio_wav:
            destination = Path(REALTIME_DEBUG_AUDIO_DIR)
            if not destination.is_absolute():
                destination = root / destination
            path = destination / "last_model_response.wav"
            self._write_pcm_wav(path, raw_audio)
            logger.info(
                "REALTIME_AUDIO: dumped model audio wav path=%s bytes=%s",
                self._display_path(path),
                len(raw_audio),
            )
        if CASE_VOICE_FX_DUMP_WAV:
            destination = Path(CASE_VOICE_FX_DEBUG_DIR)
            if not destination.is_absolute():
                destination = root / destination
            raw_path = destination / "last_model_response_raw.wav"
            fx_path = destination / "last_model_response_fx.wav"
            self._write_pcm_wav(raw_path, raw_audio)
            self._write_pcm_wav(fx_path, fx_audio)
            logger.info(
                "CASE_VOICE_FX: dumped raw=%s fx=%s",
                self._display_path(raw_path),
                self._display_path(fx_path),
            )

    @staticmethod
    def _write_pcm_wav(path: Path, audio: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24_000)
            wav_file.writeframes(audio)

    @staticmethod
    def _display_path(path: Path) -> Path:
        try:
            return path.relative_to(Path(__file__).resolve().parents[2])
        except ValueError:
            return path

    def _log_latency(self) -> None:
        logger.info("REALTIME_LATENCY:")
        for name in (
            "wake_detected_at",
            "session_connect_start_at",
            "session_connected_at",
            "first_audio_sent_at",
            "first_user_transcript_at",
            "first_model_audio_at",
            "first_audio_play_start_at",
            "tool_call_at",
            "tool_result_sent_at",
            "session_done_at",
        ):
            value = self.metrics.get(name)
            logger.info("  %s = %s", name, f"{value:.6f}" if value else "n/a")
