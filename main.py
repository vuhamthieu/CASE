import os

os.environ["VOSK_LOG_LEVEL"] = "-1"

import asyncio
import logging
import time
import traceback
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent

from src.utils.console_transcript import configure_case_logging, console
from src.audio.playback_manager import close_playback_manager

debug_log_path = configure_case_logging(PROJECT_ROOT)

from middleware.message_bus import AsyncMessageBus
from cognition.personality import CASEPersonality
from actuation.serial_comms import SerialBridge
from actuation.audio_output.tts_engine import CASEVoice
from perception.audio.stt_engine import STATE_IDLE, STTEngine
from src.vision.vision_engine import (
    VISION_ENABLED,
    VISION_GREETING_ENABLED,
    VISION_GREETING_COOLDOWN_SEC,
    VISION_MODE,
    VISION_ON_DEMAND_ONLY,
    VISION_OPEN_CAMERA_ON_BOOT,
    VISION_BACKGROUND_TASK_ENABLED,
    VISION_LOG_MIN_INTERVAL_SEC,
    VISION_IDLE_GREETING_COOLDOWN_AFTER_CONVERSATION_SEC,
    VISION_RUNTIME_ENABLED,
    VISION_RUNTIME_PUBLISH_FRAME_READY,
    VISION_RUNTIME_SAVE_DEBUG_FRAMES,
    VISION_STARTUP_DELAY_SEC,
    VisionEngine,
    run_vision_once,
)
from src.vision.vision_scheduler import (
    VISION_SCHEDULER_ENABLED,
    VISION_STARTUP_ENABLED,
    VisionScheduler,
)
from src.cognition.intent_router import IntentRouter
from src.realtime import resolve_voice_mode
from src.realtime.realtime_config import (
    CASE_CLOUD_STT_FALLBACK,
    CASE_STT_ENDPOINT_BACKEND,
    CASE_STT_FINAL_MODE,
    CASE_STT_LOCAL_FINAL_BACKEND,
    CASE_STT_PROFILE,
    HYBRID_LATENCY_PROFILE,
    HYBRID_MUTE_MIC_DURING_TTS,
    HYBRID_RESUME_MIC_AFTER_TTS_DELAY_SEC,
    HYBRID_STT_ACCEPT_FINAL_ON_SILENCE,
    HYBRID_STT_CONFIRM_SEC,
    HYBRID_STT_DISABLE_REOPEN_AFTER_FINAL,
    HYBRID_STT_FOLLOWUP_TIMEOUT_SEC,
    HYBRID_STT_MAX_COMMAND_SEC,
    HYBRID_STT_MIN_UTTERANCE_SEC,
    HYBRID_STT_REOPEN_SEC,
    HYBRID_STT_SILENCE_SEC,
    HYBRID_TEXT_TTS_STREAMING,
    VOICE_OUTPUT_BACKEND,
    resolve_voice_pipeline,
)
from src.realtime.realtime_voice_engine import RealtimeVoiceEngine
from src.realtime.realtime_tools import RealtimeToolRouter
from src.voice_pipeline.hybrid_text_tts import HybridTextTTSPipeline


WAKEWORD_MODEL_PATH = PROJECT_ROOT / "models" / "wakewords" / "hey_case_v2.onnx"
BOOT_GREETING_GUARD_SECONDS = 6.0
VISION_GREETING_TEXT = "I see you."


logger = logging.getLogger(__name__)

try:
    from display.display_manager import DisplayManager
    from display.bus_adapter import DisplayBusAdapter
except ModuleNotFoundError as exc:
    print(f"DISPLAY import failed: {exc}", flush=True)
    traceback.print_exc()
    DisplayManager = None  # type: ignore[assignment]
    DisplayBusAdapter = None  # type: ignore[assignment]


class VisionEventHandler:
    """Log vision state and allow greetings only in wake-word idle mode."""

    def __init__(
        self,
        bus: AsyncMessageBus,
        is_idle: Callable[[], bool],
        greeting_enabled: bool = VISION_GREETING_ENABLED,
        greeting_cooldown_sec: float = VISION_GREETING_COOLDOWN_SEC,
        conversation_cooldown_sec: float = (
            VISION_IDLE_GREETING_COOLDOWN_AFTER_CONVERSATION_SEC
        ),
        log_min_interval_sec: float = VISION_LOG_MIN_INTERVAL_SEC,
    ) -> None:
        self.bus = bus
        self.is_idle = is_idle
        self.greeting_enabled = greeting_enabled
        self.greeting_cooldown_sec = greeting_cooldown_sec
        self.conversation_cooldown_sec = conversation_cooldown_sec
        self.log_min_interval_sec = log_min_interval_sec
        self.tts_active_count = 0
        self.last_greeting_at = float("-inf")
        self.last_target_log_at = float("-inf")
        self.last_target_log_state = None
        self.last_conversation_ended_at = float("-inf")
        self._conversation_turn_pending = False

        bus.subscribe("TTS_START", self._on_tts_start)
        bus.subscribe("TTS_END", self._on_tts_end)
        bus.subscribe("USER_SPOKE", self._on_user_spoke)
        bus.subscribe("VISION_USER_DETECTED", self._on_user_detected)
        bus.subscribe("VISION_USER_LOST", self._on_user_lost)
        bus.subscribe("VISION_FACE_LEFT", self._on_face_left)
        bus.subscribe("VISION_FACE_CENTER", self._on_face_center)
        bus.subscribe("VISION_FACE_RIGHT", self._on_face_right)
        bus.subscribe("VISION_TARGET_UPDATE", self._on_target_update)
        bus.subscribe("VISION_TARGET_LOST", self._on_target_lost)

    async def _on_tts_start(self, payload):
        self.tts_active_count += 1

    async def _on_tts_end(self, payload):
        self.tts_active_count = max(0, self.tts_active_count - 1)
        if self.tts_active_count == 0 and self._conversation_turn_pending:
            self.last_conversation_ended_at = time.monotonic()
            self._conversation_turn_pending = False

    async def _on_user_spoke(self, payload):
        self._conversation_turn_pending = True
        self.last_conversation_ended_at = time.monotonic()

    async def _on_user_detected(self, payload):
        now = time.monotonic()
        if self.tts_active_count > 0:
            logger.info("VISION: greeting skipped because CASE is speaking")
            return
        if not self.is_idle():
            logger.info("VISION: greeting skipped because conversation active")
            return
        if (
            now - self.last_conversation_ended_at
            < self.conversation_cooldown_sec
        ):
            logger.info("VISION: greeting suppressed after conversation turn")
            return
        if now - self.last_greeting_at < self.greeting_cooldown_sec:
            logger.info("VISION: greeting cooldown active")
            return

        logger.info("VISION: user detected while idle")
        if not self.greeting_enabled:
            return
        self.last_greeting_at = now
        await self.bus.publish("AI_SPEAK", VISION_GREETING_TEXT)
        logger.info("VISION: queued idle user greeting")

    async def _on_user_lost(self, payload):
        logger.info("VISION: user lost")

    async def _on_face_left(self, payload):
        self._log_direction("LEFT", payload)

    async def _on_face_center(self, payload):
        self._log_direction("CENTER", payload)

    async def _on_face_right(self, payload):
        self._log_direction("RIGHT", payload)

    @staticmethod
    def _log_direction(direction: str, payload) -> None:
        if not payload.get("stable", False):
            logger.debug(
                "VISION: ignored unstable direction event direction=%s",
                direction,
            )
            return
        logger.info("VISION: target is %s", direction)

    async def _on_target_update(self, payload):
        now = time.monotonic()
        target_state = (
            payload.get("direction", "UNKNOWN"),
            bool(payload.get("stable", False)),
        )
        if (
            target_state == self.last_target_log_state
            and now - self.last_target_log_at < self.log_min_interval_sec
        ):
            return
        self.last_target_log_at = now
        self.last_target_log_state = target_state
        logger.info(
            "VISION: target direction %s stable=%s",
            target_state[0],
            target_state[1],
        )
        console.vision(
            f"target {target_state[0]} stable={target_state[1]}"
        )

    async def _on_target_lost(self, payload):
        logger.info("VISION: target lost")


async def run_vision_background(vision: VisionEngine) -> None:
    """Delay vision startup and contain failures outside the voice pipeline."""
    try:
        if VISION_STARTUP_DELAY_SEC > 0:
            await asyncio.sleep(VISION_STARTUP_DELAY_SEC)
        logger.info("VISION: background task started")
        await vision.run()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("VISION: disabled after background task failure: %s", exc)
    finally:
        vision.stop()
        console.system("vision stopped")


async def run_vision_scheduler_background(
    scheduler: VisionScheduler,
) -> None:
    """Contain scheduler failures so voice tasks remain authoritative."""
    try:
        await scheduler.run()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("VISION_SCHEDULER: disabled after failure: %s", exc)
    finally:
        scheduler.stop()


async def close_audio_streams() -> None:
    """Best-effort final PortAudio stop after owned streams have closed."""
    try:
        await asyncio.to_thread(close_playback_manager)
        import sounddevice as sd

        await asyncio.to_thread(sd.stop)
        logger.info("AUDIO: sounddevice streams closed")
        console.system("audio streams closed")
    except Exception as exc:
        logger.debug("AUDIO: final sounddevice stop skipped: %s", exc)


async def boot_sequence():
    logger.info("Initializing system components...")
    console.system(f"CASE starting; debug log={debug_log_path.relative_to(PROJECT_ROOT)}")

    voice_mode = resolve_voice_mode()
    voice_pipeline = (
        resolve_voice_pipeline() if voice_mode == "realtime" else "classic"
    )
    hybrid_fast = (
        voice_mode == "realtime"
        and voice_pipeline == "hybrid_text_tts"
        and HYBRID_LATENCY_PROFILE == "fast"
        and CASE_STT_LOCAL_FINAL_BACKEND
        in {"auto", "vosk_lgraph", "vosk_small", "sherpa_sensevoice", "sensevoice", "local_vosk_fast"}
    )

    # Instantiate core components
    bus = AsyncMessageBus()
    bridge = SerialBridge(bus)
    personality = CASEPersonality(
        bus,
        input_topic="CHAT_USER_SPOKE",
        realtime_hybrid=hybrid_fast,
    )
    voice = CASEVoice(bus)
    await voice.prewarm()
    endpoint_kwargs = {}
    if hybrid_fast:
        endpoint_kwargs = {
            "speech_end_silence_sec": HYBRID_STT_SILENCE_SEC,
            "final_confirm_delay_sec": HYBRID_STT_CONFIRM_SEC,
            "min_utterance_sec": HYBRID_STT_MIN_UTTERANCE_SEC,
            "reopen_after_final_sec": HYBRID_STT_REOPEN_SEC,
            "max_command_listen_sec": HYBRID_STT_MAX_COMMAND_SEC,
            "followup_timeout_sec": HYBRID_STT_FOLLOWUP_TIMEOUT_SEC,
            "accept_final_on_silence": HYBRID_STT_ACCEPT_FINAL_ON_SILENCE,
            "disable_reopen_after_final": HYBRID_STT_DISABLE_REOPEN_AFTER_FINAL,
        }
    stt = STTEngine(
        bus,
        wakeword_model_path=WAKEWORD_MODEL_PATH,
        post_tts_guard_seconds=HYBRID_RESUME_MIC_AFTER_TTS_DELAY_SEC,
        mute_during_tts=HYBRID_MUTE_MIC_DURING_TTS,
        cached_wake_ack_enabled=(
            voice_mode == "realtime" and voice_pipeline == "hybrid_text_tts"
        ),
        **endpoint_kwargs,
    )
    vision = None
    vision_scheduler = None
    logger.info("VISION_MODE: %s", VISION_MODE)
    if VISION_ON_DEMAND_ONLY:
        logger.info("VISION: hard off, device closed")
        logger.info("VISION_SCHEDULER: disabled because VISION_ON_DEMAND_ONLY=true")
    elif (
        VISION_ENABLED
        and VISION_RUNTIME_ENABLED
        and VISION_STARTUP_ENABLED
        and VISION_OPEN_CAMERA_ON_BOOT
        and VISION_BACKGROUND_TASK_ENABLED
    ):
        try:
            vision = VisionEngine(
                bus,
                case_state_provider=stt._current_state,
                publish_frame_ready=VISION_RUNTIME_PUBLISH_FRAME_READY,
                scheduler_controlled=VISION_SCHEDULER_ENABLED,
            )
            if not VISION_RUNTIME_SAVE_DEBUG_FRAMES:
                logger.info("VISION: runtime debug frame saving disabled")
            if VISION_SCHEDULER_ENABLED:
                vision_scheduler = VisionScheduler(
                    bus,
                    vision,
                    case_state_provider=stt._current_state,
                )
            VisionEventHandler(
                bus,
                is_idle=lambda: (
                    stt._is_enabled() and stt._current_state() == STATE_IDLE
                ),
            )
        except Exception as exc:
            logger.warning("VISION: disabled during initialization: %s", exc)
    else:
        logger.info("VISION: hard off, device closed")
        logger.info("VISION: runtime disabled by configuration")

    async def vision_once(reason: str, mode: str = "single_frame", **kwargs):
        message_bus = kwargs.pop("message_bus", bus)
        return await run_vision_once(
            reason,
            mode=mode,
            case_state_provider=stt._current_state,
            message_bus=message_bus,
            **kwargs,
        )

    intent_router = IntentRouter(
        bus,
        vision_scheduler=vision_scheduler,
        vision_engine=vision,
        vision_once=vision_once,
    )

    realtime_engine = None
    hybrid_pipeline = None
    if voice_mode == "realtime":
        if voice_pipeline == "gemini_live_native":
            realtime_engine = RealtimeVoiceEngine(
                message_bus=bus,
                shared_audio_queue=stt.audio_queue,
                tool_router=RealtimeToolRouter(
                    vision_scheduler=vision_scheduler,
                    vision_engine=vision,
                    vision_once=vision_once,
                ),
                state_callback=stt.set_external_state,
            )
            stt.set_realtime_session_runner(realtime_engine.run_session)
            logger.info("GEMINI_LIVE_NATIVE_AUDIO: enabled by explicit pipeline")
        else:
            hybrid_pipeline = HybridTextTTSPipeline.from_runtime(
                bus,
                backend=VOICE_OUTPUT_BACKEND,
                streaming=HYBRID_TEXT_TTS_STREAMING,
            )
            logger.info(
                "HYBRID_TEXT_TTS: using local Vosk input, Gemini text, and CASE TTS backend=%s",
                VOICE_OUTPUT_BACKEND,
            )
            logger.info("VOICE_PIPELINE: hybrid_text_tts")
            logger.info("VOICE_OUTPUT_BACKEND: %s", VOICE_OUTPUT_BACKEND)
            if VOICE_OUTPUT_BACKEND == "piper_onnx":
                logger.info("PIPER_ONNX: active for CASE_TTS")
            logger.info("STT_PROFILE: %s", CASE_STT_PROFILE)
            logger.info("LATENCY_PROFILE: %s", HYBRID_LATENCY_PROFILE)
            logger.info("STT_FINAL_MODE: %s", CASE_STT_FINAL_MODE)
            logger.info("STT_ENDPOINT_MODE: %s", CASE_STT_ENDPOINT_BACKEND)
            logger.info("STT_LOCAL_FINAL_BACKEND: %s", CASE_STT_LOCAL_FINAL_BACKEND)
            if CASE_STT_FINAL_MODE == "cloud":
                logger.info("STT_FINAL_FALLBACK: %s", CASE_CLOUD_STT_FALLBACK)
            logger.info("GEMINI_LIVE_NATIVE_AUDIO: disabled")
        logger.info(
            "VOICE_MODE: realtime pipeline=%s (local wake word, shared microphone stream)",
            voice_pipeline,
        )
    else:
        logger.info("VOICE_MODE: classic (Vosk + Gemini text + local TTS)")

    banner = """
===================================================
    PROJECT CASE: DUAL-BRAIN ONLINE
===================================================
"""
    print(banner)

    background_tasks: list[asyncio.Task] = []
    display_manager: DisplayManager | None = None
    try:
        await bus.publish("STT_DISABLE", "booting")
        await asyncio.sleep(0)

        if DisplayManager is not None and DisplayBusAdapter is not None:
            try:
                display_manager = DisplayManager()
                DisplayBusAdapter(bus, display_manager)
                display_manager.start()
                logger.info("DISPLAY: subsystem online and tracking telemetry")
            except Exception as exc:
                display_manager = None
                logger.warning("DISPLAY: subsystem bypassed or failed: %s", exc)
        else:
            logger.warning("DISPLAY: subsystem bypassed; display modules not available")

        # Start long-running loops as background tasks first.
        stt_task = asyncio.create_task(stt.run())
        bridge_task = asyncio.create_task(bridge.listen_loop())
        background_tasks = [stt_task, bridge_task]

        # Give STT a short moment to open the mic stream.
        await asyncio.sleep(0.5)

        await bus.publish(
            "AI_SPEAK",
            "All systems online. I'm here."
        )

        # The message bus schedules TTS asynchronously, so use a guard delay
        # before enabling STT to keep CASE from hearing its own boot greeting.
        await asyncio.sleep(BOOT_GREETING_GUARD_SECONDS)
        await bus.publish("STT_ENABLE", "boot complete")
        if vision is not None:
            background_tasks.append(
                asyncio.create_task(
                    run_vision_background(vision),
                    name="case-vision",
                )
            )
        if vision_scheduler is not None:
            background_tasks.append(
                asyncio.create_task(
                    run_vision_scheduler_background(vision_scheduler),
                    name="case-vision-scheduler",
                )
            )

        await asyncio.gather(*background_tasks)

    except asyncio.CancelledError:
        logger.info("Tasks cancelled, shutting down...")
        raise
    finally:
        if vision_scheduler is not None:
            vision_scheduler.stop()
        if vision is not None:
            vision.stop()
        if display_manager is not None:
            display_manager.stop()
        for task in background_tasks:
            if not task.done():
                task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
        await close_audio_streams()
        if vision is not None:
            logger.info("VISION: background task stopped")


if __name__ == "__main__":
    try:
        asyncio.run(boot_sequence())
    except KeyboardInterrupt:
        print("\nPROJECT CASE shutting down gracefully...")
        console.system("shutdown requested")
    finally:
        logger.info("System shutdown complete.")
        console.system("shutdown complete")
