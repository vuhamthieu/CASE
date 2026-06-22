import asyncio
import logging
import time
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

load_dotenv()

from middleware.message_bus import AsyncMessageBus
from cognition.personality import CASEPersonality
from actuation.serial_comms import SerialBridge
from actuation.audio_output.tts_engine import CASEVoice
from perception.audio.stt_engine import STATE_IDLE, STTEngine
from src.vision.vision_engine import (
    VISION_ENABLED,
    VISION_GREETING_ENABLED,
    VISION_GREETING_COOLDOWN_SEC,
    VISION_LOG_MIN_INTERVAL_SEC,
    VISION_RUNTIME_ENABLED,
    VISION_RUNTIME_PUBLISH_FRAME_READY,
    VISION_RUNTIME_SAVE_DEBUG_FRAMES,
    VISION_STARTUP_DELAY_SEC,
    VisionEngine,
)
from src.vision.vision_scheduler import (
    VISION_SCHEDULER_ENABLED,
    VISION_STARTUP_ENABLED,
    VisionScheduler,
)
from src.cognition.intent_router import IntentRouter


PROJECT_ROOT = Path(__file__).resolve().parent
WAKEWORD_MODEL_PATH = PROJECT_ROOT / "models" / "wakewords" / "hey_case_v2.onnx"
BOOT_GREETING_GUARD_SECONDS = 6.0
VISION_GREETING_TEXT = "I see you, boss."


# Filter system logs to gray color (\033[90m)
class ColoredFormatter(logging.Formatter):
    def format(self, record):
        log_fmt = f"\033[90m%(asctime)s - %(name)s - %(levelname)s - %(message)s\033[0m"
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = logging.getLogger()
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

ch = logging.StreamHandler()
ch.setFormatter(ColoredFormatter())
logger.addHandler(ch)


class VisionEventHandler:
    """Log vision state and allow greetings only in wake-word idle mode."""

    def __init__(
        self,
        bus: AsyncMessageBus,
        is_idle: Callable[[], bool],
        greeting_enabled: bool = VISION_GREETING_ENABLED,
        greeting_cooldown_sec: float = VISION_GREETING_COOLDOWN_SEC,
        log_min_interval_sec: float = VISION_LOG_MIN_INTERVAL_SEC,
    ) -> None:
        self.bus = bus
        self.is_idle = is_idle
        self.greeting_enabled = greeting_enabled
        self.greeting_cooldown_sec = greeting_cooldown_sec
        self.log_min_interval_sec = log_min_interval_sec
        self.tts_active_count = 0
        self.last_greeting_at = float("-inf")
        self.last_target_log_at = float("-inf")
        self.last_target_log_state = None

        bus.subscribe("TTS_START", self._on_tts_start)
        bus.subscribe("TTS_END", self._on_tts_end)
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

    async def _on_user_detected(self, payload):
        now = time.monotonic()
        if self.tts_active_count > 0:
            logger.info("VISION: greeting skipped because CASE is speaking")
            return
        if not self.is_idle():
            logger.info("VISION: greeting skipped because conversation active")
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


async def boot_sequence():
    logger.info("Initializing system components...")

    # Instantiate core components
    bus = AsyncMessageBus()
    bridge = SerialBridge(bus)
    personality = CASEPersonality(bus, input_topic="CHAT_USER_SPOKE")
    voice = CASEVoice(bus)
    stt = STTEngine(
        bus,
        wakeword_model_path=WAKEWORD_MODEL_PATH,
    )
    vision = None
    vision_scheduler = None
    if VISION_ENABLED and VISION_RUNTIME_ENABLED and VISION_STARTUP_ENABLED:
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
        logger.info("VISION: runtime disabled by configuration")

    intent_router = IntentRouter(
        bus,
        vision_scheduler=vision_scheduler,
        vision_engine=vision,
    )

    banner = """
===================================================
    PROJECT CASE: DUAL-BRAIN ONLINE
===================================================
"""
    print(banner)

    background_tasks: list[asyncio.Task] = []
    try:
        await bus.publish("STT_DISABLE", "booting")
        await asyncio.sleep(0)

        # Start long-running loops as background tasks first.
        stt_task = asyncio.create_task(stt.run())
        bridge_task = asyncio.create_task(bridge.listen_loop())
        background_tasks = [stt_task, bridge_task]

        # Give STT a short moment to open the mic stream.
        await asyncio.sleep(0.5)

        await bus.publish(
            "AI_SPEAK",
            "All systems online. Awaiting your command, Boss."
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
        for task in background_tasks:
            if not task.done():
                task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
        if vision is not None:
            logger.info("VISION: background task stopped")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()

    try:
        loop.run_until_complete(boot_sequence())

    except KeyboardInterrupt:
        print("\nPROJECT CASE shutting down gracefully...")

    finally:
        # Gracefully cancel all remaining tasks
        tasks = asyncio.all_tasks(loop=loop)

        for task in tasks:
            task.cancel()

        # Gather all tasks to let them finish cancelling
        group = asyncio.gather(*tasks, return_exceptions=True)
        loop.run_until_complete(group)

        loop.close()
        logger.info("System shutdown complete.")
