import asyncio
import logging
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from middleware.message_bus import AsyncMessageBus
from cognition.personality import CASEPersonality
from actuation.serial_comms import SerialBridge
from actuation.audio_output.tts_engine import CASEVoice
from perception.audio.stt_engine import STTEngine
from src.vision.vision_engine import (
    VISION_ENABLED,
    VISION_GREETING_ENABLED,
    VISION_GREETING_COOLDOWN_SEC,
    VisionEngine,
)


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


class VisionGreetingGate:
    """Allow a presence greeting only when conversation and TTS are idle."""

    def __init__(self, bus, cooldown_sec=VISION_GREETING_COOLDOWN_SEC):
        self.bus = bus
        self.cooldown_sec = cooldown_sec
        self.conversation_active = False
        self.tts_active_count = 0
        self.last_greeting_at = float("-inf")

        bus.subscribe("USER_SPOKE", self._on_user_spoke)
        bus.subscribe("TTS_START", self._on_tts_start)
        bus.subscribe("TTS_END", self._on_tts_end)
        bus.subscribe("VISION_USER_DETECTED", self._on_user_detected)

    async def _on_user_spoke(self, payload):
        self.conversation_active = True

    async def _on_tts_start(self, payload):
        self.tts_active_count += 1

    async def _on_tts_end(self, payload):
        self.tts_active_count = max(0, self.tts_active_count - 1)
        if self.tts_active_count == 0:
            self.conversation_active = False

    async def _on_user_detected(self, payload):
        now = time.monotonic()
        if self.conversation_active or self.tts_active_count > 0:
            logger.info("VISION: greeting skipped because CASE is busy")
            return
        if now - self.last_greeting_at < self.cooldown_sec:
            logger.debug("VISION: greeting skipped due to cooldown")
            return

        self.last_greeting_at = now
        await self.bus.publish("AI_SPEAK", VISION_GREETING_TEXT)
        logger.info("VISION: queued idle user greeting")


async def boot_sequence():
    logger.info("Initializing system components...")

    # Instantiate core components
    bus = AsyncMessageBus()
    bridge = SerialBridge(bus)
    personality = CASEPersonality(bus)
    voice = CASEVoice(bus)
    stt = STTEngine(
        bus,
        wakeword_model_path=WAKEWORD_MODEL_PATH,
    )
    vision = None
    if VISION_ENABLED:
        try:
            vision = VisionEngine(bus)
            if VISION_GREETING_ENABLED:
                VisionGreetingGate(bus)
        except Exception as exc:
            logger.warning("Vision disabled: %s", exc)
    else:
        logger.info("VISION: disabled by VISION_ENABLED")

    banner = """
===================================================
    PROJECT CASE: DUAL-BRAIN ONLINE
===================================================
"""
    print(banner)

    try:
        await bus.publish("STT_DISABLE", "booting")
        await asyncio.sleep(0)

        # Start long-running loops as background tasks first.
        stt_task = asyncio.create_task(stt.run())
        bridge_task = asyncio.create_task(bridge.listen_loop())

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

        background_tasks = [stt_task, bridge_task]
        if vision is not None:
            background_tasks.append(asyncio.create_task(vision.run()))

        await asyncio.gather(*background_tasks)

    except asyncio.CancelledError:
        logger.info("Tasks cancelled, shutting down...")


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
