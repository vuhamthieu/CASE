import asyncio
import logging
from pathlib import Path

from middleware.message_bus import AsyncMessageBus
from cognition.personality import CASEPersonality
from actuation.serial_comms import SerialBridge
from actuation.audio_output.tts_engine import CASEVoice
from perception.audio.stt_engine import STTEngine
from dotenv import load_dotenv

load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent
WAKEWORD_MODEL_PATH = PROJECT_ROOT / "models" / "wakewords" / "hey_case_v2.onnx"
BOOT_GREETING_GUARD_SECONDS = 6.0


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

        await asyncio.gather(
            stt_task,
            bridge_task,
        )

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
