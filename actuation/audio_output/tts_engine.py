import os
import asyncio
import logging
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from middleware.message_bus import AsyncMessageBus

logger = logging.getLogger(__name__)

class CASEVoice:
    def __init__(self, bus: 'AsyncMessageBus'):
        self.bus = bus
        # Dynamically find the project root (CASE directory) relative to this file
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.piper_bin = os.path.join(self.base_dir, "ai/tts/piper/piper")
        # Switch to medium model for much faster Pi inference without hurting quality
        self.model = os.path.join(self.base_dir, "ai/tts/en_US-ryan-medium.onnx")
        
        # Subscribe to AI responses
        self.bus.subscribe("AI_SPEAK", self.handle_speak_request)

    async def handle_speak_request(self, text: str) -> None:
        """Generates and plays audio asynchronously using a streaming pipeline."""
        print(f"\033[96m[CASE]: {text}\033[0m")
        logger.info(f"Synthesizing audio for: {text}")
        
        await self.bus.publish("TTS_START", "CASE speaking")
        
        def _run_pipeline():
            # For Piper to speak faster on Pi, limit threads using OMP_NUM_THREADS
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            
            piper_cmd = [self.piper_bin, "--model", self.model, "--output_raw"]
            aplay_cmd = ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-", "-D", "default"]
            
            # 1. Start Piper
            piper_proc = subprocess.Popen(
                piper_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env
            )
            
            # 2. Start aplay, piping Piper's stdout into aplay's stdin
            aplay_proc = subprocess.Popen(
                aplay_cmd,
                stdin=piper_proc.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Close Piper's stdout in this process so aplay receives EOF when Piper finishes
            if piper_proc.stdout is not None:
                piper_proc.stdout.close()
                
            # Safely pass the text securely as bytes, avoiding any shell injection
            piper_proc.communicate(input=text.encode('utf-8'))
            
            # Ensure aplay finishes playing before returning
            aplay_proc.communicate()

        # Run the blocking pipeline in a background thread
        await asyncio.to_thread(_run_pipeline)
        
        logging.info("Audio playback finished")
        await self.bus.publish("TTS_END", "CASE finished")

