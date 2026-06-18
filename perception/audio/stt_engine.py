import asyncio
import json
import logging
import os
import queue
import random
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd
from vosk import KaldiRecognizer, Model


ACK_PHRASES = [
    "I'm here, boss.",
    "Go ahead.",
]


class STTEngine:
    """STT engine using a single sounddevice stream, wake detection,
    and local Vosk transcription.

    Architecture:
    - Single always-open `sounddevice.InputStream` opened in `run()`.
    - Callback pushes raw int16 PCM frames into `self.audio_queue`.
    - Background processing thread reads from that queue and performs:
        * Wake detection using openwakeword if enabled, otherwise VAD/energy.
        * On wake: publish AI_SPEAK acknowledgement.
        * Then stream bytes into Vosk recognizer.
        * On final valid text: publish USER_SPOKE via the message bus.
    - Honors `TTS_START` / `TTS_END` via `self.is_muted`.
    """

    def __init__(
        self,
        message_bus,
        model_path="ai/stt/vosk-model-small-en-us-0.15",
        samplerate: int = 16000,
    ):
        self.bus = message_bus
        self.model_path = model_path
        self.samplerate = samplerate
        self._stream_samplerate = samplerate
        self.is_muted = False

        # Subscribe to TTS events to deafen the microphone while CASE speaks.
        self.bus.subscribe("TTS_START", self._mute)
        self.bus.subscribe("TTS_END", self._unmute)

        # Internal queues and flags.
        self.audio_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=512)
        self._stop_event = threading.Event()
        self._processing_thread: Optional[threading.Thread] = None

        # Load Vosk model.
        logging.info(f"Loading Vosk model from: {self.model_path}")

        if not os.path.isdir(self.model_path):
            # Try resolving relative to repository root.
            repo_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            alt_path = os.path.join(repo_root, self.model_path)

            if os.path.isdir(alt_path):
                logging.info(f"Resolved Vosk model path to: {alt_path}")
                self.model_path = alt_path

        try:
            self.model = Model(self.model_path)

        except Exception as e:
            logging.error(f"Failed to load Vosk model: {e}")
            logging.error("Ensure the model exists at the configured path.")
            logging.error("Example:")
            logging.error("  mkdir -p ai/stt")
            logging.error(
                "  wget -O /tmp/vosk-model.zip "
                "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
            )
            logging.error(
                "  unzip /tmp/vosk-model.zip -d ai/stt"
            )
            raise

        # Current mode: VAD-based wake trigger.
        # If you later add a real wake-word model, set these accordingly.
        self._wake_model = None
        self._use_openwakeword = False
        self._wake_threshold = 0.9

        try:
            import webrtcvad  # type: ignore

            self._vad = webrtcvad.Vad(2)
            logging.info("VAD initialized with aggressiveness=2.")

        except Exception:
            self._vad = None
            logging.info("webrtcvad not available; using energy-based detection.")

        # Speech frame counter used by VAD-based wake detection.
        self._speech_frame_count = 0

        # Event loop reference will be set when `run()` is called.
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    async def _mute(self, payload: str):
        self.is_muted = True
        logging.debug("[STT] Muted by TTS_START.")

    async def _unmute(self, payload: str):
        # Drain the queue so CASE does not hear itself.
        logging.debug("[STT] Unmute requested; draining audio queue...")

        self.is_muted = False

        try:
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()
        except Exception:
            pass

        logging.debug("[STT] Audio queue drained; resuming.")

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice callback — push raw int16 bytes to queue."""
        if status:
            logging.debug(f"InputStream status: {status}")

        try:
            # Flatten to mono int16 bytes.
            data = indata.reshape(-1).astype(np.int16)

            if self._stream_samplerate and self._stream_samplerate != self.samplerate:
                data = self._resample_int16(
                    data,
                    self._stream_samplerate,
                    self.samplerate,
                )

            audio_bytes = data.tobytes()

            try:
                self.audio_queue.put_nowait(audio_bytes)

            except queue.Full:
                # Drop oldest then try again so queue stays fresh.
                try:
                    _ = self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(audio_bytes)
                except Exception:
                    pass

        except Exception as e:
            logging.debug(f"Error in audio callback: {e}")

    def _resample_int16(
        self,
        samples: np.ndarray,
        source_rate: int,
        target_rate: int,
    ) -> np.ndarray:
        """Lightweight mono resampler for int16 PCM."""
        if source_rate == target_rate or samples.size == 0:
            return samples.astype(np.int16, copy=False)

        source = samples.astype(np.float32)
        target_length = max(
            1,
            int(round(len(source) * float(target_rate) / float(source_rate))),
        )

        x_old = np.linspace(0.0, 1.0, num=len(source), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=target_length, endpoint=False)

        resampled = np.interp(x_new, x_old, source)

        return np.clip(resampled, -32768, 32767).astype(np.int16)

    def _energy_wake_detector(
        self,
        frame_bytes: bytes,
        threshold: float = 2000.0,
    ) -> bool:
        """Simple energy-based wake detection on int16 PCM bytes."""
        try:
            arr = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32)

            if arr.size == 0:
                return False

            rms = np.sqrt(np.mean(arr * arr))
            return rms > threshold

        except Exception:
            return False

    def _is_valid_transcript(self, text: str) -> bool:
        """Reject empty, tiny, or junk STT results before sending to LLM."""
        cleaned = " ".join(text.strip().lower().split())

        if not cleaned:
            return False

        # Reject one-letter / tiny garbage.
        if len(cleaned) < 2:
            return False

        # Require at least two alphabetic characters.
        alpha_count = sum(ch.isalpha() for ch in cleaned)
        if alpha_count < 2:
            return False

        # Common junk/filler phrases that should not trigger the LLM.
        junk_phrases = {
            "huh",
            "uh",
            "um",
            "umm",
            "ah",
            "er",
            "eh",
            "noise",
            "static",
            "background noise",
        }

        if cleaned in junk_phrases:
            return False

        return True

    def _publish_user_spoke_if_valid(self, text: str) -> bool:
        """Publish USER_SPOKE only if transcript passes quality filter."""
        text = text.strip()

        if not self._is_valid_transcript(text):
            logging.info(f"Ignoring low-quality transcript: {text!r}")
            return False

        if self.loop:
            asyncio.run_coroutine_threadsafe(
                self.bus.publish("USER_SPOKE", text),
                self.loop,
            )

        print(f"\033[92m[You]: {text}\033[0m")
        return True

    def _publish_ai_speak_from_thread(self, text: str) -> None:
        """Thread-safe AI_SPEAK publish from processing thread."""
        if self.loop:
            asyncio.run_coroutine_threadsafe(
                self.bus.publish("AI_SPEAK", text),
                self.loop,
            )

    def _drain_audio_queue(self) -> None:
        """Remove queued audio frames."""
        try:
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()
        except Exception:
            pass

    def _wait_for_tts_to_finish(self, start_timeout: float = 3.0) -> None:
        """Wait until TTS_START occurs, then wait for TTS_END."""
        deadline = time.time() + start_timeout

        # Wait briefly for TTS_START to set self.is_muted.
        while not self.is_muted and time.time() < deadline:
            time.sleep(0.05)

        # Wait for TTS_END to unmute.
        while self.is_muted:
            time.sleep(0.05)

    def _detect_wake(self, frame: bytes) -> bool:
        """Wake detection using openwakeword if available, otherwise VAD/energy."""
        if self._use_openwakeword and getattr(self, "_wake_model", None) is not None:
            try:
                pred = self._wake_model.predict(
                    np.frombuffer(frame, dtype=np.int16)
                )
                return any(score > self._wake_threshold for score in pred.values())

            except Exception:
                return False

        if getattr(self, "_vad", None) is not None:
            # webrtcvad needs exact 20ms frames at 16kHz = 640 bytes.
            frame_bytes = 640

            if not hasattr(self, "_vad_buffer"):
                self._vad_buffer = b""

            self._vad_buffer += frame

            while len(self._vad_buffer) >= frame_bytes:
                chunk = self._vad_buffer[:frame_bytes]
                self._vad_buffer = self._vad_buffer[frame_bytes:]

                try:
                    is_speech = self._vad.is_speech(
                        chunk,
                        sample_rate=16000,
                    )

                    if is_speech:
                        self._speech_frame_count += 1
                    else:
                        self._speech_frame_count = max(
                            0,
                            self._speech_frame_count - 1,
                        )

                    # 8 x 20ms = about 160ms of speech.
                    if self._speech_frame_count >= 8:
                        self._speech_frame_count = 0
                        self._vad_buffer = b""
                        return True

                except Exception as ex:
                    logging.debug(f"VAD error: {ex}")
                    self._vad_buffer = b""
                    self._speech_frame_count = 0
                    return False

            return False

        return self._energy_wake_detector(frame)

    def _processing_loop(self):
        """Background thread: wake spotting + Vosk transcription."""
        while not self._stop_event.is_set():
            try:
                try:
                    frame = self.audio_queue.get(timeout=0.2)

                except queue.Empty:
                    continue

                # If muted, drop frame and reset VAD state.
                if self.is_muted:
                    self._speech_frame_count = 0

                    if hasattr(self, "_vad_buffer"):
                        self._vad_buffer = b""

                    continue

                woke = self._detect_wake(frame)

                if not woke:
                    continue

                # Wake detected — acknowledge before recording prompt.
                logging.info("Wake detected — acknowledging before transcription.")

                ack = random.choice(ACK_PHRASES)
                self._publish_ai_speak_from_thread(ack)

                # Wait for acknowledgement speech to finish so CASE does not hear itself.
                self._wait_for_tts_to_finish()

                # Drain any audio captured during ack playback.
                self._drain_audio_queue()

                # Create a fresh recognizer instance for this utterance.
                recognizer = KaldiRecognizer(self.model, self.samplerate)

                last_audio_time = time.time()
                final_text = ""

                # Collect subsequent frames until end-of-speech.
                while not self._stop_event.is_set():
                    try:
                        frame = self.audio_queue.get(timeout=1.0)
                        last_audio_time = time.time()

                        if self.is_muted:
                            logging.info("Muted during transcription — aborting.")
                            break

                        is_final = recognizer.AcceptWaveform(frame)

                        if is_final:
                            try:
                                res = json.loads(recognizer.Result())
                                text = res.get("text", "").strip()

                                if self._publish_user_spoke_if_valid(text):
                                    final_text += text + " "

                            except Exception as ex:
                                logging.debug(f"Failed to parse Vosk result: {ex}")

                            break

                    except queue.Empty:
                        # No audio for a while means end of utterance.
                        if time.time() - last_audio_time > 1.0:
                            try:
                                res = json.loads(recognizer.FinalResult())
                                text = res.get("text", "").strip()

                                if self._publish_user_spoke_if_valid(text):
                                    final_text += text + " "

                            except Exception as ex:
                                logging.debug(f"Failed to parse Vosk final result: {ex}")

                            break

                if final_text:
                    logging.info(f"Transcribed: {final_text.strip()}")

            except Exception as e:
                logging.error(f"Error in STT processing loop: {e}")

        logging.info("STT processing loop exiting.")

    async def run(self):
        """Open the mic stream once and run processing in background thread."""
        self.loop = asyncio.get_running_loop()

        self._stop_event.clear()
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
        )

        stream = None

        try:
            stream_rate = self.samplerate

            try:
                device_info = sd.query_devices(kind="input")
                default_rate = int(device_info.get("default_samplerate") or 0)

            except Exception:
                default_rate = 0

            try:
                stream = sd.InputStream(
                    samplerate=stream_rate,
                    channels=1,
                    dtype="int16",
                    callback=self._audio_callback,
                )

            except Exception:
                if default_rate and default_rate != stream_rate:
                    logging.info(
                        "16 kHz capture unavailable; "
                        f"falling back to device default rate {default_rate} Hz."
                    )

                    stream_rate = default_rate

                    stream = sd.InputStream(
                        samplerate=stream_rate,
                        channels=1,
                        dtype="int16",
                        callback=self._audio_callback,
                    )

                else:
                    raise

            self._stream_samplerate = stream_rate
            stream.start()

        except Exception as e:
            logging.error(f"Failed to open microphone stream: {e}")
            raise

        # Start processing thread.
        self._processing_thread.start()

        try:
            while True:
                await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            logging.info("STT run cancelled — shutting down.")

        finally:
            self._stop_event.set()

            if self._processing_thread and self._processing_thread.is_alive():
                self._processing_thread.join(timeout=1.0)

            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
