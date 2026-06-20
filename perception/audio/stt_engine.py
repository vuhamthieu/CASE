import asyncio
import json
import logging
import os
import queue
import random
import re
import threading
import time
from collections import deque
from math import gcd
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from vosk import KaldiRecognizer, Model as VoskModel


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        logging.warning("Invalid %s; using default %.2f", name, default)
        return default


ACTIVE_CONVERSATION_ENABLED = True
INITIAL_COMMAND_TIMEOUT_SEC = _env_float("CASE_INITIAL_COMMAND_TIMEOUT_SEC", 8.0)
FOLLOWUP_TIMEOUT_SEC = _env_float("CASE_FOLLOWUP_TIMEOUT_SEC", 6.0)
SPEECH_END_SILENCE_SEC = _env_float("CASE_SPEECH_END_SILENCE_SEC", 1.3)
FINAL_CONFIRM_DELAY_SEC = _env_float("CASE_FINAL_CONFIRM_DELAY_SEC", 0.5)
MIN_UTTERANCE_SEC = _env_float("CASE_MIN_UTTERANCE_SEC", 0.8)
ALLOW_REOPEN_AFTER_FINAL_SEC = _env_float(
    "CASE_ALLOW_REOPEN_AFTER_FINAL_SEC",
    0.7,
)
MAX_COMMAND_LISTEN_SEC = _env_float("CASE_MAX_COMMAND_LISTEN_SEC", 12.0)
POST_TTS_RESUME_DELAY_SEC = 0.4
MAX_SESSION_SECONDS = 120.0
REQUIRE_DIRECTED_SPEECH_IN_FOLLOWUP = False
CONVERSATION_MODE_ENABLED = True
LONG_CONVERSATION_TIMEOUT_SEC = 120.0
LONG_FOLLOWUP_TIMEOUT_SEC = 20.0

TARGET_SAMPLE_RATE = 16_000
WAKE_FRAME_SECONDS = 0.08
WAKE_FRAME_SAMPLES = int(TARGET_SAMPLE_RATE * WAKE_FRAME_SECONDS)
WAKE_TTS_START_TIMEOUT_SEC = 3.0
RESPONSE_TTS_START_TIMEOUT_SEC = 45.0
MIN_TRANSCRIPT_CHARS = 4
MIN_TRANSCRIPT_WORDS = 2
SPEECH_RMS_THRESHOLD = 250.0
WAKE_THRESHOLD = 0.995
WAKE_STRONG_THRESHOLD = 0.998
WAKE_MIN_HITS = 3
WAKE_HIT_WINDOW_SEC = 0.7
WAKE_COOLDOWN_SEC = 2.0

STATE_IDLE = "IDLE"
STATE_WAKE_ACK = "WAKE_ACK"
STATE_LISTEN_COMMAND = "LISTEN_COMMAND"
STATE_THINKING = "THINKING"
STATE_SPEAKING = "SPEAKING"
STATE_SHORT_FOLLOW_UP = "SHORT_FOLLOW_UP"
STATE_LONG_CONVERSATION = "LONG_CONVERSATION"

STT_STATE_LISTENING = "LISTENING"
STT_STATE_POSSIBLE_END = "POSSIBLE_END"
STT_STATE_FINAL_CONFIRM = "FINAL_CONFIRM"
STT_STATE_FINAL_ACCEPTED = "FINAL_ACCEPTED"
STT_STATE_REOPENED_AFTER_FINAL = "REOPENED_AFTER_FINAL"

ACK_PHRASES = [
    "I'm listening, boss.",
    "Go ahead.",
]

FILLER_TRANSCRIPTS = {
    "huh",
    "uh",
    "um",
    "umm",
    "ah",
    "er",
    "eh",
    "hm",
    "hmm",
    "static",
    "noise",
    "background noise",
}

GARBAGE_TRANSCRIPTS = {
    "hey",
    "case",
    "okay",
    "one",
    "the",
    "a",
}

GARBAGE_WORDS = {
    "a",
    "an",
    "the",
    "one",
    "two",
    "to",
    "too",
    "uh",
    "um",
    "hm",
    "hmm",
    "oh",
    "okay",
}

FOLLOWUP_COMMAND_STARTERS = {
    "what",
    "who",
    "where",
    "when",
    "why",
    "how",
    "can",
    "could",
    "would",
    "should",
    "tell",
    "show",
    "give",
    "make",
    "do",
    "are",
    "is",
    "explain",
    "continue",
    "also",
    "and",
    "then",
}

WAKE_ONLY_TRANSCRIPTS = {
    "hey case",
    "hay case",
    "hey cases",
    "case",
}

END_SESSION_PHRASES = {
    "stop",
    "sleep",
    "go to sleep",
    "stop listening",
    "that's all",
    "thats all",
    "bye",
    "goodbye",
    "thôi",
    "ngủ đi",
    "dừng lại",
    "im đi",
    "kết thúc",
}

LONG_CONVERSATION_TRIGGERS = {
    "let's talk",
    "lets talk",
    "keep listening",
    "conversation mode",
    "talk with me",
    "continue listening",
}


class STTEngine:
    """Wake-word-gated active conversation STT for CASE.

    Wake word opens a short conversation session. Inside that session, follow-up
    prompts are transcribed directly with Vosk until the user is silent too long,
    says an end-session phrase, or the max session time is reached.
    """

    def __init__(
        self,
        message_bus,
        model_path: str | os.PathLike = "ai/stt/vosk-model-small-en-us-0.15",
        wakeword_model_path: str | os.PathLike = "models/wakewords/hey_case_v2.onnx",
        samplerate: int = TARGET_SAMPLE_RATE,
        wake_threshold: float = WAKE_THRESHOLD,
        wake_strong_threshold: float = WAKE_STRONG_THRESHOLD,
        wake_min_hits: int = WAKE_MIN_HITS,
        wake_hit_window_sec: float = WAKE_HIT_WINDOW_SEC,
        wake_cooldown_sec: float = WAKE_COOLDOWN_SEC,
        post_tts_guard_seconds: float = POST_TTS_RESUME_DELAY_SEC,
    ):
        self.bus = message_bus
        self.repo_root = Path(__file__).resolve().parents[2]
        self.model_path = self._resolve_path(model_path)
        self.wakeword_model_path = self._resolve_path(wakeword_model_path)
        self.samplerate = samplerate
        self._stream_samplerate = samplerate
        self._stream_channels = 1
        self._resample_up = 1
        self._resample_down = 1

        self.wake_threshold = wake_threshold
        self.wake_strong_threshold = wake_strong_threshold
        self.wake_min_hits = wake_min_hits
        self.wake_hit_window_sec = wake_hit_window_sec
        self.wake_cooldown_sec = wake_cooldown_sec
        self.post_tts_guard_seconds = post_tts_guard_seconds
        self._last_wake_time = 0.0
        self._wake_hits: deque[tuple[float, float]] = deque()
        self._wake_scores: deque[tuple[float, float]] = deque()

        self._enabled = True
        self._tts_active_count = 0
        self._post_tts_guard_until = 0.0
        self._state = STATE_IDLE
        self._state_lock = threading.Lock()
        self._tts_started_event = threading.Event()
        self._tts_idle_event = threading.Event()
        self._tts_idle_event.set()
        self._last_published_transcript = ""

        self.audio_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=512)
        self._stop_event = threading.Event()
        self._processing_thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        self.bus.subscribe("TTS_START", self._on_tts_start)
        self.bus.subscribe("TTS_END", self._on_tts_end)
        self.bus.subscribe("STT_DISABLE", self._disable_stt)
        self.bus.subscribe("STT_ENABLE", self._enable_stt)

        self._validate_wake_settings()
        logging.info(
            "STT endpointing: silence=%.2fs, confirm=%.2fs, min_utterance=%.2fs, "
            "reopen=%.2fs, max_command=%.2fs, followup_timeout=%.2fs",
            SPEECH_END_SILENCE_SEC,
            FINAL_CONFIRM_DELAY_SEC,
            MIN_UTTERANCE_SEC,
            ALLOW_REOPEN_AFTER_FINAL_SEC,
            MAX_COMMAND_LISTEN_SEC,
            FOLLOWUP_TIMEOUT_SEC,
        )
        self._load_vosk_model()
        self._load_wakeword_model()

    def _resolve_path(self, path: str | os.PathLike) -> Path:
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            resolved = self.repo_root / resolved
        return resolved.resolve()

    def _validate_wake_settings(self) -> None:
        if self.wake_min_hits < 1:
            raise ValueError("wake_min_hits must be at least 1")
        if self.wake_hit_window_sec <= 0:
            raise ValueError("wake_hit_window_sec must be greater than 0")
        if self.wake_cooldown_sec < 0:
            raise ValueError("wake_cooldown_sec must not be negative")

    def _load_vosk_model(self) -> None:
        logging.info("Loading Vosk model from: %s", self.model_path)

        if not self.model_path.is_dir():
            raise FileNotFoundError(
                "Vosk model folder is missing: "
                f"{self.model_path}\n"
                "Place the Vosk model at ai/stt/vosk-model-small-en-us-0.15 "
                "or pass model_path=... to STTEngine."
            )

        self.model = VoskModel(str(self.model_path))

    def _load_wakeword_model(self) -> None:
        data_path = Path(f"{self.wakeword_model_path}.data")

        logging.info("Loading wake-word model: %s", self.wakeword_model_path.stem)
        logging.info("Wake-word ONNX path: %s", self.wakeword_model_path)
        logging.info("Wake-word threshold: %.3f", self.wake_threshold)
        logging.info("Wake-word strong threshold: %.3f", self.wake_strong_threshold)
        logging.info(
            "Wake-word confirmation: min_hits=%s, hit_window_sec=%.2f, cooldown_sec=%.2f",
            self.wake_min_hits,
            self.wake_hit_window_sec,
            self.wake_cooldown_sec,
        )
        logging.info("Wake-word ONNX exists: %s", self.wakeword_model_path.is_file())
        logging.info("Wake-word ONNX data path: %s", data_path)
        logging.info("Wake-word ONNX data exists: %s", data_path.is_file())

        if not self.wakeword_model_path.is_file():
            raise FileNotFoundError(
                f"Wake-word ONNX file is missing: {self.wakeword_model_path}"
            )
        if not data_path.is_file():
            raise FileNotFoundError(
                f"Wake-word ONNX data file is missing: {data_path}"
            )

        try:
            from openwakeword.model import Model as WakeWordModel
        except ImportError as exc:
            raise RuntimeError(
                "openWakeWord is not installed. Install it in the venv with: "
                "python3 -m pip install openwakeword onnxruntime"
            ) from exc

        self.wakeword_name = self.wakeword_model_path.stem
        self.wakeword_model = WakeWordModel(
            wakeword_models=[str(self.wakeword_model_path)],
            inference_framework="onnx",
        )

    def _transition(self, new_state: str) -> None:
        with self._state_lock:
            old_state = self._state
            if old_state == new_state:
                return
            self._state = new_state

        logging.info("STATE: %s -> %s", old_state, new_state)

    def _current_state(self) -> str:
        with self._state_lock:
            return self._state

    async def _disable_stt(self, payload: str) -> None:
        with self._state_lock:
            self._enabled = False
        self._drain_audio_queue()
        logging.info("STT disabled: %s", payload)

    async def _enable_stt(self, payload: str) -> None:
        self._drain_audio_queue()
        with self._state_lock:
            self._enabled = True
            self._post_tts_guard_until = time.monotonic() + self.post_tts_guard_seconds
        logging.info("STT enabled: %s", payload)

    async def _on_tts_start(self, payload: str) -> None:
        with self._state_lock:
            self._tts_active_count += 1
            active_count = self._tts_active_count
            state = self._state
        self._tts_started_event.set()
        self._tts_idle_event.clear()

        if state == STATE_THINKING:
            self._transition(STATE_SPEAKING)

        logging.info("TTS_START received; active TTS count=%s", active_count)

    async def _on_tts_end(self, payload: str) -> None:
        with self._state_lock:
            self._tts_active_count = max(0, self._tts_active_count - 1)
            active_count = self._tts_active_count
            if active_count == 0:
                self._post_tts_guard_until = (
                    time.monotonic() + self.post_tts_guard_seconds
                )

        if active_count == 0:
            self._tts_idle_event.set()
            self._drain_audio_queue()

        logging.info("TTS_END received; active TTS count=%s", active_count)

    def _is_enabled(self) -> bool:
        with self._state_lock:
            return self._enabled

    def _tts_active(self) -> bool:
        with self._state_lock:
            return self._tts_active_count > 0

    def _should_ignore_audio(self) -> bool:
        now = time.monotonic()
        with self._state_lock:
            return (
                not self._enabled
                or self._tts_active_count > 0
                or now < self._post_tts_guard_until
            )

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice callback: convert to 16 kHz mono int16 bytes."""
        if status:
            logging.debug("InputStream status: %s", status)

        if self._should_ignore_audio():
            return

        try:
            data = np.asarray(indata)

            if data.ndim == 2:
                if data.shape[1] > 1:
                    data = data.astype(np.float32).mean(axis=1)
                else:
                    data = data[:, 0]

            data = data.astype(np.float32).reshape(-1)

            if self._stream_samplerate != self.samplerate:
                data = resample_poly(data, self._resample_up, self._resample_down)

            data = np.clip(
                np.rint(data),
                np.iinfo(np.int16).min,
                np.iinfo(np.int16).max,
            ).astype(np.int16)

            if data.size == 0:
                return

            try:
                self.audio_queue.put_nowait(data.tobytes())
            except queue.Full:
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(data.tobytes())
                except Exception:
                    pass

        except Exception as exc:
            logging.debug("Error in audio callback: %s", exc)

    def _normalize_transcript(self, text: str) -> str:
        cleaned = " ".join(text.strip().lower().split())
        cleaned = cleaned.replace("’", "'")
        return cleaned

    def _is_end_session_command(self, text: str) -> bool:
        cleaned = self._normalize_transcript(text)
        return any(phrase in cleaned for phrase in END_SESSION_PHRASES)

    def _is_long_conversation_trigger(self, text: str) -> bool:
        cleaned = self._normalize_transcript(text)
        return any(phrase in cleaned for phrase in LONG_CONVERSATION_TRIGGERS)

    def _transcript_reject_reason(
        self,
        text: str,
        *,
        followup: bool = False,
    ) -> Optional[str]:
        cleaned = self._normalize_transcript(text)

        if not cleaned:
            return "empty"
        if self._is_end_session_command(cleaned):
            return None
        if CONVERSATION_MODE_ENABLED and self._is_long_conversation_trigger(cleaned):
            return None
        if len(cleaned) < MIN_TRANSCRIPT_CHARS:
            return "too_short"
        if cleaned in FILLER_TRANSCRIPTS:
            return "filler"
        if cleaned in GARBAGE_TRANSCRIPTS:
            return "garbage"
        if cleaned in WAKE_ONLY_TRANSCRIPTS:
            return "wake_word_only"
        if cleaned == self._last_published_transcript:
            return "duplicate"

        word_count = len(re.findall(r"\w+", cleaned, flags=re.UNICODE))
        if word_count < MIN_TRANSCRIPT_WORDS:
            return "too_few_words"

        alpha_count = sum(ch.isalpha() for ch in cleaned)
        if alpha_count < MIN_TRANSCRIPT_CHARS:
            return "not_enough_letters"

        words = re.findall(r"\w+", cleaned, flags=re.UNICODE)
        unique_words = set(words)
        if len(words) >= 3 and len(unique_words) <= 1:
            return "repeated_filler"

        if followup:
            if len(cleaned) < 8:
                return "followup_too_short"
            if word_count < 2:
                return "followup_too_few_words"
            if words and all(word in GARBAGE_WORDS for word in words):
                return "followup_garbage"
            if word_count <= 2 and words and words[0] not in FOLLOWUP_COMMAND_STARTERS:
                return "followup_unclear"
            if REQUIRE_DIRECTED_SPEECH_IN_FOLLOWUP and "case" not in cleaned:
                return "not_directed"

        return None

    def _publish_user_spoke(self, text: str, *, followup: bool = False) -> bool:
        text = text.strip()
        reject_reason = self._transcript_reject_reason(text, followup=followup)

        if reject_reason:
            logging.info("TRANSCRIPT ignored: reason=%s text=%r", reject_reason, text)
            return False

        if self.loop:
            asyncio.run_coroutine_threadsafe(
                self.bus.publish("USER_SPOKE", text),
                self.loop,
            )

        self._last_published_transcript = self._normalize_transcript(text)
        logging.info("Published USER_SPOKE: %s", text)
        print(f"\033[92m[You]: {text}\033[0m")
        return True

    def _publish_ai_speak_from_thread(self, text: str):
        if not self.loop:
            return None

        return asyncio.run_coroutine_threadsafe(
            self.bus.publish("AI_SPEAK", text),
            self.loop,
        )

    def _drain_audio_queue(self) -> None:
        try:
            while True:
                self.audio_queue.get_nowait()
        except queue.Empty:
            pass
        except Exception:
            pass

    def _reset_wake_history(self) -> None:
        self._wake_hits.clear()
        self._wake_scores.clear()

    def _prepare_for_tts_wait(self) -> None:
        self._tts_started_event.clear()
        if not self._tts_active():
            self._tts_idle_event.set()

    def _wait_for_tts_cycle(self, start_timeout: float) -> bool:
        if not self._tts_started_event.wait(timeout=start_timeout):
            logging.warning("Timed out waiting for TTS_START.")
            return False

        while not self._tts_idle_event.wait(timeout=0.1):
            if self._stop_event.is_set():
                return False

        self._drain_audio_queue()
        time.sleep(self.post_tts_guard_seconds)
        self._drain_audio_queue()
        return True

    def _say_and_wait(self, text: str, start_timeout: float) -> bool:
        self._prepare_for_tts_wait()
        future = self._publish_ai_speak_from_thread(text)
        if future is not None:
            try:
                future.result(timeout=2.0)
            except Exception as exc:
                logging.debug("AI_SPEAK publish did not finish cleanly: %s", exc)

        return self._wait_for_tts_cycle(start_timeout=start_timeout)

    def _model_score_from_prediction(self, prediction: dict) -> float:
        if self.wakeword_name in prediction:
            return float(prediction[self.wakeword_name])

        if not prediction:
            return 0.0

        return float(max(prediction.values()))

    def _predict_wake_frame(self, frame: np.ndarray) -> tuple[bool, float]:
        frame = np.asarray(frame, dtype=np.int16).reshape(-1)

        if frame.shape != (WAKE_FRAME_SAMPLES,):
            raise ValueError(
                "Wake-word frame must have shape "
                f"({WAKE_FRAME_SAMPLES},), got {frame.shape}"
            )

        prediction = self.wakeword_model.predict(frame)
        score = self._model_score_from_prediction(prediction)
        now = time.monotonic()

        self._wake_scores.append((now, score))
        while self._wake_scores and now - self._wake_scores[0][0] > 1.0:
            self._wake_scores.popleft()

        if score >= self.wake_threshold:
            self._wake_hits.append((now, score))

        while (
            self._wake_hits
            and now - self._wake_hits[0][0] > self.wake_hit_window_sec
        ):
            self._wake_hits.popleft()

        hit_count = len(self._wake_hits)
        window_max = max((hit_score for _, hit_score in self._wake_hits), default=0.0)
        cooldown_ready = now - self._last_wake_time >= self.wake_cooldown_sec
        confirmed = (
            score >= self.wake_threshold
            and hit_count >= self.wake_min_hits
            and window_max >= self.wake_strong_threshold
            and cooldown_ready
        )

        if score >= 0.5 or confirmed:
            logging.info(
                "Wake score %s=%.3f, hit_count=%s, window_max=%.3f, confirmed=%s",
                self.wakeword_name,
                score,
                hit_count,
                window_max,
                confirmed,
            )
        else:
            logging.debug(
                "Wake score %s=%.3f, hit_count=%s, window_max=%.3f, confirmed=%s",
                self.wakeword_name,
                score,
                hit_count,
                window_max,
                confirmed,
            )

        if confirmed:
            self._last_wake_time = now
            self._reset_wake_history()

        return confirmed, score

    def _parse_vosk_text(self, recognizer: KaldiRecognizer, final: bool) -> str:
        try:
            raw = recognizer.FinalResult() if final else recognizer.Result()
            result = json.loads(raw)
            return result.get("text", "").strip()
        except Exception as exc:
            logging.debug("Failed to parse Vosk result: %s", exc)
            return ""

    def _parse_vosk_partial(self, recognizer: KaldiRecognizer) -> str:
        try:
            result = json.loads(recognizer.PartialResult())
            return result.get("partial", "").strip()
        except Exception:
            return ""

    @staticmethod
    def _join_transcript_parts(parts: list[str]) -> str:
        return " ".join(part.strip() for part in parts if part.strip()).strip()

    def _frame_rms(self, frame: bytes) -> float:
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples * samples)))

    def _listen_for_transcript(
        self,
        listen_timeout: float,
        label: str,
        *,
        followup: bool = False,
    ) -> Optional[str]:
        listen_started_at = time.monotonic()
        speech_wait_deadline = listen_started_at + listen_timeout
        recognizer = KaldiRecognizer(self.model, self.samplerate)
        transcript_parts: list[str] = []
        speech_started_at: Optional[float] = None
        last_speech_at: Optional[float] = None
        active_speech_duration = 0.0
        confirm_started_at: Optional[float] = None
        forced_final = False
        endpoint_state = STT_STATE_LISTENING
        last_partial_text = ""

        def log_stt_state(new_state: str, **details) -> None:
            nonlocal endpoint_state
            endpoint_state = new_state
            detail_text = " ".join(f"{key}={value}" for key, value in details.items())
            logging.info(
                "STT_STATE: %s%s",
                new_state,
                f" {detail_text}" if detail_text else "",
            )

        def reset_after_rejected_candidate() -> None:
            nonlocal recognizer
            nonlocal speech_started_at, last_speech_at, active_speech_duration
            nonlocal confirm_started_at, last_partial_text, forced_final
            recognizer = KaldiRecognizer(self.model, self.samplerate)
            transcript_parts.clear()
            speech_started_at = None
            last_speech_at = None
            active_speech_duration = 0.0
            confirm_started_at = None
            forced_final = False
            last_partial_text = ""
            log_stt_state(STT_STATE_LISTENING)

        def append_transcript_part(text: str) -> None:
            cleaned = text.strip()
            if cleaned and (not transcript_parts or transcript_parts[-1] != cleaned):
                transcript_parts.append(cleaned)

        def start_final_confirm(now: float) -> None:
            nonlocal recognizer, confirm_started_at
            silence_duration = now - (last_speech_at or now)
            partial_text = self._parse_vosk_partial(recognizer)
            log_stt_state(
                STT_STATE_POSSIBLE_END,
                speech_started_at=f"{speech_started_at:.6f}" if speech_started_at else "n/a",
                last_speech_at=f"{last_speech_at:.6f}" if last_speech_at else "n/a",
                silence_duration=f"{silence_duration:.3f}",
                partial_text=repr(partial_text),
            )

            append_transcript_part(self._parse_vosk_text(recognizer, final=True))
            recognizer = KaldiRecognizer(self.model, self.samplerate)
            confirm_started_at = now
            candidate = self._join_transcript_parts(transcript_parts)
            log_stt_state(
                STT_STATE_FINAL_CONFIRM,
                final_candidate=repr(candidate),
                confirm_delay=f"{FINAL_CONFIRM_DELAY_SEC:.2f}",
                reopen_window=f"{ALLOW_REOPEN_AFTER_FINAL_SEC:.2f}",
                reopened_after_final=False,
            )

        log_stt_state(STT_STATE_LISTENING)

        logging.info("Listening for user prompt (%s, timeout %.1fs)...", label, listen_timeout)

        while not self._stop_event.is_set():
            now = time.monotonic()
            if speech_started_at is None and now >= speech_wait_deadline:
                break

            if (
                speech_started_at is not None
                and now - speech_started_at >= MAX_COMMAND_LISTEN_SEC
                and confirm_started_at is None
            ):
                logging.info(
                    "STT maximum command duration reached: %.2fs",
                    MAX_COMMAND_LISTEN_SEC,
                )
                forced_final = True
                start_final_confirm(now)

            if not self._is_enabled():
                logging.info("STT disabled during %s listening; aborting.", label)
                return None

            if self._tts_active():
                logging.info("TTS started during %s listening; aborting.", label)
                return None

            try:
                frame = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                frame = None

            if frame is not None and self._should_ignore_audio():
                continue

            now = time.monotonic()
            rms = self._frame_rms(frame) if frame is not None else 0.0
            speech_present = frame is not None and rms >= SPEECH_RMS_THRESHOLD

            if speech_present:
                if speech_started_at is None:
                    speech_started_at = now
                    logging.info("%s speech started, rms=%.1f", label, rms)
                elif not forced_final and endpoint_state in {
                    STT_STATE_POSSIBLE_END,
                    STT_STATE_FINAL_CONFIRM,
                }:
                    log_stt_state(
                        STT_STATE_REOPENED_AFTER_FINAL,
                        speech_started_at=f"{speech_started_at:.6f}",
                        last_speech_at=f"{last_speech_at:.6f}" if last_speech_at else "n/a",
                        silence_duration=(
                            f"{now - last_speech_at:.3f}" if last_speech_at else "n/a"
                        ),
                        final_candidate=repr(
                            self._join_transcript_parts(transcript_parts)
                        ),
                        reopened_after_final=True,
                    )
                    confirm_started_at = None
                    log_stt_state(STT_STATE_LISTENING)

                last_speech_at = now
                if frame is not None:
                    active_speech_duration += (
                        len(frame) / np.dtype(np.int16).itemsize / self.samplerate
                    )

            if speech_started_at is None:
                continue

            if frame is not None:
                try:
                    if recognizer.AcceptWaveform(frame):
                        append_transcript_part(
                            self._parse_vosk_text(recognizer, final=False)
                        )
                        if endpoint_state == STT_STATE_LISTENING:
                            log_stt_state(
                                STT_STATE_POSSIBLE_END,
                                speech_started_at=f"{speech_started_at:.6f}",
                                last_speech_at=(
                                    f"{last_speech_at:.6f}"
                                    if last_speech_at
                                    else "n/a"
                                ),
                                silence_duration=(
                                    f"{now - (last_speech_at or now):.3f}"
                                ),
                                final_candidate=repr(
                                    self._join_transcript_parts(transcript_parts)
                                ),
                            )
                        logging.info(
                            "Vosk final candidate held for endpoint confirmation: %r",
                            self._join_transcript_parts(transcript_parts),
                        )
                    else:
                        partial_text = self._parse_vosk_partial(recognizer)
                        if partial_text and partial_text != last_partial_text:
                            last_partial_text = partial_text
                            logging.debug("STT partial_text=%r", partial_text)
                except Exception as exc:
                    logging.debug("Failed while feeding Vosk recognizer: %s", exc)

            silence_duration = now - (last_speech_at or now)
            if (
                confirm_started_at is None
                and silence_duration >= SPEECH_END_SILENCE_SEC
            ):
                start_final_confirm(now)

            if confirm_started_at is None:
                continue

            confirmation_window = max(
                FINAL_CONFIRM_DELAY_SEC,
                ALLOW_REOPEN_AFTER_FINAL_SEC,
            )
            if now - confirm_started_at < confirmation_window:
                continue

            candidate = self._join_transcript_parts(transcript_parts)
            if active_speech_duration < MIN_UTTERANCE_SEC:
                reject_reason = "utterance_too_short"
            else:
                reject_reason = self._transcript_reject_reason(
                    candidate,
                    followup=followup,
                )

            if reject_reason:
                logging.info(
                    "TRANSCRIPT ignored: reason=%s text=%r duration=%.3f",
                    reject_reason,
                    candidate,
                    active_speech_duration,
                )
                if followup:
                    return None
                reset_after_rejected_candidate()
                continue

            log_stt_state(
                STT_STATE_FINAL_ACCEPTED,
                speech_started_at=f"{speech_started_at:.6f}",
                last_speech_at=f"{last_speech_at:.6f}" if last_speech_at else "n/a",
                silence_duration=f"{silence_duration:.3f}",
                final_candidate=repr(candidate),
                confirm_delay=f"{now - confirm_started_at:.3f}",
                reopened_after_final=False,
            )
            return candidate

        logging.info("TRANSCRIPT ignored: reason=session_timeout label=%s", label)
        return None

    def _wait_for_assistant_response(self, next_state: str) -> bool:
        spoken = self._wait_for_tts_cycle(start_timeout=RESPONSE_TTS_START_TIMEOUT_SEC)
        if not spoken:
            logging.warning("No assistant TTS response observed after USER_SPOKE.")
            return False

        if self._current_state() == STATE_SPEAKING:
            self._transition(next_state)

        return True

    def _handle_prompt(
        self,
        text: str,
        session_start: float,
        *,
        followup: bool = False,
        long_mode: bool = False,
    ) -> tuple[bool, bool]:
        """Return (keep_session_open, long_mode)."""
        logging.info("TRANSCRIPT accepted: %r", text)

        if self._is_end_session_command(text):
            logging.info("SESSION ended: reason=user_command_sleep text=%r", text)
            self._transition(STATE_SPEAKING)
            self._say_and_wait(
                "Standing by, boss.",
                start_timeout=WAKE_TTS_START_TIMEOUT_SEC,
            )
            self._transition(STATE_IDLE)
            return False, long_mode

        if CONVERSATION_MODE_ENABLED and self._is_long_conversation_trigger(text):
            logging.info("SESSION mode: long_conversation trigger=%r", text)
            self._transition(STATE_LONG_CONVERSATION)
            return True, True

        self._transition(STATE_THINKING)
        self._prepare_for_tts_wait()
        if not self._publish_user_spoke(text, followup=followup and not long_mode):
            if followup and not long_mode:
                logging.info("SESSION ended: reason=ignored_short_followup")
                self._transition(STATE_IDLE)
                return False, long_mode
            return True, long_mode

        next_state = STATE_LONG_CONVERSATION if long_mode else STATE_SHORT_FOLLOW_UP
        self._wait_for_assistant_response(next_state)

        max_seconds = LONG_CONVERSATION_TIMEOUT_SEC if long_mode else MAX_SESSION_SECONDS
        if time.monotonic() - session_start >= max_seconds:
            logging.info("SESSION ended: reason=max_session_seconds")
            self._transition(STATE_IDLE)
            return False, long_mode

        if not ACTIVE_CONVERSATION_ENABLED:
            logging.info("SESSION ended: reason=active_conversation_disabled")
            self._transition(STATE_IDLE)
            return False, long_mode

        if self._current_state() == STATE_THINKING:
            self._transition(next_state)

        return self._current_state() in {STATE_SHORT_FOLLOW_UP, STATE_LONG_CONVERSATION}, long_mode

    def _run_conversation_session(self) -> None:
        session_start = time.monotonic()
        long_mode = False

        self._transition(STATE_WAKE_ACK)
        ack = random.choice(ACK_PHRASES)
        self._say_and_wait(ack, start_timeout=WAKE_TTS_START_TIMEOUT_SEC)
        self._drain_audio_queue()

        self._transition(STATE_LISTEN_COMMAND)
        text = self._listen_for_transcript(
            INITIAL_COMMAND_TIMEOUT_SEC,
            label="initial command",
        )

        if not text:
            logging.info("SESSION ended: reason=initial_command_timeout")
            self._transition(STATE_IDLE)
            return

        keep_session, long_mode = self._handle_prompt(
            text,
            session_start,
            followup=False,
            long_mode=long_mode,
        )
        if not keep_session:
            return

        while (
            ACTIVE_CONVERSATION_ENABLED
            and not self._stop_event.is_set()
            and self._current_state() in {STATE_SHORT_FOLLOW_UP, STATE_LONG_CONVERSATION}
        ):
            max_seconds = LONG_CONVERSATION_TIMEOUT_SEC if long_mode else MAX_SESSION_SECONDS
            if time.monotonic() - session_start >= max_seconds:
                logging.info("SESSION ended: reason=max_session_seconds")
                self._transition(STATE_IDLE)
                return

            listen_timeout = LONG_FOLLOWUP_TIMEOUT_SEC if long_mode else FOLLOWUP_TIMEOUT_SEC
            label = "long conversation" if long_mode else "short follow-up"
            text = self._listen_for_transcript(
                listen_timeout,
                label=label,
                followup=not long_mode,
            )

            if not text:
                reason = "long_conversation_timeout" if long_mode else "short_followup_timeout"
                logging.info("SESSION ended: reason=%s", reason)
                self._transition(STATE_IDLE)
                return

            keep_session, long_mode = self._handle_prompt(
                text,
                session_start,
                followup=True,
                long_mode=long_mode,
            )
            if not keep_session:
                return

    def _processing_loop(self):
        """Background thread: IDLE wake spotting + active conversation sessions."""
        wake_buffer = np.empty(0, dtype=np.int16)

        while not self._stop_event.is_set():
            try:
                try:
                    audio_bytes = self.audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                if self._should_ignore_audio():
                    wake_buffer = np.empty(0, dtype=np.int16)
                    self._reset_wake_history()
                    continue

                if self._current_state() != STATE_IDLE:
                    continue

                samples = np.frombuffer(audio_bytes, dtype=np.int16)
                if samples.size == 0:
                    continue

                wake_buffer = np.concatenate((wake_buffer, samples))

                while len(wake_buffer) >= WAKE_FRAME_SAMPLES:
                    frame = wake_buffer[:WAKE_FRAME_SAMPLES]
                    wake_buffer = wake_buffer[WAKE_FRAME_SAMPLES:]

                    confirmed, score = self._predict_wake_frame(frame)
                    if not confirmed:
                        continue

                    logging.info(
                        "Wake word detected: %s score=%.3f",
                        self.wakeword_name,
                        score,
                    )

                    self._drain_audio_queue()
                    wake_buffer = np.empty(0, dtype=np.int16)
                    self._run_conversation_session()
                    self._drain_audio_queue()
                    self._reset_wake_history()
                    break

            except Exception as exc:
                logging.error("Error in STT processing loop: %s", exc)
                self._transition(STATE_IDLE)

        logging.info("STT processing loop exiting.")

    async def run(self):
        """Open the mic stream once and run processing in a background thread."""
        self.loop = asyncio.get_running_loop()
        self._stop_event.clear()
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
        )

        stream = None

        try:
            try:
                device_info = sd.query_devices(kind="input")
                default_rate = int(round(device_info.get("default_samplerate") or 0))
                device_name = device_info.get("name", "default input")
                max_channels = int(device_info.get("max_input_channels") or 1)
            except Exception:
                default_rate = self.samplerate
                device_name = "default input"
                max_channels = 1

            stream_rate = default_rate or self.samplerate
            divisor = gcd(stream_rate, self.samplerate)
            self._resample_up = self.samplerate // divisor
            self._resample_down = stream_rate // divisor
            blocksize = max(1, int(round(stream_rate * WAKE_FRAME_SECONDS)))

            self._stream_samplerate = stream_rate
            self._stream_channels = 1
            logging.info(
                "Starting mic stream from '%s' at %s Hz, %s channel(s); "
                "resampling to %s Hz for wake/Vosk.",
                device_name,
                stream_rate,
                self._stream_channels,
                self.samplerate,
            )

            try:
                stream = sd.InputStream(
                    samplerate=stream_rate,
                    blocksize=blocksize,
                    channels=self._stream_channels,
                    dtype="int16",
                    callback=self._audio_callback,
                    latency="high",
                )
            except Exception:
                if max_channels <= 1:
                    raise

                self._stream_channels = min(2, max_channels)
                logging.info(
                    "Mono mic stream failed; opening %s channels and downmixing.",
                    self._stream_channels,
                )
                stream = sd.InputStream(
                    samplerate=stream_rate,
                    blocksize=blocksize,
                    channels=self._stream_channels,
                    dtype="int16",
                    callback=self._audio_callback,
                    latency="high",
                )

            stream.start()
            logging.info("Mic stream started.")

        except Exception as exc:
            logging.error("Failed to open microphone stream: %s", exc)
            raise

        self._processing_thread.start()

        try:
            while True:
                await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            logging.info("STT run cancelled; shutting down.")

        finally:
            self._stop_event.set()

            if self._processing_thread and self._processing_thread.is_alive():
                self._processing_thread.join(timeout=1.5)

            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass

            logging.info("STT shutdown complete.")
