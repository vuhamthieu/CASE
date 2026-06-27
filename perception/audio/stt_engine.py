import asyncio
import concurrent.futures
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
from typing import Awaitable, Callable, Optional

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from vosk import KaldiRecognizer, Model as VoskModel

try:
    from vosk import SetLogLevel as VoskSetLogLevel
except ImportError:
    VoskSetLogLevel = None

from src.utils.console_transcript import CASE_CONSOLE_MODE, console
from src.audio.input_device import configured_input_device
from src.config import defaults as case_defaults
from src.config.env import get_bool, get_str
from src.realtime.realtime_config import (
    CASE_STT_ACCEPT_FAST_CANDIDATE_ON_TIMEOUT,
    CASE_STT_FINAL_BACKEND,
    CASE_STT_LGRAPH_FINAL_TIMEOUT_SEC,
    CASE_STT_PROFILE,
    GTCRN_MODEL_PATH,
    HYBRID_HOLD_WEAK_ENDING_SEC,
    HYBRID_REJECT_TOO_SHORT,
    HYBRID_REJECT_WAKEWORD_ONLY,
    HYBRID_WEAK_ENDING_ENABLED,
    SHERPA_SENSEVOICE_MODEL_DIR,
    SILERO_VAD_MIN_SILENCE_MS,
    SILERO_VAD_MIN_SPEECH_MS,
    SILERO_VAD_MODEL_PATH,
    SILERO_VAD_SPEECH_PAD_MS,
    SILERO_VAD_THRESHOLD,
    SMART_TURN_HOLD_SEC,
    SMART_TURN_ACCEPT_IMMEDIATELY_IF_COMPLETE,
    SMART_TURN_FAST_ACCEPT_PROBABILITY,
    SMART_TURN_MIN_AUDIO_SEC,
    SMART_TURN_MODEL_PATH,
    SMART_TURN_THRESHOLD,
    STT_USE_GTCRN_DENOISER,
    STT_USE_SILERO_VAD,
    STT_USE_SMART_TURN,
    TRANSCRIPT_INPUT_BACKEND,
    VOSK_LGRAPH_MODEL_PATH,
    VOSK_SMALL_MODEL_PATH,
)
from src.stt_backends.sherpa_sensevoice_backend import SherpaSenseVoiceBackend
from src.stt_backends.smart_turn import SmartTurnDetector, has_weak_ending
from src.stt_backends.stt_profile import resolve_stt_profile
from src.stt_backends.transcript_selection import (
    choose_final_transcript,
    dedupe_repeated_transcript,
    is_usable_transcript,
    normalize_transcript,
)
from src.stt_backends.transcript_repair import (
    malformed_transcript_reason,
    repair_common_transcript,
)
from src.stt_backends.vad_gate import GtcrnDenoiser, SileroVadGate
from src.voice_pipeline.wake_ack import (
    choose_wake_ack,
    play_wake_ack_asset,
    play_wake_ack_beep,
)


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
STT_PREROLL_MS = _env_float("STT_PREROLL_MS", 500.0)
STT_SPEECH_PAD_MS = _env_float("STT_SPEECH_PAD_MS", 300.0)
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
WAKE_DISABLE_DURING_TTS = get_bool("WAKE_DISABLE_DURING_TTS", True)
WAKE_POST_TTS_COOLDOWN_SEC = _env_float("WAKE_POST_TTS_COOLDOWN_SEC", 3.0)
WAKE_POST_WAKE_ACK_COOLDOWN_SEC = _env_float("WAKE_POST_WAKE_ACK_COOLDOWN_SEC", 0.5)
WAKE_POST_FOLLOWUP_TIMEOUT_COOLDOWN_SEC = _env_float(
    "WAKE_POST_FOLLOWUP_TIMEOUT_COOLDOWN_SEC", 2.0
)
WAKE_CLEAR_RING_BUFFER_ON_STATE_CHANGE = get_bool(
    "WAKE_CLEAR_RING_BUFFER_ON_STATE_CHANGE", True
)
WAKE_REJECT_STALE_AUDIO_FRAMES = get_bool("WAKE_REJECT_STALE_AUDIO_FRAMES", True)
WAKE_COOLDOWN_AFTER_IGNORED_TRANSCRIPT_SEC = _env_float(
    "WAKE_COOLDOWN_AFTER_IGNORED_TRANSCRIPT_SEC", 1.5
)
FAST_INTENT_PATTERNS = (
    r"\b(tell me|say|make)\s+(a\s+)?joke\b",
    r"\btell me something funny\b",
    r"\b(can you\s+)?roast me\b",
    r"\btell me about yourself\b",
    r"\b(can you\s+)?see me\b",
    r"\blook around\b",
    r"\bstop\b",
    r"\bcancel\b",
)

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

FOLLOWUP_FILLERS = {
    "yeah",
    "yea",
    "yes",
    "ok",
    "okay",
    "uh",
    "um",
    "well",
    "so",
}

FOLLOWUP_ACCEPT_PATTERNS = (
    "what ",
    "what is ",
    "what are ",
    "how ",
    "why ",
    "when ",
    "where ",
    "who ",
    "can you ",
    "could you ",
    "would you ",
    "do you ",
    "did you ",
    "are you ",
    "is it ",
    "tell me ",
    "explain ",
    "give me ",
)

FOLLOWUP_MEANINGFUL_PHRASES = {
    "very funny",
    "that is very funny",
    "that was funny",
    "thats funny",
    "that's funny",
    "that is funny",
    "not funny",
    "that was bad",
    "boring",
    "haha",
    "lol",
    "nice",
    "good one",
    "try again",
    "yeah that's your problem",
    "yeah thats your problem",
    "that's your problem",
    "thats your problem",
    "your problem",
    "sounds like your problem",
    "not my problem",
    "that's not my problem",
    "thats not my problem",
    "good luck with that",
    "you deal with it",
    "that's on you",
    "thats on you",
    "you should move out",
    "yeah you should move out",
    "you should leave",
    "then move out",
    "that sounds like your problem",
}

FOLLOWUP_MORE_REQUESTS = {
    "again",
    "one more",
    "another one",
    "tell me more",
    "make it longer",
    "make it shorter",
    "funnier",
    "more funny",
    "tell me another one",
    "tell me something longer",
    "tell me something funnier",
    "continue",
    "go on",
}

FOLLOWUP_CORRECTION_STARTERS = {
    "sorry",
    "i mean",
    "no i mean",
    "actually",
    "wait",
}

INITIAL_CLEAR_SHORT_COMMANDS = {
    "tell me",
    "listen",
    "stop",
    "continue",
    "again",
    "look",
    "wake up",
}

INITIAL_FRAGMENT_PHRASES = {
    "to hear me",
}

JOKE_CONTEXT_WORDS = {"joke", "funny", "roast", "laugh", "punchline"}

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
        model_path: str | os.PathLike | None = None,
        wakeword_model_path: str | os.PathLike = "models/wakewords/hey_case_v2.onnx",
        samplerate: int = TARGET_SAMPLE_RATE,
        wake_threshold: float = WAKE_THRESHOLD,
        wake_strong_threshold: float = WAKE_STRONG_THRESHOLD,
        wake_min_hits: int = WAKE_MIN_HITS,
        wake_hit_window_sec: float = WAKE_HIT_WINDOW_SEC,
        wake_cooldown_sec: float = WAKE_COOLDOWN_SEC,
        post_tts_guard_seconds: float = POST_TTS_RESUME_DELAY_SEC,
        mute_during_tts: bool = True,
        speech_end_silence_sec: float = SPEECH_END_SILENCE_SEC,
        final_confirm_delay_sec: float = FINAL_CONFIRM_DELAY_SEC,
        min_utterance_sec: float = MIN_UTTERANCE_SEC,
        reopen_after_final_sec: float = ALLOW_REOPEN_AFTER_FINAL_SEC,
        max_command_listen_sec: float = MAX_COMMAND_LISTEN_SEC,
        followup_timeout_sec: float = FOLLOWUP_TIMEOUT_SEC,
        accept_final_on_silence: bool = True,
        disable_reopen_after_final: bool = False,
        cached_wake_ack_enabled: bool = False,
        input_device: int | str | None = None,
    ):
        self.bus = message_bus
        self.repo_root = Path(__file__).resolve().parents[2]
        self.stt_plan = resolve_stt_profile(
            CASE_STT_PROFILE,
            CASE_STT_FINAL_BACKEND or TRANSCRIPT_INPUT_BACKEND,
        )
        self.transcript_backend = self.stt_plan.preferred_final_mode
        self.model_path = self._select_vosk_model_path(model_path)
        self.vosk_lgraph_model_path = self._resolve_vosk_lgraph_model_path()
        self.final_vosk_model_path: Optional[Path] = None
        self.final_mode = "vosk_small"
        self.final_fallback_mode = ""
        self.lgraph_final_timeout_sec = max(0.0, CASE_STT_LGRAPH_FINAL_TIMEOUT_SEC)
        self.accept_fast_candidate_on_timeout = (
            CASE_STT_ACCEPT_FAST_CANDIDATE_ON_TIMEOUT
        )
        self.wakeword_model_path = self._resolve_path(wakeword_model_path)
        self.samplerate = samplerate
        self.input_device = (
            configured_input_device() if input_device is None else input_device
        )
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
        self.mute_during_tts = mute_during_tts
        self.speech_end_silence_sec = speech_end_silence_sec
        self.final_confirm_delay_sec = final_confirm_delay_sec
        self.min_utterance_sec = min_utterance_sec
        self.reopen_after_final_sec = reopen_after_final_sec
        self.max_command_listen_sec = max_command_listen_sec
        self.followup_timeout_sec = followup_timeout_sec
        self.accept_final_on_silence = accept_final_on_silence
        self.disable_reopen_after_final = disable_reopen_after_final
        self.cached_wake_ack_enabled = cached_wake_ack_enabled
        self._last_wake_time = 0.0
        self._last_tts_end_time = 0.0
        self._wake_suppressed_until = 0.0
        self._wake_suppression_reason = ""
        self._wake_hits: deque[tuple[float, float]] = deque()
        self._wake_scores: deque[tuple[float, float]] = deque()

        self._enabled = True
        self._listening_for_transcript = False
        self._tts_active_count = 0
        self._post_tts_guard_until = 0.0
        self._state = STATE_IDLE
        self._state_lock = threading.Lock()
        self._tts_started_event = threading.Event()
        self._tts_idle_event = threading.Event()
        self._tts_idle_event.set()
        self._last_published_transcript = ""
        self._session_turn_metrics: dict[str, float] = {}
        self._last_transcript_metrics: dict[str, float] = {}
        self.vad_gate: Optional[SileroVadGate] = None
        self.smart_turn: Optional[SmartTurnDetector] = None
        self.denoiser: Optional[GtcrnDenoiser] = None
        self.sensevoice: Optional[SherpaSenseVoiceBackend] = None
        self.final_vosk_model: Optional[VoskModel] = None
        self._final_stt_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="case-stt-final",
        )

        self.audio_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=512)
        self._stop_event = threading.Event()
        self._processing_thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._realtime_session_runner: Optional[
            Callable[[float], Awaitable[bool]]
        ] = None

        self.bus.subscribe("TTS_START", self._on_tts_start)
        self.bus.subscribe("TTS_END", self._on_tts_end)
        self.bus.subscribe("STT_DISABLE", self._disable_stt)
        self.bus.subscribe("STT_ENABLE", self._enable_stt)

        self._validate_wake_settings()
        logging.info(
            "STT endpointing: silence=%.2fs, confirm=%.2fs, min_utterance=%.2fs, "
            "reopen=%.2fs, max_command=%.2fs, followup_timeout=%.2fs",
            self.speech_end_silence_sec,
            self.final_confirm_delay_sec,
            self.min_utterance_sec,
            self.reopen_after_final_sec,
            self.max_command_listen_sec,
            self.followup_timeout_sec,
        )
        self._load_vosk_model()
        self._load_optional_stt_components()
        self._load_wakeword_model()

    def _resolve_path(self, path: str | os.PathLike) -> Path:
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            resolved = self.repo_root / resolved
        return resolved.resolve()

    def _select_vosk_model_path(
        self,
        explicit_path: str | os.PathLike | None,
    ) -> Path:
        if explicit_path is not None:
            return self._resolve_path(explicit_path)
        small = self._resolve_path(VOSK_SMALL_MODEL_PATH)
        logging.info("STT_ENDPOINT_BACKEND: vosk_small model=%s", small)
        return small

    def _resolve_vosk_lgraph_model_path(self) -> Path:
        lgraph = self._resolve_path(VOSK_LGRAPH_MODEL_PATH)
        legacy_lgraph = self.repo_root / Path(VOSK_LGRAPH_MODEL_PATH).name
        if not lgraph.is_dir() and legacy_lgraph.is_dir():
            logging.warning(
                "STT_MODEL: using legacy root lgraph path; move it to %s when convenient",
                lgraph,
            )
            lgraph = legacy_lgraph.resolve()
        return lgraph

    def _load_optional_stt_components(self) -> None:
        if "sensevoice" in self.stt_plan.final_chain:
            try:
                self.sensevoice = SherpaSenseVoiceBackend(
                    self._resolve_path(SHERPA_SENSEVOICE_MODEL_DIR)
                )
                logging.info("STT_BACKEND: sherpa_sensevoice final transcription enabled")
            except Exception as exc:
                logging.warning(
                    "STT_BACKEND: SenseVoice unavailable: %s",
                    exc,
                )
                fallback = self._first_available_final_after("sensevoice")
                logging.warning(
                    "STT_PROFILE_DEGRADED: requested=%s missing=sensevoice fallback=%s",
                    self.stt_plan.requested_label,
                    fallback,
                )

        if STT_USE_SILERO_VAD:
            try:
                self.vad_gate = SileroVadGate(
                    self._resolve_path(SILERO_VAD_MODEL_PATH),
                    sample_rate=self.samplerate,
                    threshold=SILERO_VAD_THRESHOLD,
                    min_speech_ms=SILERO_VAD_MIN_SPEECH_MS,
                    min_silence_ms=SILERO_VAD_MIN_SILENCE_MS,
                )
                logging.info(
                    "VAD: silero enabled model=%s",
                    self._resolve_path(SILERO_VAD_MODEL_PATH),
                )
            except Exception as exc:
                logging.warning(
                    "VAD: silero unavailable; using RMS endpointing fallback: %s",
                    exc,
                )
                logging.info("VAD_MODE: rms_fallback")
                logging.warning(
                    "RUNTIME_ARTIFACTS: degraded_turn_taking missing=%s",
                    SILERO_VAD_MODEL_PATH,
                )

        if STT_USE_SMART_TURN:
            try:
                self.smart_turn = SmartTurnDetector(
                    self._resolve_path(SMART_TURN_MODEL_PATH),
                    threshold=SMART_TURN_THRESHOLD,
                    sample_rate=self.samplerate,
                )
                logging.info(
                    "SMART_TURN: enabled model=%s",
                    self._resolve_path(SMART_TURN_MODEL_PATH),
                )
            except Exception as exc:
                logging.warning(
                    "SMART_TURN: unavailable; using timing/weak-ending fallback: %s",
                    exc,
                )
                logging.info("TURN_END_MODE: vad_timing")
                logging.info("OPTIONAL_ARTIFACT_MISSING: smart_turn")

        if STT_USE_GTCRN_DENOISER:
            try:
                self.denoiser = GtcrnDenoiser(
                    self._resolve_path(GTCRN_MODEL_PATH),
                    sample_rate=self.samplerate,
                )
                logging.info("GTCRN: enabled model=%s", self._resolve_path(GTCRN_MODEL_PATH))
            except Exception as exc:
                logging.warning("GTCRN: unavailable; using raw microphone audio: %s", exc)
        else:
            logging.info("GTCRN: disabled (benchmark before enabling on Pi 4)")

        self._configure_final_stt_runtime()
        self._log_stt_profile_runtime()

    def _final_mode_available(self, mode: str) -> bool:
        if mode == "sensevoice":
            return self.sensevoice is not None
        if mode == "vosk_lgraph":
            return self.vosk_lgraph_model_path.is_dir()
        if mode == "vosk_small":
            return self.model_path.is_dir()
        return False

    def _first_available_final_after(self, mode: str) -> str:
        try:
            start = self.stt_plan.final_chain.index(mode) + 1
        except ValueError:
            start = 0
        for candidate in self.stt_plan.final_chain[start:]:
            if self._final_mode_available(candidate):
                return candidate
        return "vosk_small"

    def _configure_final_stt_runtime(self) -> None:
        missing_lgraph = (
            "vosk_lgraph" in self.stt_plan.final_chain
            and not self.vosk_lgraph_model_path.is_dir()
        )
        if missing_lgraph:
            if self.stt_plan.profile == "balanced" or self.stt_plan.final_backend == "vosk_lgraph":
                logging.warning(
                    "STT_PROFILE_DEGRADED: requested=%s missing=vosk_lgraph "
                    "fallback=vosk_small accuracy=lower",
                    self.stt_plan.requested_label,
                )
            else:
                logging.warning(
                    "STT_PROFILE_DEGRADED: requested=%s missing=vosk_lgraph "
                    "fallback=vosk_small",
                    self.stt_plan.requested_label,
                )

        selected = "vosk_small"
        for candidate in self.stt_plan.final_chain:
            if self._final_mode_available(candidate):
                selected = candidate
                break

        self.final_mode = selected
        self.transcript_backend = selected
        self.final_fallback_mode = self._first_available_final_after(selected)
        if self.final_fallback_mode == selected:
            self.final_fallback_mode = ""
        self.final_vosk_model_path = (
            self.vosk_lgraph_model_path if selected == "vosk_lgraph" else None
        )

        if selected == "vosk_lgraph":
            try:
                self._ensure_final_vosk_model()
            except Exception as exc:
                logging.warning(
                    "STT_BACKEND: vosk_lgraph failed to load; falling back to small: %s",
                    exc,
                )
                self.final_mode = "vosk_small"
                self.transcript_backend = "vosk_small"
                self.final_fallback_mode = ""
                self.final_vosk_model_path = None

    def _log_stt_profile_runtime(self) -> None:
        logging.info("STT_PROFILE: %s", self.stt_plan.profile)
        logging.info("VAD_MODE: %s", "silero" if self.vad_gate is not None else "rms_fallback")
        logging.info("STT_ENDPOINT_MODE: vosk_small")
        logging.info("STT_FINAL_MODE: %s", self.final_mode)
        if self.final_fallback_mode:
            logging.info("STT_FINAL_FALLBACK: %s", self.final_fallback_mode)

    def _validate_wake_settings(self) -> None:
        if self.wake_min_hits < 1:
            raise ValueError("wake_min_hits must be at least 1")
        if self.wake_hit_window_sec <= 0:
            raise ValueError("wake_hit_window_sec must be greater than 0")
        if self.wake_cooldown_sec < 0:
            raise ValueError("wake_cooldown_sec must not be negative")

    def _load_vosk_model(self) -> None:
        logging.info("Loading Vosk endpoint model from: %s", self.model_path)

        # Vosk writes native C++ diagnostics directly to stderr. Its supported
        # log-level hook is the least invasive way to keep clean mode readable.
        if CASE_CONSOLE_MODE == "clean" and VoskSetLogLevel is not None:
            VoskSetLogLevel(-1)

        if not self.model_path.is_dir():
            raise FileNotFoundError(
                "Vosk model folder is missing: "
                f"{self.model_path}\n"
                "Place the Vosk model at ai/stt/vosk-model-small-en-us-0.15 "
                "or pass model_path=... to STTEngine."
            )

        self.model = VoskModel(str(self.model_path))

    def _ensure_final_vosk_model(self) -> Optional[VoskModel]:
        if "vosk_lgraph" not in self.stt_plan.final_chain:
            return None
        if self.final_vosk_model is not None:
            return self.final_vosk_model
        path = self.final_vosk_model_path or self.vosk_lgraph_model_path
        if not path.is_dir():
            raise FileNotFoundError(f"Vosk lgraph model folder is missing: {path}")
        logging.info("Loading Vosk final model from: %s", path)
        self.final_vosk_model = VoskModel(str(path))
        return self.final_vosk_model

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
        reset_transitions = {
            (STATE_WAKE_ACK, STATE_LISTEN_COMMAND),
            (STATE_SPEAKING, STATE_SHORT_FOLLOW_UP),
            (STATE_SHORT_FOLLOW_UP, STATE_IDLE),
            (STATE_LISTEN_COMMAND, STATE_IDLE),
            (STATE_THINKING, STATE_SPEAKING),
        }
        if WAKE_CLEAR_RING_BUFFER_ON_STATE_CHANGE and (
            old_state,
            new_state,
        ) in reset_transitions:
            self._reset_wake_detector(
                f"state_transition from={old_state} to={new_state}"
            )
        if old_state == STATE_SHORT_FOLLOW_UP and new_state == STATE_IDLE:
            self._start_wake_guard(
                WAKE_POST_FOLLOWUP_TIMEOUT_COOLDOWN_SEC,
                "post_followup_timeout_cooldown",
            )

    def _current_state(self) -> str:
        with self._state_lock:
            return self._state

    def set_realtime_session_runner(
        self,
        runner: Optional[Callable[[float], Awaitable[bool]]],
    ) -> None:
        """Install the optional cloud audio session entered after local wake."""
        self._realtime_session_runner = runner
        logging.info("STT realtime session runner enabled=%s", runner is not None)

    def set_external_state(self, state: str) -> None:
        """Allow the realtime engine to share CASE's authoritative state."""
        self._transition(state)

    def _run_realtime_session(self, wake_detected_at: float) -> bool:
        if self._realtime_session_runner is None or self.loop is None:
            return False
        logging.info("REALTIME: pausing classic Vosk conversation processing")
        future = asyncio.run_coroutine_threadsafe(
            self._realtime_session_runner(wake_detected_at),
            self.loop,
        )
        try:
            return bool(future.result())
        except concurrent.futures.CancelledError:
            logging.info("REALTIME: session cancelled during shutdown")
            return True
        except Exception as exc:
            if self._stop_event.is_set() or self.loop.is_closed():
                logging.info("REALTIME: session ended during shutdown")
                return True
            logging.warning("REALTIME: session handoff failed: %s", exc)
            return False

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
                    if self.mute_during_tts
                    else 0.0
                )

        if active_count == 0:
            self._last_tts_end_time = time.monotonic()
            self._start_wake_guard(WAKE_POST_TTS_COOLDOWN_SEC, "post_tts_cooldown")
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
                or self._state == STATE_WAKE_ACK
                or self._state == STATE_SPEAKING
                or (
                    self._state == STATE_SHORT_FOLLOW_UP
                    and not self._listening_for_transcript
                )
                or (self.mute_during_tts and self._tts_active_count > 0)
                or (self.mute_during_tts and now < self._post_tts_guard_until)
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
        words = re.findall(
            r"\w+(?:['’]\w+)?",
            " ".join(text.strip().lower().split()),
            flags=re.UNICODE,
        )
        return " ".join(word.replace("’", "'") for word in words)

    def _is_end_session_command(self, text: str) -> bool:
        cleaned = self._normalize_transcript(text)
        return any(phrase in cleaned for phrase in END_SESSION_PHRASES)

    def _is_long_conversation_trigger(self, text: str) -> bool:
        cleaned = self._normalize_transcript(text)
        return any(phrase in cleaned for phrase in LONG_CONVERSATION_TRIGGERS)

    @staticmethod
    def _is_punctuation_only(text: str) -> bool:
        stripped = str(text).strip()
        return bool(stripped) and not any(ch.isalnum() for ch in stripped)

    @staticmethod
    def _is_mostly_cjk(text: str) -> bool:
        stripped = [ch for ch in str(text).strip() if not ch.isspace()]
        if not stripped:
            return False
        cjk_count = sum(
            1
            for ch in stripped
            if "\u4e00" <= ch <= "\u9fff"
            or "\u3040" <= ch <= "\u30ff"
            or "\uac00" <= ch <= "\ud7af"
        )
        return cjk_count > 0 and cjk_count / len(stripped) >= 0.5

    def _recent_context_is_joke(self) -> bool:
        context = self._normalize_transcript(self._last_published_transcript)
        return any(word in context for word in JOKE_CONTEXT_WORDS)

    def _classify_followup(self, cleaned: str) -> str:
        normalized = self._strip_followup_fillers(cleaned)
        stripped_words = re.findall(r"\w+", normalized, flags=re.UNICODE)
        if not stripped_words:
            return "unclear_noise"
        if normalized in WAKE_ONLY_TRANSCRIPTS:
            return "unclear_noise"
        if normalized in FOLLOWUP_MORE_REQUESTS:
            return "followup_more_request"
        if normalized in FOLLOWUP_MEANINGFUL_PHRASES:
            return "followup_feedback"
        if any(normalized.startswith(starter) for starter in FOLLOWUP_CORRECTION_STARTERS):
            return "followup_correction"
        if any(normalized.startswith(pattern) for pattern in FOLLOWUP_ACCEPT_PATTERNS):
            return "clear_followup_question"
        if stripped_words[0] in FOLLOWUP_COMMAND_STARTERS:
            if stripped_words[0] in {"tell", "show", "give", "make", "continue", "explain"}:
                return "clear_followup_command"
            return "clear_followup_question"
        if any(word in normalized for word in {"joke", "roast", "picture", "camera", "vision", "funny"}):
            return "clear_followup_command"
        return "unclear_noise"

    def _has_clear_followup_intent(self, cleaned: str) -> bool:
        return self._classify_followup(cleaned) != "unclear_noise"

    def _initial_fragment_unclear(self, cleaned: str) -> bool:
        if cleaned in INITIAL_CLEAR_SHORT_COMMANDS:
            return False
        if cleaned in INITIAL_FRAGMENT_PHRASES:
            return True
        words = re.findall(r"\w+", cleaned, flags=re.UNICODE)
        if len(words) <= 3:
            if any(cleaned.startswith(pattern.strip()) for pattern in FOLLOWUP_ACCEPT_PATTERNS):
                return False
            if words and words[0] in FOLLOWUP_COMMAND_STARTERS:
                return False
            if any(word in cleaned for word in {"joke", "roast", "picture", "camera", "vision", "funny"}):
                return False
            return True
        return False

    @staticmethod
    def _strip_followup_fillers(cleaned: str) -> str:
        words = cleaned.split()
        while words and words[0] in FOLLOWUP_FILLERS:
            words.pop(0)
        return " ".join(words)

    def _transcript_reject_reason(
        self,
        text: str,
        *,
        followup: bool = False,
    ) -> Optional[str]:
        if self._is_punctuation_only(text):
            return "punctuation_only"
        if self._is_mostly_cjk(text):
            return "non_english_garbage"
        cleaned = self._normalize_transcript(text)

        if not cleaned:
            return "empty"
        if self._is_end_session_command(cleaned):
            return None
        if CONVERSATION_MODE_ENABLED and self._is_long_conversation_trigger(cleaned):
            return None
        if cleaned in INITIAL_CLEAR_SHORT_COMMANDS:
            return None
        word_count = len(re.findall(r"\w+", cleaned, flags=re.UNICODE))
        if word_count < MIN_TRANSCRIPT_WORDS and len(cleaned) > 1:
            return "too_few_words"
        if HYBRID_REJECT_TOO_SHORT and len(cleaned) < MIN_TRANSCRIPT_CHARS:
            return "too_short"
        if cleaned in FILLER_TRANSCRIPTS:
            return "filler"
        if cleaned in GARBAGE_TRANSCRIPTS:
            return "garbage"
        malformed_reason = malformed_transcript_reason(cleaned)
        if malformed_reason:
            return malformed_reason
        if HYBRID_REJECT_WAKEWORD_ONLY and cleaned in WAKE_ONLY_TRANSCRIPTS:
            return "wake_word_only"
        if cleaned == self._last_published_transcript:
            return "duplicate"

        words = re.findall(r"\w+", cleaned, flags=re.UNICODE)
        unique_words = set(words)
        if len(words) >= 3 and len(unique_words) <= 1:
            return "repeated_filler"

        if followup:
            if words and all(word in GARBAGE_WORDS for word in words):
                return "followup_garbage"
            followup_category = self._classify_followup(cleaned)
            if followup_category == "unclear_noise":
                return "followup_unclear"
            logging.info(
                "TRANSCRIPT_ACCEPT: reason=%s text=%r",
                followup_category,
                text,
            )
            if REQUIRE_DIRECTED_SPEECH_IN_FOLLOWUP and "case" not in cleaned:
                return "not_directed"
            return None

        if self._initial_fragment_unclear(cleaned):
            return "fragment_unclear"

        if word_count < MIN_TRANSCRIPT_WORDS:
            return "too_few_words"

        alpha_count = sum(ch.isalpha() for ch in cleaned)
        if alpha_count < MIN_TRANSCRIPT_CHARS:
            return "not_enough_letters"

        return None

    def _repair_transcript(self, text: str) -> str:
        repaired, reason = repair_common_transcript(
            text,
            recent_context=self._last_published_transcript,
        )
        if reason:
            if reason in {
                "embedded_known_command",
                "phonetic_followup_repair",
                "banter_phonetic_repair",
            }:
                logging.info(
                    "FOLLOWUP_REPAIR: original=%r repaired=%r reason=%s",
                    text,
                    repaired,
                    reason,
                )
                return repaired
            logging.info(
                "TRANSCRIPT_REPAIR: before=%r after=%r reason=%s",
                text,
                repaired,
                reason,
            )
            return repaired
        return repaired

    def _publish_user_spoke(self, text: str, *, followup: bool = False) -> bool:
        text = self._repair_transcript(text)
        before_dedupe = text.strip()
        text = dedupe_repeated_transcript(before_dedupe)
        if text != before_dedupe:
            logging.info(
                "TRANSCRIPT_DEDUPE: before=%r after=%r",
                before_dedupe,
                text,
            )
        reject_reason = self._transcript_reject_reason(text, followup=followup)

        if reject_reason:
            logging.info("TRANSCRIPT_REJECT: reason=%s text=%r", reject_reason, text)
            self._guard_after_ignored_transcript()
            return False

        if self.loop and not self.loop.is_closed():
            turn_metrics = {
                **self._session_turn_metrics,
                **self._last_transcript_metrics,
                "transcript_final_at": time.monotonic(),
            }
            metrics_coroutine = self.bus.publish("TURN_METRICS", turn_metrics)
            coroutine = self.bus.publish("USER_SPOKE", text)
            try:
                asyncio.run_coroutine_threadsafe(metrics_coroutine, self.loop)
                asyncio.run_coroutine_threadsafe(coroutine, self.loop)
            except RuntimeError:
                metrics_coroutine.close()
                coroutine.close()

        self._last_published_transcript = self._normalize_transcript(text)
        logging.info("Published USER_SPOKE: %s", text)
        if CASE_CONSOLE_MODE == "clean":
            console.you(text)
        else:
            print(f"\033[92m[You]: {text}\033[0m")
        return True

    def _publish_ai_speak_from_thread(self, text: str):
        if not self.loop or self.loop.is_closed():
            return None

        coroutine = self.bus.publish("AI_SPEAK", text)
        try:
            return asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        except RuntimeError:
            coroutine.close()
            return None

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

    def _reset_wake_detector(self, reason: str) -> None:
        self._reset_wake_history()
        self._drain_audio_queue()
        logging.info("WAKE_DETECTOR_RESET: reason=%s", reason)

    def _start_wake_guard(self, seconds: float, reason: str) -> None:
        seconds = max(0.0, float(seconds))
        if seconds <= 0:
            return
        self._wake_suppressed_until = max(
            self._wake_suppressed_until,
            time.monotonic() + seconds,
        )
        self._wake_suppression_reason = reason
        self._reset_wake_detector(reason)
        if reason == "cooldown_after_ignored_transcript":
            logging.info("WAKE_GUARD: cooldown_after_ignored_transcript sec=%.1f", seconds)

    def _guard_after_ignored_transcript(self) -> None:
        self._start_wake_guard(
            WAKE_COOLDOWN_AFTER_IGNORED_TRANSCRIPT_SEC,
            "cooldown_after_ignored_transcript",
        )

    def _wake_guard_reject_reason(self, now: float) -> tuple[str, dict[str, float]]:
        state = self._current_state()
        if WAKE_DISABLE_DURING_TTS and self._tts_active():
            return "tts_active", {}
        if state == STATE_SPEAKING:
            return "speaking_state", {}
        if state == STATE_SHORT_FOLLOW_UP:
            return "short_followup_state", {}
        if now < self._wake_suppressed_until:
            reason = self._wake_suppression_reason or "wake_guard_cooldown"
            details: dict[str, float] = {
                "cooldown_remaining": self._wake_suppressed_until - now,
            }
            if reason == "post_tts_cooldown":
                details["seconds_since_tts_end"] = now - self._last_tts_end_time
            return reason, details
        return "", {}

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
        guard_reason, guard_details = self._wake_guard_reject_reason(now)
        if guard_reason:
            if score >= self.wake_threshold:
                detail_lines = ""
                if "seconds_since_tts_end" in guard_details:
                    detail_lines += (
                        "\n  seconds_since_tts_end="
                        f"{guard_details['seconds_since_tts_end']:.3f}"
                    )
                if "cooldown_remaining" in guard_details:
                    detail_lines += (
                        "\n  cooldown_remaining="
                        f"{guard_details['cooldown_remaining']:.3f}"
                    )
                logging.info(
                    "WAKE_CONFIRM_REJECT:\n"
                    "  current_score_raw=%.6f\n"
                    "  reason=%s%s",
                    score,
                    guard_reason,
                    detail_lines,
                )
            self._reset_wake_history()
            return False, score

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
        state_gate_ok = self._current_state() == STATE_IDLE
        current_score_strong_ok = score >= self.wake_strong_threshold
        window_max_strong_ok = window_max >= self.wake_strong_threshold
        confirmed = (
            score >= self.wake_threshold
            and hit_count >= self.wake_min_hits
            and window_max_strong_ok
            and cooldown_ready
            and state_gate_ok
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
            logging.info(
                "WAKE_CONFIRM_ACCEPT: key=%s hit_count=%s window_max=%.6f "
                "reason=hit_window_and_strong_window_max",
                self.wakeword_name,
                hit_count,
                window_max,
            )
            self._last_wake_time = now
            self._reset_wake_history()
        elif hit_count >= self.wake_min_hits:
            reasons = []
            if score < self.wake_threshold:
                reasons.append("current_score_below_threshold")
            if not window_max_strong_ok:
                reasons.append("window_max_below_strong_threshold")
            if not cooldown_ready:
                reasons.append("cooldown_active")
            if not state_gate_ok:
                reasons.append("state_gate_blocked")
            logging.info(
                "WAKE_CONFIRM_REJECT:\n"
                "  current_score_raw=%.6f\n"
                "  window_max_raw=%.6f\n"
                "  threshold=%.6f\n"
                "  strong_threshold=%.6f\n"
                "  hit_count=%s\n"
                "  min_hits=%s\n"
                "  hit_window_sec=%.3f\n"
                "  cooldown_active=%s\n"
                "  state_gate_ok=%s\n"
                "  current_score_strong_ok=%s\n"
                "  window_max_strong_ok=%s\n"
                "  reason=%s",
                score,
                window_max,
                self.wake_threshold,
                self.wake_strong_threshold,
                hit_count,
                self.wake_min_hits,
                self.wake_hit_window_sec,
                not cooldown_ready,
                state_gate_ok,
                current_score_strong_ok,
                window_max_strong_ok,
                ",".join(reasons) or "unknown",
            )

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

    def _transcribe_final_vosk(self, waveform: np.ndarray) -> str:
        model = self._ensure_final_vosk_model()
        if model is None or waveform.size == 0:
            return ""
        recognizer = KaldiRecognizer(model, self.samplerate)
        recognizer.AcceptWaveform(np.asarray(waveform, dtype="<i2").tobytes())
        return self._parse_vosk_text(recognizer, final=True)

    def _is_clear_fast_intent(self, text: str) -> bool:
        cleaned = self._normalize_transcript(text)
        if not cleaned:
            return False
        return any(
            re.search(pattern, cleaned, flags=re.IGNORECASE)
            for pattern in FAST_INTENT_PATTERNS
        )

    def _transcribe_lgraph_timeboxed(
        self,
        waveform: np.ndarray,
        *,
        fallback_candidate: str,
    ) -> tuple[str, str | None]:
        timeout = max(0.0, float(self.lgraph_final_timeout_sec))
        logging.info(
            "STT_FINAL_TIMEBOX: backend=vosk_lgraph timeout=%.2fs",
            timeout,
        )
        future = self._final_stt_executor.submit(self._transcribe_final_vosk, waveform)
        try:
            text = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            logging.warning(
                "STT_FINAL_TIMEOUT: backend=vosk_lgraph fallback=vosk_small "
                "candidate=%r",
                fallback_candidate,
            )
            return "", "lgraph_timeout"
        except Exception as exc:
            logging.warning(
                "Vosk lgraph final decode failed; using fallback: %s",
                exc,
            )
            return "", str(exc)
        if is_usable_transcript(text):
            return text, None
        return text, "lgraph_empty"

    def _select_final_transcript_for_utterance(
        self,
        *,
        vosk_candidate: str,
        sense_text: str,
        waveform: np.ndarray,
        backend_status: dict,
    ) -> tuple[str, str]:
        if is_usable_transcript(vosk_candidate) and self._is_clear_fast_intent(
            vosk_candidate
        ):
            backend_status["selected_source"] = "vosk_small"
            return dedupe_repeated_transcript(vosk_candidate), "clear_fast_intent"

        lgraph_text = ""
        reason = "profile_chain"
        needs_lgraph_candidate = (
            self.final_mode == "vosk_lgraph"
            or bool(backend_status.get("sensevoice_error"))
            or not is_usable_transcript(sense_text)
        )
        if (
            "vosk_lgraph" in self.stt_plan.final_chain
            and self.vosk_lgraph_model_path.is_dir()
            and waveform.size
            and needs_lgraph_candidate
        ):
            lgraph_text, lgraph_error = self._transcribe_lgraph_timeboxed(
                waveform,
                fallback_candidate=vosk_candidate,
            )
            if lgraph_error:
                backend_status["vosk_lgraph_error"] = lgraph_error
                if lgraph_error == "lgraph_timeout":
                    reason = "lgraph_timeout"
            elif is_usable_transcript(lgraph_text):
                reason = "final_ready_within_timeout"
                logging.info("Vosk lgraph final candidate: %r", lgraph_text)

        selected = choose_final_transcript(
            vosk_candidate,
            sense_text,
            backend_status,
            lgraph_candidate=lgraph_text,
        )
        if reason == "profile_chain" and backend_status.get("selected_source") == "vosk_small":
            reason = "fallback"
        return selected, reason

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
        utterance_frames: list[np.ndarray] = []
        pre_roll_frames: deque[bytes] = deque(
            maxlen=max(
                1,
                int(
                    round(
                        STT_PREROLL_MS
                        / (WAKE_FRAME_SECONDS * 1000.0)
                    )
                ),
            )
        )
        semantic_held_candidate = ""
        denoiser_frames = 0
        denoiser_seconds = 0.0

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
            nonlocal semantic_held_candidate
            recognizer = KaldiRecognizer(self.model, self.samplerate)
            transcript_parts.clear()
            utterance_frames.clear()
            pre_roll_frames.clear()
            if self.vad_gate is not None:
                self.vad_gate.reset()
            speech_started_at = None
            last_speech_at = None
            active_speech_duration = 0.0
            confirm_started_at = None
            forced_final = False
            last_partial_text = ""
            semantic_held_candidate = ""
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

            vosk_final = self._parse_vosk_text(recognizer, final=True)
            vosk_candidate = self._join_transcript_parts(
                [*transcript_parts, vosk_final]
            )
            sense_text = ""
            waveform = (
                np.concatenate(utterance_frames)
                if utterance_frames
                else np.empty(0, dtype=np.int16)
            )
            backend_status = {
                "sensevoice_available": self.sensevoice is not None,
                "sensevoice_error": None,
                "vosk_lgraph_available": self.vosk_lgraph_model_path.is_dir(),
                "vosk_lgraph_error": None,
                "final_chain": self.stt_plan.final_chain,
            }
            if self.sensevoice is not None and waveform.size:
                try:
                    sent_duration = len(waveform) / float(self.samplerate)
                    captured_duration = max(0.0, (last_speech_at or now) - (speech_started_at or now))
                    logging.info(
                        "STT_AUDIO_BUFFER:\n"
                        "  preroll_ms=%s\n"
                        "  captured_duration=%.3f\n"
                        "  sent_to_sensevoice_duration=%.3f",
                        int(STT_PREROLL_MS),
                        captured_duration,
                        sent_duration,
                    )
                    sense_text = self.sensevoice.transcribe(waveform, self.samplerate)
                    logging.info("SenseVoice final candidate: %r", sense_text)
                except Exception as exc:
                    logging.warning("SenseVoice decode failed; using Vosk result: %s", exc)
                    backend_status["sensevoice_error"] = str(exc)
            selected, select_reason = self._select_final_transcript_for_utterance(
                vosk_candidate=vosk_candidate,
                sense_text=sense_text,
                waveform=waveform,
                backend_status=backend_status,
            )
            transcript_parts.clear()
            append_transcript_part(selected)
            selected_raw = {
                "sensevoice": sense_text,
                "vosk_lgraph": selected,
                "vosk_small": vosk_candidate,
                "vosk_fallback": vosk_candidate,
            }.get(backend_status["selected_source"], "")
            if selected and selected != normalize_transcript(selected_raw):
                logging.info(
                    "TRANSCRIPT_DEDUPE: before=%r after=%r",
                    selected_raw,
                    selected,
                )
            logging.info(
                "TRANSCRIPT_SELECT: source=%s reason=%s text=%r",
                backend_status["selected_source"],
                select_reason,
                selected,
            )
            recognizer = KaldiRecognizer(self.model, self.samplerate)
            confirm_started_at = now
            candidate = self._join_transcript_parts(transcript_parts)
            log_stt_state(
                STT_STATE_FINAL_CONFIRM,
                final_candidate=repr(candidate),
                confirm_delay=f"{self.final_confirm_delay_sec:.2f}",
                reopen_window=f"{self.reopen_after_final_sec:.2f}",
                reopened_after_final=False,
            )

        log_stt_state(STT_STATE_LISTENING)

        logging.info("Listening for user prompt (%s, timeout %.1fs)...", label, listen_timeout)
        self._listening_for_transcript = True

        while not self._stop_event.is_set():
            now = time.monotonic()
            if speech_started_at is None and now >= speech_wait_deadline:
                break

            if (
                speech_started_at is not None
                and now - speech_started_at >= self.max_command_listen_sec
                and confirm_started_at is None
                and not forced_final
            ):
                logging.info(
                    "STT maximum command duration reached: %.2fs",
                    self.max_command_listen_sec,
                )
                forced_final = True
                start_final_confirm(now)

            if not self._is_enabled():
                logging.info("STT disabled during %s listening; aborting.", label)
                self._listening_for_transcript = False
                return None

            if self._tts_active():
                logging.info("TTS started during %s listening; aborting.", label)
                self._listening_for_transcript = False
                return None

            try:
                frame = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                frame = None

            if frame is not None and self._should_ignore_audio():
                continue

            if frame is not None and self.denoiser is not None:
                try:
                    frame, denoise_seconds = self.denoiser.process(frame)
                    denoiser_frames += 1
                    denoiser_seconds += denoise_seconds
                    if denoiser_frames % 50 == 0:
                        logging.info(
                            "GTCRN: avg latency %.1fms/frame",
                            denoiser_seconds / denoiser_frames * 1000.0,
                        )
                except Exception as exc:
                    logging.warning("GTCRN: processing failed; disabling: %s", exc)
                    self.denoiser = None

            now = time.monotonic()
            rms = self._frame_rms(frame) if frame is not None else 0.0
            vad_speech = self.vad_gate.is_speech(frame) if (
                frame is not None and self.vad_gate is not None
            ) else None
            speech_present = (
                bool(vad_speech)
                if vad_speech is not None
                else frame is not None and rms >= SPEECH_RMS_THRESHOLD
            )
            if frame is not None and speech_started_at is None:
                pre_roll_frames.append(frame)

            if speech_present:
                started_now = speech_started_at is None
                if speech_started_at is None:
                    speech_started_at = now
                    logging.info("%s speech started, rms=%.1f", label, rms)
                    buffered = list(pre_roll_frames)
                    for buffered_frame in buffered:
                        utterance_frames.append(
                            np.frombuffer(buffered_frame, dtype="<i2").copy()
                        )
                        try:
                            if recognizer.AcceptWaveform(buffered_frame):
                                append_transcript_part(
                                    self._parse_vosk_text(recognizer, final=False)
                                )
                        except Exception as exc:
                            logging.debug("Vosk pre-roll feed failed: %s", exc)
                    pre_roll_frames.clear()
                elif (
                    not forced_final
                    and not self.disable_reopen_after_final
                    and endpoint_state in {
                        STT_STATE_POSSIBLE_END,
                        STT_STATE_FINAL_CONFIRM,
                    }
                ):
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
            else:
                started_now = False

            if speech_started_at is None:
                continue

            if frame is not None:
                if not started_now:
                    utterance_frames.append(
                        np.frombuffer(frame, dtype="<i2").copy()
                    )
                try:
                    if not started_now and recognizer.AcceptWaveform(frame):
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
                and self.accept_final_on_silence
                and silence_duration >= self.speech_end_silence_sec
            ):
                start_final_confirm(now)

            if confirm_started_at is None:
                continue

            confirmation_window = max(
                self.final_confirm_delay_sec,
                0.0 if self.disable_reopen_after_final else self.reopen_after_final_sec,
            )
            if now - confirm_started_at < confirmation_window:
                continue

            candidate = self._join_transcript_parts(transcript_parts)
            waveform = (
                np.concatenate(utterance_frames)
                if utterance_frames
                else np.empty(0, dtype=np.int16)
            )
            hold_reason = ""
            if HYBRID_WEAK_ENDING_ENABLED and has_weak_ending(candidate):
                hold_reason = "weak_ending"
            elif (
                self.smart_turn is not None
                and len(waveform) / self.samplerate >= SMART_TURN_MIN_AUDIO_SEC
            ):
                try:
                    complete, probability = self.smart_turn.is_complete(waveform)
                    logging.info(
                        "SMART_TURN: complete=%s probability=%.3f threshold=%.3f",
                        complete,
                        probability,
                        SMART_TURN_THRESHOLD,
                    )
                    if not complete:
                        hold_reason = "smart_turn_incomplete"
                    elif (
                        SMART_TURN_ACCEPT_IMMEDIATELY_IF_COMPLETE
                        and probability >= SMART_TURN_FAST_ACCEPT_PROBABILITY
                    ):
                        logging.info(
                            "SMART_TURN: complete=True probability=%.3f action=accept_now",
                            probability,
                        )
                except Exception as exc:
                    logging.warning(
                        "SMART_TURN: inference failed; disabling semantic model: %s",
                        exc,
                    )
                    self.smart_turn = None

            if hold_reason:
                if candidate == semantic_held_candidate:
                    logging.info(
                        "TRANSCRIPT ignored: reason=incomplete_after_hold text=%r",
                        candidate,
                    )
                    self._guard_after_ignored_transcript()
                    if followup:
                        self._listening_for_transcript = False
                        return None
                    reset_after_rejected_candidate()
                    continue
                semantic_held_candidate = candidate
                hold_seconds = max(
                    HYBRID_HOLD_WEAK_ENDING_SEC,
                    SMART_TURN_HOLD_SEC,
                )
                confirm_started_at = None
                last_speech_at = now + hold_seconds - self.speech_end_silence_sec
                logging.info(
                    "STT endpoint held reason=%s text=%r hold=%.2fs",
                    hold_reason,
                    candidate,
                    hold_seconds,
                )
                console.system("holding incomplete phrase")
                continue

            candidate = self._repair_transcript(candidate)
            transcript_parts.clear()
            append_transcript_part(candidate)
            if active_speech_duration < self.min_utterance_sec:
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
                self._guard_after_ignored_transcript()
                if followup:
                    self._listening_for_transcript = False
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
            self._last_transcript_metrics = {
                "speech_started_at": speech_started_at,
                "last_speech_at": last_speech_at or now,
                "transcript_final_at": time.monotonic(),
            }
            self._listening_for_transcript = False
            return candidate

        logging.info("TRANSCRIPT ignored: reason=session_timeout label=%s", label)
        self._guard_after_ignored_transcript()
        self._listening_for_transcript = False
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
                "I'm here when you need me.",
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

    def _run_conversation_session(self, wake_detected_at: float | None = None) -> None:
        session_start = time.monotonic()
        long_mode = False
        self._session_turn_metrics = {}
        self._last_transcript_metrics = {}
        if wake_detected_at is not None:
            self._session_turn_metrics["wake_detected_at"] = wake_detected_at

        self._transition(STATE_WAKE_ACK)
        ack = choose_wake_ack()
        self._session_turn_metrics["wake_ack_start_at"] = time.monotonic()
        wake_ack_mode = get_str(
            "WAKE_ACK_MODE", case_defaults.WAKE_ACK_MODE
        ).lower()
        use_asset = (
            not get_bool(
                "WAKE_ACK_USE_VOICE_BACKEND",
                case_defaults.WAKE_ACK_USE_VOICE_BACKEND,
            )
            and wake_ack_mode in {"recorded_wav", "cached_wav"}
        )
        played_asset = use_asset and play_wake_ack_asset(
            ack,
            mode=wake_ack_mode,
        )
        if wake_ack_mode == "beep":
            played_asset = play_wake_ack_beep()
        if not played_asset and wake_ack_mode != "none":
            played_tts = self._say_and_wait(
                ack,
                start_timeout=WAKE_TTS_START_TIMEOUT_SEC,
            )
            if not played_tts:
                play_wake_ack_beep()
        self._session_turn_metrics["wake_ack_done_at"] = time.monotonic()
        self._drain_audio_queue()
        self._start_wake_guard(
            WAKE_POST_WAKE_ACK_COOLDOWN_SEC,
            "post_wake_ack_cooldown",
        )

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

            listen_timeout = (
                LONG_FOLLOWUP_TIMEOUT_SEC if long_mode else self.followup_timeout_sec
            )
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
                    wake_detected_at = time.monotonic()
                    realtime_ok = False
                    if self._realtime_session_runner is not None:
                        self._transition(STATE_WAKE_ACK)
                        realtime_ok = self._run_realtime_session(wake_detected_at)
                    if not realtime_ok:
                        if self._stop_event.is_set():
                            break
                        if self._realtime_session_runner is not None:
                            logging.warning(
                                "REALTIME: unavailable; falling back to classic voice"
                            )
                        self._run_conversation_session(wake_detected_at)
                    else:
                        self._transition(STATE_IDLE)
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
                device_info = sd.query_devices(self.input_device, "input")
                default_rate = int(round(device_info.get("default_samplerate") or 0))
                device_name = device_info.get("name", "default input")
                max_channels = int(device_info.get("max_input_channels") or 1)
            except Exception as exc:
                if self.input_device is not None:
                    raise RuntimeError(
                        "configured microphone is unavailable: "
                        f"{self.input_device!r}: {exc}"
                    ) from exc
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
                "Starting mic stream selector=%r resolved='%s' at %s Hz, "
                "%s channel(s); "
                "resampling to %s Hz for wake/Vosk.",
                self.input_device,
                device_name,
                stream_rate,
                self._stream_channels,
                self.samplerate,
            )

            try:
                stream = sd.InputStream(
                    device=self.input_device,
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
                    device=self.input_device,
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
                await asyncio.to_thread(self._processing_thread.join, 0.5)

            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass

            self._final_stt_executor.shutdown(wait=False, cancel_futures=True)

            logging.info("STT shutdown complete.")
            console.system("STT stopped")
