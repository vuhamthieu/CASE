"""Central turn-state and latency helpers for CASE realtime voice.

This module is deliberately small for now: it provides one owner for the
canonical turn states and one formatter for per-turn latency diagnostics. The
existing STT, LLM, and TTS components can report timestamps into the same
metrics dict without taking hard dependencies on each other.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


logger = logging.getLogger(__name__)


class TurnState(str, Enum):
    IDLE = "IDLE"
    WAKE_ACK = "WAKE_ACK"
    LISTEN_COMMAND = "LISTEN_COMMAND"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    SHORT_FOLLOW_UP = "SHORT_FOLLOW_UP"
    ECHO_GUARD = "ECHO_GUARD"


@dataclass
class TurnManager:
    """Track CASE turn state and emit consistent latency breakdowns."""

    state: TurnState = TurnState.IDLE
    metrics: dict[str, float] = field(default_factory=dict)

    def transition(self, new_state: TurnState | str) -> None:
        next_state = TurnState(new_state)
        if next_state == self.state:
            return
        previous = self.state
        self.state = next_state
        logger.info("TURN_STATE: %s -> %s", previous.value, next_state.value)

    def mark(self, name: str, value: float | None = None) -> float:
        timestamp = time.monotonic() if value is None else float(value)
        self.metrics[name] = timestamp
        return timestamp

    @staticmethod
    def _delta(
        metrics: Mapping[str, float],
        start: str,
        end: str,
    ) -> str:
        start_value = metrics.get(start)
        end_value = metrics.get(end)
        if start_value is None or end_value is None:
            return "n/a"
        return f"{end_value - start_value:.3f}s"

    @classmethod
    def latency_breakdown(cls, metrics: Mapping[str, float]) -> dict[str, str]:
        return {
            "wake_to_ack_start": cls._delta(
                metrics, "wake_detected_at", "wake_ack_start_at"
            ),
            "ack_playback": cls._delta(
                metrics, "wake_ack_start_at", "wake_ack_done_at"
            ),
            "user_speech_duration": cls._delta(
                metrics, "speech_started_at", "last_speech_at"
            ),
            "last_speech_to_transcript_final": cls._delta(
                metrics, "last_speech_at", "transcript_final_at"
            ),
            "transcript_final_to_llm_first_token": cls._delta(
                metrics, "transcript_final_at", "first_llm_chunk_at"
            ),
            "llm_first_token_to_tts_start": cls._delta(
                metrics, "first_llm_chunk_at", "first_tts_chunk_start_at"
            ),
            "tts_synth": cls._delta(
                metrics, "first_tts_chunk_start_at", "first_tts_chunk_done_at"
            ),
            "playback": cls._delta(
                metrics, "first_audio_play_start_at", "full_audio_done_at"
            ),
            "total_last_speech_to_first_audio": cls._delta(
                metrics, "last_speech_at", "first_audio_play_start_at"
            ),
            "total_wake_to_first_audio": cls._delta(
                metrics, "wake_detected_at", "first_audio_play_start_at"
            ),
        }

    @classmethod
    def log_latency(cls, metrics: Mapping[str, float]) -> None:
        breakdown = cls.latency_breakdown(metrics)
        logger.info(
            "TURN_LATENCY_BREAKDOWN:\n"
            "  wake_to_ack_start=%s\n"
            "  ack_playback=%s\n"
            "  user_speech_duration=%s\n"
            "  last_speech_to_transcript_final=%s\n"
            "  transcript_final_to_llm_first_token=%s\n"
            "  llm_first_token_to_tts_start=%s\n"
            "  tts_synth=%s\n"
            "  playback=%s\n"
            "  total_last_speech_to_first_audio=%s\n"
            "  total_wake_to_first_audio=%s",
            breakdown["wake_to_ack_start"],
            breakdown["ack_playback"],
            breakdown["user_speech_duration"],
            breakdown["last_speech_to_transcript_final"],
            breakdown["transcript_final_to_llm_first_token"],
            breakdown["llm_first_token_to_tts_start"],
            breakdown["tts_synth"],
            breakdown["playback"],
            breakdown["total_last_speech_to_first_audio"],
            breakdown["total_wake_to_first_audio"],
        )
