"""Thin v1 adapter for CASE's existing text conversation pipeline.

The working implementation is the established local wake/Vosk input, Gemini
text personality, and CASEVoice streaming TTS path. This module names that
composition so its providers can be swapped later without duplicating runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .voice_backend import VoiceOutputBackend, create_voice_output_backend


@dataclass
class HybridTextTTSPipeline:
    output: VoiceOutputBackend
    streaming: bool = True

    @classmethod
    def from_runtime(
        cls,
        message_bus: Any,
        *,
        backend: str = "local_case_tts",
        streaming: bool = True,
    ) -> "HybridTextTTSPipeline":
        return cls(
            output=create_voice_output_backend(backend, message_bus),
            streaming=streaming,
        )
