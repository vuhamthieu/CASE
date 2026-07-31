"""Provider-neutral hooks for future CASE expressive voice research."""

from .interfaces import (
    StreamingLLMProvider,
    StreamingSTTProvider,
    StreamingTTSProvider,
    VoicePipelineEvent,
)
from .hybrid_text_tts import HybridTextTTSPipeline
from .voice_backend import VoiceOutputBackend
from .transcript_backend import TranscriptInputBackend

__all__ = [
    "StreamingLLMProvider",
    "StreamingSTTProvider",
    "StreamingTTSProvider",
    "VoicePipelineEvent",
    "HybridTextTTSPipeline",
    "VoiceOutputBackend",
    "TranscriptInputBackend",
]
