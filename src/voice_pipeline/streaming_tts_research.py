"""Configuration boundary for a future expressive streaming TTS pipeline.

No provider SDK is imported here. Gemini Live native audio remains the runtime
fallback until a complete provider implementation is deliberately integrated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.config import defaults
from src.config.env import get_str


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExpressiveTTSResearchConfig:
    provider: str
    voice_id: str
    configured: bool


def load_research_config() -> ExpressiveTTSResearchConfig:
    provider = get_str(
        "EXPRESSIVE_TTS_PROVIDER", defaults.EXPRESSIVE_TTS_PROVIDER
    ).lower()
    if provider == "elevenlabs":
        key = get_str("ELEVENLABS_API_KEY", "")
        voice_id = get_str("ELEVENLABS_VOICE_ID", "")
    elif provider == "cartesia":
        key = get_str("CARTESIA_API_KEY", "")
        voice_id = get_str("CARTESIA_VOICE_ID", "")
    else:
        key = ""
        voice_id = ""
    return ExpressiveTTSResearchConfig(
        provider=provider,
        voice_id=voice_id,
        configured=bool(key and voice_id),
    )


def describe_research_status() -> str:
    config = load_research_config()
    if config.provider == "none":
        return "Expressive TTS research is disabled; Gemini Live native remains active."
    if config.provider not in {"elevenlabs", "cartesia"}:
        return f"Unsupported research provider: {config.provider}."
    if not config.configured:
        return (
            f"Research provider {config.provider} is missing its API key or voice ID; "
            "Gemini Live native remains active."
        )
    return (
        f"Research provider {config.provider} is configured, but production "
        "streaming integration is not enabled."
    )
