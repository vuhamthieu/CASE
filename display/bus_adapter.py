"""Passive listener bridging the AsyncMessageBus to the DisplayManager."""

from __future__ import annotations

import logging
from typing import Any

from .display_manager import DisplayManager

logger = logging.getLogger(__name__)


class DisplayBusAdapter:
    """Translate verified CASE bus topics into display state updates."""

    def __init__(self, bus: Any, display_manager: DisplayManager) -> None:
        self.bus = bus
        self.dm = display_manager

        self.bus.subscribe("USER_SPOKE", self._on_user_spoke)
        self.bus.subscribe("AI_SPEAK", self._on_ai_speak)
        self.bus.subscribe("TTS_START", self._on_tts_start)
        self.bus.subscribe("TTS_END", self._on_tts_end)

    @staticmethod
    def _extract_text(payload: Any) -> str:
        if isinstance(payload, dict):
            return str(payload.get("text", ""))
        if payload is None:
            return ""
        return str(payload)

    async def _on_user_spoke(self, payload: Any) -> None:
        text = self._extract_text(payload)
        if text:
            self.dm.append_message("YOU", text)
            self.dm.set_status("THINKING")

    async def _on_ai_speak(self, payload: Any) -> None:
        text = self._extract_text(payload)
        if text:
            self.dm.append_message("CASE", text)
            self.dm.set_status("SPEAKING")

    async def _on_tts_start(self, payload: Any) -> None:
        self.dm.set_status("SPEAKING")

    async def _on_tts_end(self, payload: Any) -> None:
        self.dm.set_status("IDLE")
