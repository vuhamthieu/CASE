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
        self._active_turn_id: int | None = None
        self._stream_buffer = ""

        self.bus.subscribe("USER_SPOKE", self._on_user_spoke)
        self.bus.subscribe("AI_SPEAK", self._on_ai_speak)
        self.bus.subscribe("AI_SPEAK_STREAM_START", self._on_stream_start)
        self.bus.subscribe("AI_SPEAK_STREAM_CHUNK", self._on_stream_chunk)
        self.bus.subscribe("AI_SPEAK_STREAM_END", self._on_stream_end)
        self.bus.subscribe("TTS_START", self._on_tts_start)
        self.bus.subscribe("TTS_END", self._on_tts_end)

    @staticmethod
    def _extract_text(payload: Any) -> str:
        if isinstance(payload, dict):
            return str(payload.get("text", ""))
        if payload is None:
            return ""
        return str(payload)

    @staticmethod
    def _extract_turn_id(payload: Any) -> int | None:
        if isinstance(payload, dict):
            turn_id = payload.get("turn_id")
            if isinstance(turn_id, int):
                return turn_id
        return None

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

    async def _on_stream_start(self, payload: Any) -> None:
        self._active_turn_id = self._extract_turn_id(payload)
        self._stream_buffer = ""
        self.dm.clear_stream()

    async def _on_stream_chunk(self, payload: Any) -> None:
        turn_id = self._extract_turn_id(payload)
        if self._active_turn_id is None:
            self._active_turn_id = turn_id
        elif turn_id is not None and turn_id != self._active_turn_id:
            self._stream_buffer = ""
            self._active_turn_id = turn_id

        text = self._extract_text(payload)
        if not text:
            return

        self._stream_buffer += text
        self.dm.update_stream(self._stream_buffer)

    async def _on_stream_end(self, payload: Any) -> None:
        turn_id = self._extract_turn_id(payload)
        if self._active_turn_id is not None and turn_id is not None and turn_id != self._active_turn_id:
            return
        self.dm.finish_stream()
        self._stream_buffer = ""
        self._active_turn_id = None

    async def _on_tts_start(self, payload: Any) -> None:
        self.dm.set_status("SPEAKING")

    async def _on_tts_end(self, payload: Any) -> None:
        self.dm.set_status("IDLE")
