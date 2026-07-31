"""Deterministic local intent routing before CASE sends text to Gemini."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from src.vision.vision_engine import run_vision_once


logger = logging.getLogger(__name__)


class IntentType:
    VISION_SEE_ME = "VISION_SEE_ME"
    VISION_TAKE_PICTURE = "VISION_TAKE_PICTURE"
    CHAT = "CHAT"


@dataclass(frozen=True)
class LocalIntent:
    type: str
    transcript: str
    normalized_transcript: str


_TAKE_PICTURE_PATTERNS = (
    re.compile(r"\btake (?:a |my )?(?:picture|photo|snapshot)\b"),
    re.compile(r"\bcapture (?:an? )?image\b"),
)

_SEE_ME_PATTERNS = (
    re.compile(r"\bcan you (?:actually )?see me\b"),
    re.compile(r"\bdo you see me\b"),
    re.compile(r"\blook at me\b"),
    re.compile(r"\blook around\b"),
    re.compile(r"\buse (?:the )?(?:camera|vision)\b"),
    re.compile(r"\b(?:camera|vision) check\b"),
    re.compile(r"\bam i centered\b"),
    re.compile(r"\bam i in (?:the )?center\b"),
    re.compile(r"\bwhere am i\b"),
)


def _normalize_transcript(transcript: str) -> str:
    normalized = re.sub(r"[^a-z0-9 ]+", " ", transcript.lower())
    return " ".join(normalized.split())


class IntentRouter:
    """Route local camera commands and forward all other text to Gemini."""

    def __init__(
        self,
        message_bus: Any,
        *,
        vision_scheduler: Optional[Any] = None,
        vision_engine: Optional[Any] = None,
        vision_once: Optional[Any] = None,
        input_topic: str = "USER_SPOKE",
        chat_topic: str = "CHAT_USER_SPOKE",
    ) -> None:
        self.message_bus = message_bus
        self.vision_scheduler = vision_scheduler
        self.vision_engine = vision_engine
        self.vision_once = vision_once or run_vision_once
        self.chat_topic = chat_topic
        self._local_command_lock = asyncio.Lock()
        message_bus.subscribe(input_topic, self.handle_transcript)

    @staticmethod
    def classify(transcript: str) -> LocalIntent:
        text = transcript if isinstance(transcript, str) else ""
        normalized = _normalize_transcript(text)
        if any(pattern.search(normalized) for pattern in _TAKE_PICTURE_PATTERNS):
            intent_type = IntentType.VISION_TAKE_PICTURE
        elif any(pattern.search(normalized) for pattern in _SEE_ME_PATTERNS):
            intent_type = IntentType.VISION_SEE_ME
        else:
            intent_type = IntentType.CHAT
        return LocalIntent(intent_type, text, normalized)

    async def handle_transcript(self, transcript: Any) -> None:
        if not isinstance(transcript, str) or not transcript.strip():
            return
        intent = self.classify(transcript)
        if intent.type == IntentType.CHAT:
            await self.message_bus.publish(self.chat_topic, transcript)
            await asyncio.sleep(0)
            return

        logger.info(
            "INTENT_ROUTER: local intent=%s transcript=%r",
            intent.type,
            transcript,
        )
        async with self._local_command_lock:
            if intent.type == IntentType.VISION_SEE_ME:
                await self._handle_see_me()
            elif intent.type == IntentType.VISION_TAKE_PICTURE:
                await self._handle_take_picture()

    async def _handle_see_me(self) -> None:
        capture = await self.vision_once(
            IntentType.VISION_SEE_ME,
            mode="single_frame",
            message_bus=self.message_bus,
            allow_during_thinking=True,
        )
        status = "STABLE" if capture.ok and capture.faces else capture.status
        face = capture.faces[0] if capture.faces else {}
        direction = face.get("direction")
        replies = {
            "CENTER": "Yeah. You're centered.",
            "LEFT": "I see you on my left.",
            "RIGHT": "I see you on my right.",
        }
        if status == "STABLE" and direction in replies:
            reply = replies[direction]
        elif status == "ERROR":
            reply = "I tried, but the camera did not give me a clean frame."
        else:
            reply = "I don't have a clean visual lock on you."
        await self._local_reply(reply)

    async def _handle_take_picture(self) -> None:
        result = await self.vision_once(
            IntentType.VISION_TAKE_PICTURE,
            mode="snapshot",
            message_bus=self.message_bus,
            allow_during_thinking=True,
        )
        path = result.path if result.ok else None

        if path is None:
            reply = "I tried, but the camera capture failed."
        else:
            logger.info("VISION: saved requested snapshot path=%s", path)
            reply = "Done. I saved the snapshot."
        await self._local_reply(reply)

    async def _local_reply(self, text: str) -> None:
        logger.info("LOCAL_REPLY: %s", text)
        await self.message_bus.publish("AI_SPEAK", text)
        await asyncio.sleep(0)
