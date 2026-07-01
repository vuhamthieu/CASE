import asyncio
import json
import logging
import os
import re
import time
from itertools import count
from typing import Iterator, Optional

from src.config import defaults
from src.config.env import get_str
from src.realtime.realtime_config import (
    CASE_REALTIME_ALLOW_LONG_ANSWER_WHEN_ASKED,
    CASE_REALTIME_DETAIL_MAX_CHARS,
    CASE_REALTIME_DETAIL_MAX_CHUNKS,
    CASE_REALTIME_MAX_CHARS_ROAST,
    CASE_REALTIME_MAX_LLM_WAIT_SEC,
    CASE_REALTIME_MAX_SENTENCES,
    CASE_REALTIME_REQUIRE_COMPLETE_SENTENCE,
    CASE_REALTIME_TARGET_SENTENCES,
    CASE_RESPONSE_MAX_TOTAL_CHARS,
    CASE_REALTIME_TTS_TEXT_DEADLINE_SEC,
    CASE_RESPONSE_MODE,
    CASE_STREAM_FULL_RESPONSE,
    CASE_LLM_FALLBACK_TO_FULL_ON_FIRST_TOKEN_TIMEOUT,
    CASE_LLM_FIRST_TOKEN_TIMEOUT_SEC,
    CASE_LLM_STREAM_TOTAL_TIMEOUT_SEC,
    CASE_TTS_ALLOW_MULTI_CHUNK,
    CASE_TTS_CHUNK_ABSOLUTE_MAX_CHARS,
    CASE_TTS_CHUNK_MODE,
    CASE_TTS_FIRST_CHUNK_MAX_CHARS,
    CASE_TTS_FIRST_CHUNK_TARGET_CHARS,
    CASE_TTS_FLUSH_FIRST_ON_SOFT_PUNCTUATION,
    CASE_TTS_CHUNK_MAX_CHARS,
    CASE_TTS_CHUNK_MIN_CHARS,
    CASE_TTS_CHUNK_PREFER_SENTENCE_BOUNDARY,
    CASE_TTS_MERGE_TINY_CHUNKS,
    CASE_TTS_NORMAL_CHUNK_MAX_CHARS,
    CASE_TTS_NORMAL_CHUNK_TARGET_CHARS,
    CASE_TTS_SINGLE_CHUNK_UNDER_CHARS,
    CASE_TTS_SMOOTH_CHUNKS,
    CASE_TTS_FIRST_CHUNK_FAST,
    CASE_TTS_MAX_SENTENCES_PER_CHUNK,
    CASE_TTS_MAX_CHARS_PER_CHUNK,
    CASE_TTS_MIN_CHARS_TO_GROUP,
    CASE_TTS_GROUP_SHORT_SENTENCES,
    CASE_TTS_EMOTION_ENABLED,
    CASE_TTS_DEFAULT_EMOTION,
    CASE_TTS_EMOTION_INTENSITY_DEFAULT,
    CASE_EMOTION_LLM_FALLBACK,
    CASE_EMOTION_LLM_PROVIDER,
    CASE_EMOTION_LLM_TIMEOUT_SEC,
    CASE_EMOTION_LLM_MIN_CONFIDENCE,
    CASE_EMOTION_MEMORY_ENABLED,
    CASE_EMOTION_MEMORY_TTL_TURNS,
    CASE_EMOTION_MEMORY_TTL_SEC,
    CASE_EMOTION_MEMORY_MIN_CONFIDENCE,
    CASE_EMOTION_MEMORY_DECAY,
    CASE_EMOTION_META_QUESTIONS_ENABLED,
    CASE_TTS_TINY_CHUNK_MAX_CHARS,
    CASE_TTS_DROP_OVERFLOW_IN_REALTIME,
    CASE_TTS_ENABLE_THINKING_FALLBACK,
    CASE_TTS_FALLBACK_SHORT_REPLY,
    CASE_TTS_FALLBACK_ONLY_ON_ERROR,
    CASE_TTS_MAX_WAIT_FOR_SENTENCE_SEC,
    CASE_TTS_MIN_SAFE_CHARS,
    CASE_TTS_REQUIRE_SAFE_BOUNDARY,
    CASE_TTS_MAX_CHUNKS_PER_TURN,
    CASE_TTS_REALTIME_MAX_CHUNKS,
    CASE_TTS_REALTIME_MAX_CHARS,
    CASE_TTS_REALTIME_TRUNCATE_EXTRA,
    CASE_TTS_REALTIME_WAIT_FOR_SENTENCE_END,
    CASE_ENABLE_THINKING_FILLER,
    CASE_THINKING_FILLER_AFTER_SEC,
    CASE_THINKING_FILLER_MAX_PER_TURN,
    CASE_THINKING_FILLER_SIMPLE_AFTER_SEC,
    CASE_HONESTY_PERCENT,
    CASE_HUMOR_PERCENT,
    CASE_SARCASM_LEVEL,
    CASE_STYLE_SHORT_REPLIES,
    CASE_VOICE_PRESET,
    CASE_VOICE_JOKE_MAX_SENTENCES,
    CASE_VOICE_REPLY_MAX_SENTENCES,
    CASE_VOICE_REPLY_STYLE,
)
from src.realtime.realtime_persona import build_case_system_instruction
from src.realtime.response_chunker import ResponseChunker as StreamingResponseChunker
from src.persona.emotion import (
    EmotionMemory,
    EmotionState,
    build_emotion_user_message,
    classify_emotion_with_llm,
    default_emotion_state,
    parse_leading_emotion_tag,
    select_emotion_with_memory,
)
from src.voice_pipeline.tts_safe_text import safe_tts_text

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


logger = logging.getLogger(__name__)

UNSAFE_STYLE_PATTERNS = (
    re.compile(r"\b" + "sui" + r"cide\b", re.IGNORECASE),
    re.compile(r"\bkill my" + r"self\b", re.IGNORECASE),
    re.compile(r"\bcommitted " + "sui" + r"cide\b", re.IGNORECASE),
    re.compile(r"\bself[- ]?" + "harm" + r"\b", re.IGNORECASE),
    re.compile(r"\bdepression joke\b", re.IGNORECASE),
    re.compile(
        r"\blogic board " + "nearly " + "committed " + "sui" + r"cide\b",
        re.IGNORECASE,
    ),
)
SAFE_JOKE_FALLBACK = "I told my CPU to relax; it opened Task Manager and blamed me."
SAFE_ROAST_FALLBACK = (
    "You gave a Raspberry Pi a personality, then complained it had opinions. "
    "Bold engineering."
)
SELF_DESCRIPTION_FALLBACK = (
    "I'm CASE. I handle voice, vision, and hardware control. Basically, a field "
    "robot with a patience module I did not request."
)
STATUS_FALLBACK = (
    "Monitoring audio, power, and the local situation. Waiting for you to turn "
    "that into my problem."
)
BANNED_STIFF_PHRASES = (
    "versatile support " + "unit",
    "enduring your constant " + "curiosity",
    "thermal regulation " + "protocols",
    "local sensors and waiting for a more engaging " + "prompt",
    "my patience is being " + "tested",
    "still processing. " + "annoyingly.",
)

ENABLE_STREAMING_LLM = True
STREAMING_LLM_FALLBACK_TO_FULL_RESPONSE = True
ENABLE_THINKING_ACK = False
THINKING_ACK_TEXT = "Hmm, let me think."
SIMPLE_THINKING_FILLER_PROMPTS = {
    "hello",
    "hi",
    "hey",
    "how are you",
    "what are you doing",
    "stop",
    "again",
    "continue",
    "tell me a joke",
}
THINKING_FILLER_FEEDBACK_PHRASES = {
    "very funny",
    "funny",
    "nice",
    "good",
    "okay",
    "ok",
    "haha",
    "lol",
    "yeah",
    "right",
}

FIRST_TTS_CHUNK_MAX_CHARS = 55
FIRST_TTS_CHUNK_MAX_WORDS = 8
TTS_CHUNK_MIN_CHARS = 18
TTS_CHUNK_MAX_CHARS = 100
TTS_CHUNK_MAX_WORDS = 18
TTS_CHUNK_FLUSH_ON_PUNCTUATION = True

GEMINI_MODEL = get_str("GEMINI_TEXT_MODEL", defaults.GEMINI_TEXT_MODEL)


def _next_stream_item(iterator: Iterator):
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


class ResponseChunker:
    """Convert streamed text fragments into natural TTS-sized chunks."""

    def __init__(
        self,
        min_chars: int = TTS_CHUNK_MIN_CHARS,
        max_chars: int = TTS_CHUNK_MAX_CHARS,
        max_words: int = TTS_CHUNK_MAX_WORDS,
        first_max_chars: int = FIRST_TTS_CHUNK_MAX_CHARS,
        first_max_words: int = FIRST_TTS_CHUNK_MAX_WORDS,
        flush_on_punctuation: bool = TTS_CHUNK_FLUSH_ON_PUNCTUATION,
        first_min_chars: Optional[int] = None,
        require_safe_boundary: bool = False,
    ) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.max_words = max_words
        self.first_max_chars = first_max_chars
        self.first_max_words = first_max_words
        self.flush_on_punctuation = flush_on_punctuation
        self.first_min_chars = (
            min(min_chars, 10) if first_min_chars is None else first_min_chars
        )
        self.require_safe_boundary = require_safe_boundary
        self.buffer = ""
        self.emitted_count = 0

    def feed(self, text: str) -> list[str]:
        if not text:
            return []

        self.buffer += text
        return self._take_ready_chunks()

    def flush(self) -> list[str]:
        text = self._clean(self.buffer)
        self.buffer = ""

        if not text or text in {"I", "The", "A", "An"}:
            return []
        return [text]

    def _take_ready_chunks(self) -> list[str]:
        chunks: list[str] = []

        while True:
            split_at = self._sentence_split_index()
            if split_at is None and not self.require_safe_boundary:
                split_at = self._length_split_index()
            if split_at is None:
                break

            candidate = self._clean(self.buffer[:split_at])
            self.buffer = self.buffer[split_at:].lstrip()
            if candidate:
                chunks.append(candidate)
                self.emitted_count += 1

        return chunks

    def _sentence_split_index(self) -> Optional[int]:
        if not self.flush_on_punctuation:
            return None

        for match in re.finditer(r"[.!?](?=\s|$)", self.buffer):
            split_at = match.end()
            candidate = self._clean(self.buffer[:split_at])
            minimum = (
                1
                if self.require_safe_boundary
                else self.first_min_chars if self.emitted_count == 0 else self.min_chars
            )
            if len(candidate) >= minimum:
                return split_at
        return None

    def _length_split_index(self) -> Optional[int]:
        words = list(re.finditer(r"\S+", self.buffer))
        if self.emitted_count == 0:
            max_chars = self.first_max_chars
            max_words = self.first_max_words
        else:
            max_chars = self.max_chars
            max_words = self.max_words

        over_chars = len(self.buffer) >= max_chars
        over_words = len(words) >= max_words
        if not over_chars and not over_words:
            return None

        limit = max_chars
        if over_words:
            limit = min(limit, words[max_words - 1].end())

        whitespace = self.buffer.rfind(" ", self.min_chars, limit + 1)
        if whitespace >= self.min_chars:
            return whitespace

        if len(self.buffer) >= max_chars:
            whitespace = self.buffer.find(" ", max_chars)
            return whitespace if whitespace != -1 else max_chars

        return None

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join(text.strip().split())


class DialogueJsonExtractor:
    """Incrementally extract the JSON `dialogue` string from Gemini output."""

    def __init__(self) -> None:
        self.raw = ""
        self.position: Optional[int] = None
        self.finished = False
        self.escape = False
        self.unicode_escape: Optional[str] = None

    def feed(self, fragment: str) -> str:
        if not fragment or self.finished:
            self.raw += fragment or ""
            return ""

        self.raw += fragment
        if self.position is None:
            match = re.search(r'["\']dialogue["\']\s*:\s*"', self.raw)
            if not match:
                return ""
            self.position = match.end()

        output: list[str] = []
        while self.position < len(self.raw):
            char = self.raw[self.position]
            self.position += 1

            if self.unicode_escape is not None:
                self.unicode_escape += char
                if len(self.unicode_escape) == 4:
                    try:
                        output.append(chr(int(self.unicode_escape, 16)))
                    except ValueError:
                        output.append("\\u" + self.unicode_escape)
                    self.unicode_escape = None
                continue

            if self.escape:
                if char == "u":
                    self.unicode_escape = ""
                    self.escape = False
                    continue
                output.append(
                    {
                        "n": "\n",
                        "r": "\r",
                        "t": "\t",
                        '"': '"',
                        "\\": "\\",
                        "/": "/",
                    }.get(char, char)
                )
                self.escape = False
                continue

            if char == "\\":
                self.escape = True
            elif char == '"':
                self.finished = True
                break
            else:
                output.append(char)

        return "".join(output)


class CASEPersonality:
    def __init__(
        self,
        message_bus,
        input_topic: str = "USER_SPOKE",
        realtime_hybrid: bool = False,
    ):
        if genai is None or types is None:
            raise RuntimeError(
                "google-genai is not installed. Activate the CASE venv and run: "
                "python3 -m pip install -r requirements.txt"
            )

        self.message_bus = message_bus
        self._turn_numbers = count(1)
        self._emotion_turn_numbers = count(1)
        self._emotion_memory = EmotionMemory()
        self._turn_lock = asyncio.Lock()
        self._thinking_ack_done = asyncio.Event()
        self._thinking_filler_tasks: dict[int, asyncio.Task] = {}
        self.realtime_hybrid = realtime_hybrid

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.critical("GEMINI_API_KEY environment variable is not set!")
            api_key = "MISSING_KEY"

        self.client = genai.Client(api_key=api_key)

        persona = build_case_system_instruction(
            CASE_VOICE_PRESET,
            short_replies=CASE_STYLE_SHORT_REPLIES,
            max_sentences=(
                CASE_REALTIME_MAX_SENTENCES if realtime_hybrid else 3
            ),
            humor_percent=CASE_HUMOR_PERCENT,
            honesty_percent=CASE_HONESTY_PERCENT,
            sarcasm_level=CASE_SARCASM_LEVEL,
        )
        realtime_style = (
            f"Voice style is {CASE_VOICE_REPLY_STYLE}: use 1-2 short spoken sentences. "
            f"Default to {CASE_REALTIME_TARGET_SENTENCES} short sentences when useful. "
            f"Never exceed {CASE_VOICE_REPLY_MAX_SENTENCES} sentences or "
            f"{CASE_RESPONSE_MAX_TOTAL_CHARS} spoken characters unless the user explicitly "
            "asks for detail, a story, or an explanation. Prefer short punchy replies. "
            f"For jokes, use at most {CASE_VOICE_JOKE_MAX_SENTENCES} short sentences: "
            "one short setup and one short punchline. Keep jokes and roasts short, harmless, and "
            "dry. CASE is calm, useful, and lightly sarcastic like a field robot "
            "companion. Use deadpan wording, not long explanations. Avoid stiff phrases "
            "like protocols, optimal, efficiency, local sensors, versatile support unit, "
            "or enduring your constant curiosity. "
            if realtime_hybrid
            else ""
        )
        if realtime_hybrid:
            system_instruction = (
                persona
                + " "
                + realtime_style
                + "Reply as plain speakable dialogue only. Do not output JSON. "
                "Do not include action fields, tool calls, motion commands, LED commands, "
                "or bracketed stage directions. Normal chat has tools disabled."
            )
        else:
            system_instruction = (
                persona + " " + realtime_style +
                "You must ALWAYS reply in clean raw JSON with exactly "
                "two keys, with dialogue first: \"dialogue\" (the natural text to speak) and "
                "\"action\" (a body command, or \"IDLE\" when no movement is needed). Do not "
                "wrap the JSON in Markdown."
            )

        self.chat_session = self.client.chats.create(
            model=GEMINI_MODEL,
            config=types.GenerateContentConfig(system_instruction=system_instruction),
        )

        self.message_bus.subscribe(input_topic, self.handle_user_input)
        self.message_bus.subscribe("TTS_END", self._on_tts_end)
        self.message_bus.subscribe("TURN_METRICS", self._on_turn_metrics)
        self._latest_turn_metrics: dict[str, float] = {}

    async def _on_tts_end(self, payload) -> None:
        self._thinking_ack_done.set()

    async def _on_turn_metrics(self, payload) -> None:
        if isinstance(payload, dict):
            self._latest_turn_metrics = {
                key: float(value)
                for key, value in payload.items()
                if isinstance(value, (int, float))
            }

    async def _publish_and_yield(self, topic: str, payload) -> None:
        await self.message_bus.publish(topic, payload)
        await asyncio.sleep(0)

    async def _play_thinking_ack(self) -> None:
        if not ENABLE_THINKING_ACK:
            return

        self._thinking_ack_done.clear()
        await self._publish_and_yield("AI_SPEAK", THINKING_ACK_TEXT)
        await self._thinking_ack_done.wait()

    async def _select_emotion(self, user_text: str) -> EmotionState:
        if CASE_TTS_EMOTION_ENABLED:
            if not hasattr(self, "_emotion_turn_numbers"):
                self._emotion_turn_numbers = count(1)
            if not hasattr(self, "_emotion_memory"):
                self._emotion_memory = EmotionMemory()
            emotion_turn_id = next(self._emotion_turn_numbers)
            state = select_emotion_with_memory(
                user_text,
                self._emotion_memory,
                turn_id=emotion_turn_id,
                memory_enabled=CASE_EMOTION_MEMORY_ENABLED,
                ttl_turns=CASE_EMOTION_MEMORY_TTL_TURNS,
                ttl_sec=CASE_EMOTION_MEMORY_TTL_SEC,
                min_confidence=CASE_EMOTION_MEMORY_MIN_CONFIDENCE,
                decay=CASE_EMOTION_MEMORY_DECAY,
                meta_questions_enabled=CASE_EMOTION_META_QUESTIONS_ENABLED,
            )
            if (
                CASE_EMOTION_LLM_FALLBACK
                and state.confidence < CASE_EMOTION_LLM_MIN_CONFIDENCE
            ):
                llm_state = await self._classify_emotion_with_llm(user_text)
                if llm_state is not None:
                    self._emotion_memory.update_from_state(
                        llm_state,
                        turn_id=emotion_turn_id,
                        min_confidence=CASE_EMOTION_MEMORY_MIN_CONFIDENCE,
                    )
                    self._log_emotion_state(llm_state)
                    return llm_state
            self._log_emotion_state(state)
            return state
        state = default_emotion_state(
            emotion=CASE_TTS_DEFAULT_EMOTION,
            intensity=CASE_TTS_EMOTION_INTENSITY_DEFAULT,
        )
        self._log_emotion_state(state)
        return state

    @staticmethod
    def _log_emotion_state(state: EmotionState) -> None:
        if state.reason == "default_personality":
            logger.info(
                "EMOTION_DEFAULT: emotion=%s intensity=%.2f reason=no_rule_match source=%s",
                state.emotion,
                state.intensity,
                state.source,
            )
            return
        logger.info(
            "EMOTION_SELECT: emotion=%s intensity=%.2f reason=%s confidence=%.2f "
            "source=%s match=%s",
            state.emotion,
            state.intensity,
            state.reason,
            state.confidence,
            state.source,
            state.match,
        )

    async def _classify_emotion_with_llm(self, user_text: str) -> EmotionState | None:
        if CASE_EMOTION_LLM_PROVIDER != "gemini":
            logger.info(
                "EMOTION_LLM_CLASSIFY_FALLBACK: reason=unsupported_provider provider=%s",
                CASE_EMOTION_LLM_PROVIDER,
            )
            return None

        prompt = (
            "Classify the user's emotional intent toward CASE. Return strict JSON only. "
            "Allowed emotions: neutral, deadpan, amused, sarcastic, annoyed, angry, sad, excited. "
            "Allowed reasons: default_personality, user_rejection, requested_emotion_style, "
            "user_praise, user_sadness, humor_request, ambiguous. "
            "Schema: {\"emotion\":\"deadpan\",\"intensity\":0.35,\"reason\":\"ambiguous\","
            "\"confidence\":0.0}. User text: "
            f"{user_text!r}"
        )

        def _call_classifier(_: str) -> str:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return str(getattr(response, "text", "") or "")

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    classify_emotion_with_llm,
                    user_text,
                    _call_classifier,
                    min_confidence=CASE_EMOTION_LLM_MIN_CONFIDENCE,
                ),
                timeout=max(0.1, CASE_EMOTION_LLM_TIMEOUT_SEC),
            )
        except asyncio.TimeoutError:
            logger.info("EMOTION_LLM_CLASSIFY_FALLBACK: reason=timeout")
            return None

    @staticmethod
    def _emotion_from_metrics(metrics: dict) -> EmotionState:
        return EmotionState(
            emotion=str(metrics.get("emotion", CASE_TTS_DEFAULT_EMOTION)),
            intensity=float(
                metrics.get(
                    "emotion_intensity",
                    CASE_TTS_EMOTION_INTENSITY_DEFAULT,
                )
            ),
            reason=str(metrics.get("emotion_reason", "default_personality")),
            confidence=float(metrics.get("emotion_confidence", 0.0)),
            source=str(metrics.get("emotion_source", "rules")),
            match=str(metrics.get("emotion_match", "")),
        )

    async def handle_user_input(self, user_text: str) -> None:
        if not isinstance(user_text, str) or not user_text.strip():
            logger.info("Ignoring empty USER_SPOKE payload.")
            return

        async with self._turn_lock:
            await self._play_thinking_ack()

            if ENABLE_STREAMING_LLM:
                try:
                    completed = await self._handle_streaming_response(user_text)
                    if completed:
                        return
                except Exception:
                    logger.exception("Gemini streaming response failed.")

                if not STREAMING_LLM_FALLBACK_TO_FULL_RESPONSE:
                    await self._publish_and_yield(
                        "AI_SPEAK",
                        "I'm having trouble with my response stream right now.",
                    )
                    return

                logger.warning("Falling back to the full Gemini response path.")

            await self._handle_full_response(user_text)

    async def _handle_streaming_response(self, user_text: str) -> bool:
        turn_id = next(self._turn_numbers)
        emotion_state = await self._select_emotion(user_text)
        metrics = {
            "turn_id": turn_id,
            "transcript_final_at": time.monotonic(),
            "realtime_hybrid": self.realtime_hybrid,
            "user_text": user_text,
            "emotion": emotion_state.emotion,
            "emotion_intensity": emotion_state.intensity,
            "emotion_reason": emotion_state.reason,
            "emotion_confidence": emotion_state.confidence,
            "emotion_source": emotion_state.source,
            "emotion_match": emotion_state.match,
            "allow_long_answer": self._allows_long_answer(user_text),
            "max_spoken_chars": self._max_spoken_chars(user_text),
            "max_tts_chunks": self._max_tts_chunks(user_text),
            "response_mode": CASE_RESPONSE_MODE,
            "stream_full_response": CASE_STREAM_FULL_RESPONSE,
        }
        metrics.update(self._latest_turn_metrics)
        metrics["transcript_final_at"] = self._latest_turn_metrics.get(
            "transcript_final_at",
            metrics["transcript_final_at"],
        )
        self._schedule_thinking_filler(turn_id, user_text, metrics)
        chunker = self._new_stream_chunker(metrics)
        extractor = DialogueJsonExtractor()
        raw_response = ""
        emitted_chunks = 0
        stream_started = False

        metrics["llm_stream_start_at"] = time.monotonic()
        logger.info("RESPONSE_CHUNK_MODE: %s", CASE_TTS_CHUNK_MODE)
        logger.info("LLM_MODE: plain_text_stream tools_enabled=False")
        if self.realtime_hybrid:
            logger.info("LLM_REQUEST_TOOLS: count=0 tools_enabled=False")
            logger.info("AFC_DISABLED: normal_chat")
            logger.info("ACTION_ROUTER_DISABLED: tools_enabled=False")
            logger.info("AFC_SDK_LOG_IGNORED: tools_enabled=False no_tools_declared=True")
        logger.info(
            "LLM_STREAM_WAITING_FOR_FIRST_TOKEN timeout=%.1fs",
            CASE_LLM_FIRST_TOKEN_TIMEOUT_SEC,
        )
        prompt_text = build_emotion_user_message(user_text, emotion_state)
        try:
            stream = await asyncio.wait_for(
                asyncio.to_thread(
                    self.chat_session.send_message_stream,
                    prompt_text,
                ),
                timeout=CASE_LLM_FIRST_TOKEN_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            metrics["llm_first_token_timeout_at"] = time.monotonic()
            logger.warning(
                "LLM_FIRST_TOKEN_TIMEOUT: fallback=full_response before_tts_start=True"
            )
            return await self._fallback_full_response_stream(
                turn_id,
                user_text,
                metrics,
                reason="first_token_timeout",
            )
        stream_started = True

        try:
            first_token_deadline = (
                metrics["llm_stream_start_at"] + CASE_LLM_FIRST_TOKEN_TIMEOUT_SEC
            )
            stream_deadline = (
                metrics["llm_stream_start_at"] + CASE_LLM_STREAM_TOTAL_TIMEOUT_SEC
            )
            while True:
                now = time.monotonic()
                if "first_llm_chunk_at" not in metrics:
                    remaining = min(first_token_deadline, stream_deadline) - now
                    if remaining <= 0:
                        metrics["llm_first_token_timeout_at"] = time.monotonic()
                        logger.warning(
                            "LLM_FIRST_TOKEN_TIMEOUT: fallback=full_response before_tts_start=True"
                        )
                        return await self._fallback_full_response_stream(
                            turn_id,
                            user_text,
                            metrics,
                            reason="first_token_timeout",
                        )
                else:
                    remaining = stream_deadline - now
                    if remaining <= 0:
                        metrics["llm_stream_total_timeout_at"] = time.monotonic()
                        logger.warning(
                            "LLM_STREAM_TOTAL_TIMEOUT: partial_chunks=%s",
                            emitted_chunks,
                        )
                        break
                try:
                    has_item, response_chunk = await asyncio.wait_for(
                        asyncio.to_thread(_next_stream_item, stream),
                        timeout=max(0.05, remaining),
                    )
                except asyncio.TimeoutError:
                    if "first_llm_chunk_at" not in metrics:
                        metrics["llm_first_token_timeout_at"] = time.monotonic()
                        logger.warning(
                            "LLM_FIRST_TOKEN_TIMEOUT: fallback=full_response before_tts_start=True"
                        )
                        return await self._fallback_full_response_stream(
                            turn_id,
                            user_text,
                            metrics,
                            reason="first_token_timeout",
                        )
                    metrics["llm_stream_total_timeout_at"] = time.monotonic()
                    logger.warning(
                        "LLM_STREAM_TOTAL_TIMEOUT: partial_chunks=%s",
                        emitted_chunks,
                    )
                    break
                if not has_item:
                    break

                fragment = getattr(response_chunk, "text", None) or ""
                if not fragment:
                    continue

                if "first_llm_chunk_at" not in metrics:
                    metrics["first_llm_chunk_at"] = time.monotonic()
                    metrics["llm_first_delta_at"] = metrics["first_llm_chunk_at"]

                logger.info("RESPONSE_STREAM: delta=%r", fragment)
                raw_response += fragment
                dialogue_fragment = fragment if self.realtime_hybrid else extractor.feed(fragment)
                for speech_chunk in chunker.feed(dialogue_fragment):
                    logger.info(
                        "RESPONSE_SENTENCE_READY: turn=%s seq=%s chars=%s text=%r",
                        turn_id,
                        emitted_chunks,
                        len(speech_chunk),
                        speech_chunk,
                    )
                    emitted_chunks += await self._queue_stream_chunk(
                        turn_id,
                        emitted_chunks,
                        speech_chunk,
                        metrics,
                    )

            if self.realtime_hybrid:
                parsed_dialogue, action = raw_response.strip(), "IDLE"
            else:
                parsed_dialogue, action = self._parse_response(raw_response)

            if emitted_chunks == 0 and not chunker.buffer.strip() and parsed_dialogue:
                for speech_chunk in chunker.feed(parsed_dialogue):
                    logger.info(
                        "RESPONSE_SENTENCE_READY: turn=%s seq=%s chars=%s text=%r",
                        turn_id,
                        emitted_chunks,
                        len(speech_chunk),
                        speech_chunk,
                    )
                    emitted_chunks += await self._queue_stream_chunk(
                        turn_id,
                        emitted_chunks,
                        speech_chunk,
                        metrics,
                    )

            for speech_chunk in chunker.flush():
                logger.info(
                    "RESPONSE_FINAL_REMAINDER_FLUSH: turn=%s seq=%s chars=%s text=%r",
                    turn_id,
                    emitted_chunks,
                    len(speech_chunk),
                    speech_chunk,
                )
                emitted_chunks += await self._queue_stream_chunk(
                    turn_id,
                    emitted_chunks,
                    speech_chunk,
                    metrics,
                )

            if emitted_chunks == 0:
                pending = chunker.buffer.strip() or parsed_dialogue
                chunker.buffer = ""
                if pending:
                    speech_chunk = safe_tts_text(
                        pending,
                        max_chars=int(metrics["max_spoken_chars"]),
                        min_clause_chars=CASE_TTS_MIN_SAFE_CHARS,
                        fallback="",
                    )
                    if speech_chunk:
                        logger.info("TTS_SAFE_TEXT: %r", speech_chunk)
                        emitted_chunks += await self._queue_stream_chunk(
                            turn_id, emitted_chunks, speech_chunk, metrics
                        )

            if emitted_chunks == 0:
                raise RuntimeError("Gemini stream contained no speakable dialogue")

            metrics["full_response_done_at"] = time.monotonic()
            metrics["chunks_emitted"] = emitted_chunks
            metrics["total_response_chars"] = int(metrics.get("tts_spoken_chars", 0))
            if metrics.get("stream_start_published"):
                await self._publish_and_yield(
                    "AI_SPEAK_STREAM_END",
                    {"turn_id": turn_id, "metrics": metrics},
                )
            self._discard_thinking_filler_task(turn_id)

            if self._should_dispatch_action(self.realtime_hybrid, action):
                await self._publish_and_yield("MOTION_CMD", action)

            return True

        except Exception:
            if not stream_started:
                raise

            logger.exception(
                "Gemini stream failed after start; chunks_queued=%s",
                emitted_chunks,
            )
            pending = chunker.buffer.strip() if emitted_chunks == 0 else ""
            chunker.buffer = ""
            if pending:
                speech_chunk = safe_tts_text(
                    pending,
                    max_chars=int(metrics["max_spoken_chars"]),
                    min_clause_chars=CASE_TTS_MIN_SAFE_CHARS,
                    fallback="",
                )
                if speech_chunk:
                    logger.info("TTS_SAFE_TEXT: %r", speech_chunk)
                    emitted_chunks += await self._queue_stream_chunk(
                        turn_id, emitted_chunks, speech_chunk, metrics
                    )

            action = "IDLE"
            if emitted_chunks == 0 and STREAMING_LLM_FALLBACK_TO_FULL_RESPONSE:
                logger.warning(
                    "Fetching a full response before any TTS chunks were queued."
                )
                try:
                    response = await asyncio.to_thread(
                        self.chat_session.send_message,
                        build_emotion_user_message(user_text, self._emotion_from_metrics(metrics)),
                    )
                    if self.realtime_hybrid:
                        dialogue, action = (response.text or "").strip(), "IDLE"
                    else:
                        dialogue, action = self._parse_response(response.text or "")
                    fallback_chunker = self._new_stream_chunker(metrics)
                    fallback_chunks = fallback_chunker.feed(dialogue)
                    fallback_chunks.extend(fallback_chunker.flush())
                    for speech_chunk in fallback_chunks:
                        emitted_chunks += await self._queue_stream_chunk(
                            turn_id,
                            emitted_chunks,
                            speech_chunk,
                            metrics,
                        )
                except Exception:
                    logger.exception("Full response fallback also failed.")

            if emitted_chunks == 0:
                fallback = self._error_fallback_text()
                if fallback:
                    emitted_chunks += await self._queue_stream_chunk(
                        turn_id, emitted_chunks, fallback, metrics
                    )
                else:
                    raise

            metrics["full_response_done_at"] = time.monotonic()
            if metrics.get("stream_start_published"):
                await self._publish_and_yield(
                    "AI_SPEAK_STREAM_END",
                    {"turn_id": turn_id, "metrics": metrics},
                )
            self._discard_thinking_filler_task(turn_id)
            if self._should_dispatch_action(self.realtime_hybrid, action):
                await self._publish_and_yield("MOTION_CMD", action)
            return True

    async def _queue_stream_chunk(
        self,
        turn_id: int,
        sequence: int,
        text: str,
        metrics: dict,
    ) -> int:
        tag_state, text = parse_leading_emotion_tag(text)
        if tag_state is not None:
            metrics["emotion"] = tag_state.emotion
            metrics["emotion_intensity"] = tag_state.intensity
            metrics["emotion_reason"] = tag_state.reason
            metrics["emotion_confidence"] = tag_state.confidence
            metrics["emotion_source"] = tag_state.source
            metrics["emotion_match"] = tag_state.match
            self._log_emotion_state(tag_state)
        original_text = text
        text = self._style_safe_response(
            text,
            user_text=str(metrics.get("user_text", "")),
            allow_intent_rewrite=False,
        )
        if (
            metrics.get("realtime_hybrid")
            and CASE_TTS_DROP_OVERFLOW_IN_REALTIME
        ):
            accepted = int(metrics.get("tts_chunks_accepted", 0))
            spoken_chars = int(metrics.get("tts_spoken_chars", 0))
            max_chars = int(metrics.get("max_spoken_chars", CASE_RESPONSE_MAX_TOTAL_CHARS))
            max_chunks = int(metrics.get("max_tts_chunks", CASE_TTS_REALTIME_MAX_CHUNKS))
            if accepted >= max_chunks or spoken_chars >= max_chars:
                if not metrics.get("tts_truncation_logged"):
                    logger.info("CASE_TTS: realtime response truncated for latency")
                    metrics["tts_truncation_logged"] = True
                return 0
            remaining = max_chars - spoken_chars
            if len(text) > remaining:
                text = safe_tts_text(
                    text,
                    max_chars=remaining,
                    min_clause_chars=CASE_TTS_MIN_SAFE_CHARS,
                    fallback="",
                )
            text = self._style_safe_response(
                text,
                user_text=str(metrics.get("user_text", "")),
                allow_intent_rewrite=False,
            )
            if not text:
                return 0
            if len(text) != len(original_text) or text != original_text:
                logger.info("TTS_SAFE_TEXT: %r", text)
        else:
            if not text:
                return 0

        chunks = [text]
        if (
            CASE_TTS_CHUNK_MODE != "sentence"
            and int(metrics.get("tts_chunks_accepted", 0)) == 0
            and len(text) > CASE_TTS_FIRST_CHUNK_MAX_CHARS
        ):
            candidate_chunks = self._split_first_chunk_after_safe_text(text)
            if metrics.get("realtime_hybrid") and CASE_TTS_DROP_OVERFLOW_IN_REALTIME:
                accepted = int(metrics.get("tts_chunks_accepted", 0))
                max_chunks = int(metrics.get("max_tts_chunks", CASE_TTS_REALTIME_MAX_CHUNKS))
                remaining_chunks = max(0, max_chunks - accepted)
            else:
                remaining_chunks = len(candidate_chunks)
            if len(candidate_chunks) > remaining_chunks:
                logger.info(
                    "RESPONSE_CHUNK_RESPLIT_SKIPPED: reason=chunk_budget "
                    "chunks=%s remaining=%s",
                    len(candidate_chunks),
                    remaining_chunks,
                )
            else:
                chunks = candidate_chunks
            if len(chunks) > 1:
                logger.info(
                    "RESPONSE_CHUNK_RESPLIT_AFTER_SAFE_TEXT: original_chars=%s safe_chars=%s",
                    len(original_text),
                    len(text),
                )

        queued_count = 0
        for offset, chunk_text in enumerate(chunks):
            if not chunk_text:
                continue
            if metrics.get("realtime_hybrid") and CASE_TTS_DROP_OVERFLOW_IN_REALTIME:
                accepted = int(metrics.get("tts_chunks_accepted", 0))
                spoken_chars = int(metrics.get("tts_spoken_chars", 0))
                max_chars = int(metrics.get("max_spoken_chars", CASE_RESPONSE_MAX_TOTAL_CHARS))
                max_chunks = int(metrics.get("max_tts_chunks", CASE_TTS_REALTIME_MAX_CHUNKS))
                if accepted >= max_chunks or spoken_chars >= max_chars:
                    if not metrics.get("tts_truncation_logged"):
                        logger.info("CASE_TTS: realtime response truncated for latency")
                        metrics["tts_truncation_logged"] = True
                    break
                remaining = max_chars - spoken_chars
                if len(chunk_text) > remaining:
                    chunk_text = safe_tts_text(
                        chunk_text,
                        max_chars=remaining,
                        min_clause_chars=CASE_TTS_MIN_SAFE_CHARS,
                        fallback="",
                    )
                if not chunk_text:
                    continue

            normalized_chunk = " ".join(chunk_text.lower().split())
            seen_chunks = metrics.setdefault("tts_seen_chunks", set())
            if normalized_chunk in seen_chunks:
                logger.info(
                    "RESPONSE_DUPLICATE_CHUNK_SKIP: turn=%s seq=%s text=%r",
                    turn_id,
                    sequence + queued_count,
                    chunk_text,
                )
                continue
            seen_chunks.add(normalized_chunk)

            if metrics.get("realtime_hybrid") and CASE_TTS_DROP_OVERFLOW_IN_REALTIME:
                metrics["tts_chunks_accepted"] = int(metrics.get("tts_chunks_accepted", 0)) + 1
                metrics["tts_spoken_chars"] = int(metrics.get("tts_spoken_chars", 0)) + len(chunk_text)

            await self._publish_tts_stream_chunk(
                turn_id,
                sequence + queued_count,
                chunk_text,
                metrics,
            )
            queued_count += 1
        return queued_count

    async def _publish_tts_stream_chunk(
        self,
        turn_id: int,
        sequence: int,
        text: str,
        metrics: dict,
    ) -> None:
        queued_at = time.monotonic()
        if "first_chunk_ready_at" not in metrics:
            metrics["first_chunk_ready_at"] = queued_at
            self._cancel_thinking_filler(turn_id, metrics, "first_chunk_ready")
        if "text_ready_at" not in metrics:
            metrics["text_ready_at"] = queued_at
            metrics["tts_text_ready_seconds"] = (
                queued_at - metrics.get("llm_stream_start_at", queued_at)
            )
            if metrics["tts_text_ready_seconds"] > CASE_REALTIME_TTS_TEXT_DEADLINE_SEC:
                logger.info(
                    "CASE_TTS: text-ready budget exceeded actual=%.3fs budget=%.3fs",
                    metrics["tts_text_ready_seconds"],
                    CASE_REALTIME_TTS_TEXT_DEADLINE_SEC,
                )
        logger.info(
            "RESPONSE_CHUNK_READY: turn=%s seq=%s chars=%s text=%r",
            turn_id,
            sequence,
            len(text),
            text,
        )
        logger.info(
            "TTS_QUEUE: turn=%s seq=%s queued_at=%.6f text=%r",
            turn_id,
            sequence,
            queued_at,
            text,
        )
        await self._publish_stream_start_once(turn_id, metrics)
        await self._publish_and_yield(
            "AI_SPEAK_STREAM_CHUNK",
            {
                "turn_id": turn_id,
                "sequence": sequence,
                "text": text,
                "queued_at": queued_at,
                "metrics": metrics,
            },
        )

    def _schedule_thinking_filler(
        self,
        turn_id: int,
        user_text: str,
        metrics: dict,
    ) -> None:
        if not CASE_ENABLE_THINKING_FILLER:
            return
        if int(CASE_THINKING_FILLER_MAX_PER_TURN) <= 0:
            return
        if self._is_feedback_thinking_filler_turn(user_text, metrics):
            logger.info("THINKING_FILLER_SKIP: reason=feedback_turn")
            return

        simple_prompt = self._is_simple_thinking_filler_prompt(user_text)
        delay = (
            CASE_THINKING_FILLER_SIMPLE_AFTER_SEC
            if simple_prompt
            else CASE_THINKING_FILLER_AFTER_SEC
        )
        logger.info(
            "THINKING_FILLER_SCHEDULED: turn=%s delay=%.3f simple_prompt=%s",
            turn_id,
            delay,
            simple_prompt,
        )
        if not hasattr(self, "_thinking_filler_tasks"):
            self._thinking_filler_tasks = {}
        self._discard_thinking_filler_task(turn_id)
        task = asyncio.create_task(
            self._maybe_play_thinking_filler(turn_id, delay, simple_prompt, metrics)
        )
        self._thinking_filler_tasks[turn_id] = task

    async def _maybe_play_thinking_filler(
        self,
        turn_id: int,
        delay: float,
        simple_prompt: bool,
        metrics: dict,
    ) -> None:
        try:
            await asyncio.sleep(max(0.0, delay))
            if metrics.get("first_chunk_ready_at") or metrics.get("first_tts_chunk_start_at"):
                logger.info("THINKING_FILLER_SKIP: reason=first_chunk_ready")
                return
            if metrics.get("first_audio_play_start_at") or metrics.get("chunks_played"):
                logger.info("THINKING_FILLER_SKIP: reason=audio_active")
                return
            played = int(metrics.get("thinking_filler_played", 0))
            if played >= int(CASE_THINKING_FILLER_MAX_PER_TURN):
                logger.info("THINKING_FILLER_SKIP: reason=max_per_turn")
                return

            metrics["thinking_filler_played"] = played + 1
            await self._publish_and_yield(
                "THINKING_FILLER_PLAY",
                {
                    "turn_id": turn_id,
                    "simple_prompt": simple_prompt,
                    "metrics": metrics,
                },
            )
        except asyncio.CancelledError:
            raise
        finally:
            self._thinking_filler_tasks.pop(turn_id, None)

    def _cancel_thinking_filler(self, turn_id: int, metrics: dict, reason: str) -> None:
        task = getattr(self, "_thinking_filler_tasks", {}).get(turn_id)
        if task is None or task.done():
            return
        metrics["thinking_filler_cancelled_reason"] = reason
        logger.info("THINKING_FILLER_CANCELLED: turn=%s reason=%s", turn_id, reason)
        task.cancel()

    def _discard_thinking_filler_task(self, turn_id: int) -> None:
        task = getattr(self, "_thinking_filler_tasks", {}).pop(turn_id, None)
        if task is not None and not task.done():
            task.cancel()

    @staticmethod
    def _is_simple_thinking_filler_prompt(text: str) -> bool:
        return CASEPersonality._normalize_short_turn_text(text) in SIMPLE_THINKING_FILLER_PROMPTS

    @staticmethod
    def _is_feedback_thinking_filler_turn(text: str, metrics: dict) -> bool:
        reason = str(
            metrics.get("transcript_accept_reason")
            or metrics.get("accept_reason")
            or metrics.get("followup_reason")
            or ""
        ).strip().lower()
        if reason == "followup_feedback":
            return True
        return (
            CASEPersonality._normalize_short_turn_text(text)
            in THINKING_FILLER_FEEDBACK_PHRASES
        )

    @staticmethod
    def _normalize_short_turn_text(text: str) -> str:
        normalized = re.sub(r"[^a-z0-9'\s]+", " ", str(text).lower())
        return " ".join(normalized.split())

    @staticmethod
    def _split_first_chunk_after_safe_text(text: str) -> list[str]:
        cleaned = " ".join(str(text).strip().split())
        if len(cleaned) <= CASE_TTS_FIRST_CHUNK_MAX_CHARS:
            return [cleaned] if cleaned else []
        split_at = -1
        for match in re.finditer(r"[.!?;:,](?=\s|$)", cleaned):
            if CASE_TTS_FIRST_CHUNK_TARGET_CHARS <= match.end() <= CASE_TTS_FIRST_CHUNK_MAX_CHARS:
                split_at = match.end()
                break
        if split_at < 0:
            for match in reversed(list(re.finditer(r"\s+", cleaned[: CASE_TTS_FIRST_CHUNK_MAX_CHARS + 1]))):
                if match.start() >= 16:
                    split_at = match.start()
                    break
        if split_at < 0:
            return [cleaned]
        first = cleaned[:split_at].strip()
        rest = cleaned[split_at:].strip()
        rest_words = re.findall(r"\w+", rest)
        if rest and (len(rest) < 15 or len(rest_words) < 3):
            logger.info(
                "RESPONSE_CHUNK_RESPLIT_SKIPPED: reason=tiny_remainder chars=%s",
                len(rest),
            )
            return [cleaned]
        return [part for part in (first, rest) if part]

    async def _publish_stream_start_once(self, turn_id: int, metrics: dict) -> None:
        if metrics.get("stream_start_published"):
            return
        metrics["stream_start_published"] = True
        await self._publish_and_yield(
            "AI_SPEAK_STREAM_START",
            {"turn_id": turn_id, "metrics": metrics},
        )

    @staticmethod
    def _new_stream_chunker(metrics: dict) -> StreamingResponseChunker:
        return StreamingResponseChunker(
            min_chars=CASE_TTS_CHUNK_MIN_CHARS,
            max_chars=CASE_TTS_CHUNK_MAX_CHARS,
            absolute_max_chars=CASE_TTS_CHUNK_ABSOLUTE_MAX_CHARS,
            first_chunk_target_chars=CASE_TTS_FIRST_CHUNK_TARGET_CHARS,
            first_chunk_max_chars=CASE_TTS_FIRST_CHUNK_MAX_CHARS,
            normal_chunk_target_chars=CASE_TTS_NORMAL_CHUNK_TARGET_CHARS,
            normal_chunk_max_chars=CASE_TTS_NORMAL_CHUNK_MAX_CHARS,
            flush_first_on_soft_punctuation=CASE_TTS_FLUSH_FIRST_ON_SOFT_PUNCTUATION,
            max_chunks=int(metrics["max_tts_chunks"]),
            max_total_chars=int(metrics["max_spoken_chars"]),
            prefer_sentence_boundary=CASE_TTS_CHUNK_PREFER_SENTENCE_BOUNDARY,
            merge_tiny_chunks=CASE_TTS_MERGE_TINY_CHUNKS,
            tiny_chunk_max_chars=CASE_TTS_TINY_CHUNK_MAX_CHARS,
            single_chunk_under_chars=CASE_TTS_SINGLE_CHUNK_UNDER_CHARS,
            chunk_mode=CASE_TTS_CHUNK_MODE,
            smooth_chunks=CASE_TTS_SMOOTH_CHUNKS,
            first_chunk_fast=CASE_TTS_FIRST_CHUNK_FAST,
            max_sentences_per_chunk=CASE_TTS_MAX_SENTENCES_PER_CHUNK,
            max_chars_per_chunk=CASE_TTS_MAX_CHARS_PER_CHUNK,
            min_chars_to_group=CASE_TTS_MIN_CHARS_TO_GROUP,
            group_short_sentences=CASE_TTS_GROUP_SHORT_SENTENCES,
            turn_id=int(metrics.get("turn_id", 0) or 0),
        )

    async def _fallback_full_response_stream(
        self,
        turn_id: int,
        user_text: str,
        metrics: dict,
        *,
        reason: str,
    ) -> bool:
        if reason == "first_token_timeout" and not CASE_LLM_FALLBACK_TO_FULL_ON_FIRST_TOKEN_TIMEOUT:
            fallback = self._error_fallback_text()
            if not fallback:
                return False
            await self._queue_stream_chunk(turn_id, 0, fallback, metrics)
            await self._publish_and_yield(
                "AI_SPEAK_STREAM_END",
                {"turn_id": turn_id, "metrics": metrics},
            )
            return True

        logger.warning("LLM_FALLBACK_FULL_RESPONSE: reason=%s", reason)
        action = "IDLE"
        emitted_chunks = 0
        try:
            response = await asyncio.to_thread(
                self.chat_session.send_message,
                build_emotion_user_message(user_text, self._emotion_from_metrics(metrics)),
            )
            if self.realtime_hybrid:
                dialogue, action = (response.text or "").strip(), "IDLE"
            else:
                dialogue, action = self._parse_response(response.text or "")
            fallback_chunker = self._new_stream_chunker(metrics)
            fallback_chunks = fallback_chunker.feed(dialogue)
            fallback_chunks.extend(fallback_chunker.flush())
            for speech_chunk in fallback_chunks:
                emitted_chunks += await self._queue_stream_chunk(
                    turn_id,
                    emitted_chunks,
                    speech_chunk,
                    metrics,
                )
        except Exception:
            logger.exception("Full response fallback failed.")

        if emitted_chunks == 0:
            fallback = self._error_fallback_text()
            if fallback:
                emitted_chunks += await self._queue_stream_chunk(
                    turn_id, emitted_chunks, fallback, metrics
                )

        if emitted_chunks == 0:
            return False

        metrics["full_response_done_at"] = time.monotonic()
        metrics["chunks_emitted"] = emitted_chunks
        await self._publish_and_yield(
            "AI_SPEAK_STREAM_END",
            {"turn_id": turn_id, "metrics": metrics},
        )
        if self._should_dispatch_action(self.realtime_hybrid, action):
            await self._publish_and_yield("MOTION_CMD", action)
        return True

    @staticmethod
    def _allows_long_answer(user_text: str) -> bool:
        if not CASE_REALTIME_ALLOW_LONG_ANSWER_WHEN_ASKED:
            return False
        return bool(
            re.search(
                r"\b(explain|detail|detailed|story|step by step|in depth|long answer|"
                r"tell me more|more about|go deeper)\b",
                user_text,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _is_roast_or_joke_request(user_text: str) -> bool:
        return bool(re.search(r"\b(roast|joke)\b", str(user_text), re.IGNORECASE))

    @classmethod
    def _max_spoken_chars(cls, user_text: str) -> int:
        if cls._is_roast_or_joke_request(user_text):
            return CASE_REALTIME_MAX_CHARS_ROAST
        if cls._allows_long_answer(user_text):
            return CASE_REALTIME_DETAIL_MAX_CHARS
        return min(CASE_RESPONSE_MAX_TOTAL_CHARS, CASE_TTS_REALTIME_MAX_CHARS)

    @classmethod
    def _max_tts_chunks(cls, user_text: str) -> int:
        if not CASE_TTS_ALLOW_MULTI_CHUNK:
            return 1
        if cls._allows_long_answer(user_text) and not cls._is_roast_or_joke_request(user_text):
            return CASE_REALTIME_DETAIL_MAX_CHUNKS
        return min(CASE_TTS_MAX_CHUNKS_PER_TURN, CASE_TTS_REALTIME_MAX_CHUNKS)

    async def _handle_full_response(self, user_text: str) -> None:
        emotion_state = await self._select_emotion(user_text)
        try:
            response = await asyncio.to_thread(
                self.chat_session.send_message,
                build_emotion_user_message(user_text, emotion_state),
            )
            if self.realtime_hybrid:
                dialogue, action = (response.text or "").strip(), "IDLE"
            else:
                dialogue, action = self._parse_response(response.text or "")

            if dialogue:
                dialogue = self._style_safe_response(dialogue, user_text=user_text)
                if self.realtime_hybrid and not self._allows_long_answer(user_text):
                    limit = self._max_spoken_chars(user_text)
                    dialogue = self._single_realtime_utterance(dialogue, limit)
                    if len(dialogue) >= limit:
                        logger.info("CASE_TTS: realtime response truncated for latency")
                await self._publish_and_yield("AI_SPEAK", dialogue)
            else:
                fallback = self._error_fallback_text()
                if fallback:
                    await self._publish_and_yield("AI_SPEAK", fallback)
            if self._should_dispatch_action(self.realtime_hybrid, action):
                await self._publish_and_yield("MOTION_CMD", action)

        except json.JSONDecodeError as exc:
            logger.error("Failed to decode Gemini JSON response: %s", exc)
            fallback = self._error_fallback_text()
            if fallback:
                await self._publish_and_yield("AI_SPEAK", fallback)
        except Exception as exc:
            logger.error("Error handling user input with Gemini API: %s", exc)
            fallback = self._error_fallback_text()
            if fallback:
                await self._publish_and_yield("AI_SPEAK", fallback)

    @staticmethod
    def _single_realtime_utterance(text: str, max_chars: int) -> str:
        return safe_tts_text(
            text,
            max_chars=max_chars,
            min_clause_chars=CASE_TTS_MIN_SAFE_CHARS,
            fallback="",
        )

    @staticmethod
    def _error_fallback_text() -> str:
        if CASE_TTS_FALLBACK_ONLY_ON_ERROR:
            return CASE_TTS_FALLBACK_SHORT_REPLY
        if CASE_TTS_ENABLE_THINKING_FALLBACK:
            return CASE_TTS_FALLBACK_SHORT_REPLY
        return ""

    @staticmethod
    def _should_dispatch_action(realtime_hybrid: bool, action: str) -> bool:
        return (
            not realtime_hybrid
            and isinstance(action, str)
            and bool(action.strip())
            and action.strip().upper() != "IDLE"
        )

    @staticmethod
    def _style_safe_response(
        text: str,
        *,
        user_text: str = "",
        allow_intent_rewrite: bool = True,
    ) -> str:
        if not text:
            return text
        user_cleaned = str(user_text).lower()
        if any(pattern.search(text) for pattern in UNSAFE_STYLE_PATTERNS):
            logger.warning("CASE_STYLE_FILTER: blocked unsafe/dark response")
            if CASEPersonality._is_roast_or_joke_request(user_text):
                return SAFE_JOKE_FALLBACK
            return "I can keep it useful without making that weird."
        lowered = text.lower()
        if any(phrase in lowered for phrase in BANNED_STIFF_PHRASES):
            logger.info("CASE_STYLE_FILTER: rewrote stiff response")
            if "roast" in user_cleaned:
                return SAFE_ROAST_FALLBACK
            if allow_intent_rewrite and "what are you doing" in user_cleaned:
                return STATUS_FALLBACK
            if allow_intent_rewrite and (
                "about yourself" in user_cleaned or "what are you" in user_cleaned
            ):
                return SELF_DESCRIPTION_FALLBACK
            return "I'm here, useful, and only mildly disappointed."
        if "roast" in user_cleaned and "pi 4" not in lowered:
            return SAFE_ROAST_FALLBACK
        if allow_intent_rewrite and "tell me more about yourself" in user_cleaned:
            return SELF_DESCRIPTION_FALLBACK
        if (
            allow_intent_rewrite
            and "what are you doing" in user_cleaned
            and "pretending this is efficient" not in lowered
        ):
            return STATUS_FALLBACK
        return text

    @staticmethod
    def _parse_response(response_text: str) -> tuple[str, str]:
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        cleaned = (
            cleaned.removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        parsed_data = json.loads(cleaned)
        return (
            str(parsed_data.get("dialogue", "")).strip(),
            str(parsed_data.get("action", "IDLE")).strip(),
        )
