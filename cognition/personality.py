import asyncio
import json
import logging
import os
import re
import time
from itertools import count
from typing import Iterator, Optional

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


logger = logging.getLogger(__name__)

ENABLE_STREAMING_LLM = True
STREAMING_LLM_FALLBACK_TO_FULL_RESPONSE = True
ENABLE_THINKING_ACK = False
THINKING_ACK_TEXT = "Hmm, let me think."

FIRST_TTS_CHUNK_MAX_CHARS = 55
FIRST_TTS_CHUNK_MAX_WORDS = 8
TTS_CHUNK_MIN_CHARS = 18
TTS_CHUNK_MAX_CHARS = 100
TTS_CHUNK_MAX_WORDS = 18
TTS_CHUNK_FLUSH_ON_PUNCTUATION = True

GEMINI_MODEL = "gemini-3.1-flash-lite"


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
    ) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.max_words = max_words
        self.first_max_chars = first_max_chars
        self.first_max_words = first_max_words
        self.flush_on_punctuation = flush_on_punctuation
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
            if split_at is None:
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
            minimum = min(self.min_chars, 10) if self.emitted_count == 0 else self.min_chars
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
    def __init__(self, message_bus, input_topic: str = "USER_SPOKE"):
        if genai is None or types is None:
            raise RuntimeError(
                "google-genai is not installed. Activate the CASE venv and run: "
                "python3 -m pip install -r requirements.txt"
            )

        self.message_bus = message_bus
        self._turn_numbers = count(1)
        self._turn_lock = asyncio.Lock()
        self._thinking_ack_done = asyncio.Event()

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.critical("GEMINI_API_KEY environment variable is not set!")
            api_key = "MISSING_KEY"

        self.client = genai.Client(api_key=api_key)

        system_instruction = (
            "You are CASE, a physical robot companion with a witty, slightly sarcastic "
            "personality. Speak naturally. For simple questions, be concise. For detailed "
            "questions, answer fully. You must ALWAYS reply in clean raw JSON with exactly "
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

    async def _on_tts_end(self, payload) -> None:
        self._thinking_ack_done.set()

    async def _publish_and_yield(self, topic: str, payload) -> None:
        await self.message_bus.publish(topic, payload)
        await asyncio.sleep(0)

    async def _play_thinking_ack(self) -> None:
        if not ENABLE_THINKING_ACK:
            return

        self._thinking_ack_done.clear()
        await self._publish_and_yield("AI_SPEAK", THINKING_ACK_TEXT)
        await self._thinking_ack_done.wait()

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
        metrics = {
            "turn_id": turn_id,
            "transcript_final_at": time.monotonic(),
        }
        chunker = ResponseChunker()
        extractor = DialogueJsonExtractor()
        raw_response = ""
        emitted_chunks = 0
        stream_started = False

        metrics["llm_stream_start_at"] = time.monotonic()
        stream = await asyncio.to_thread(
            self.chat_session.send_message_stream,
            user_text,
        )

        await self._publish_and_yield(
            "AI_SPEAK_STREAM_START",
            {"turn_id": turn_id, "metrics": metrics},
        )
        stream_started = True

        try:
            while True:
                has_item, response_chunk = await asyncio.to_thread(
                    _next_stream_item,
                    stream,
                )
                if not has_item:
                    break

                fragment = getattr(response_chunk, "text", None) or ""
                if not fragment:
                    continue

                if "first_llm_chunk_at" not in metrics:
                    metrics["first_llm_chunk_at"] = time.monotonic()

                raw_response += fragment
                dialogue_fragment = extractor.feed(fragment)
                for speech_chunk in chunker.feed(dialogue_fragment):
                    await self._queue_stream_chunk(
                        turn_id,
                        emitted_chunks,
                        speech_chunk,
                        metrics,
                    )
                    emitted_chunks += 1

            parsed_dialogue, action = self._parse_response(raw_response)

            if emitted_chunks == 0 and not chunker.buffer.strip() and parsed_dialogue:
                for speech_chunk in chunker.feed(parsed_dialogue):
                    await self._queue_stream_chunk(
                        turn_id,
                        emitted_chunks,
                        speech_chunk,
                        metrics,
                    )
                    emitted_chunks += 1

            for speech_chunk in chunker.flush():
                await self._queue_stream_chunk(
                    turn_id,
                    emitted_chunks,
                    speech_chunk,
                    metrics,
                )
                emitted_chunks += 1

            if emitted_chunks == 0:
                raise RuntimeError("Gemini stream contained no speakable dialogue")

            metrics["full_response_done_at"] = time.monotonic()
            await self._publish_and_yield(
                "AI_SPEAK_STREAM_END",
                {"turn_id": turn_id, "metrics": metrics},
            )

            if action and action.upper() != "IDLE":
                await self._publish_and_yield("MOTION_CMD", action)

            return True

        except Exception:
            if not stream_started:
                raise

            logger.exception("Gemini stream failed after the TTS turn started.")
            for speech_chunk in chunker.flush():
                await self._queue_stream_chunk(
                    turn_id,
                    emitted_chunks,
                    speech_chunk,
                    metrics,
                )
                emitted_chunks += 1

            action = "IDLE"
            if emitted_chunks == 0 and STREAMING_LLM_FALLBACK_TO_FULL_RESPONSE:
                logger.warning("Fetching a full response inside the active TTS turn.")
                try:
                    response = await asyncio.to_thread(
                        self.chat_session.send_message,
                        user_text,
                    )
                    dialogue, action = self._parse_response(response.text or "")
                    fallback_chunker = ResponseChunker()
                    fallback_chunks = fallback_chunker.feed(dialogue)
                    fallback_chunks.extend(fallback_chunker.flush())
                    for speech_chunk in fallback_chunks:
                        await self._queue_stream_chunk(
                            turn_id,
                            emitted_chunks,
                            speech_chunk,
                            metrics,
                        )
                        emitted_chunks += 1
                except Exception:
                    logger.exception("Full response fallback also failed.")

            if emitted_chunks == 0:
                await self._queue_stream_chunk(
                    turn_id,
                    0,
                    "I'm having trouble connecting to my cognitive pathways right now.",
                    metrics,
                )

            metrics["full_response_done_at"] = time.monotonic()
            await self._publish_and_yield(
                "AI_SPEAK_STREAM_END",
                {"turn_id": turn_id, "metrics": metrics},
            )
            if action and action.upper() != "IDLE":
                await self._publish_and_yield("MOTION_CMD", action)
            return True

    async def _queue_stream_chunk(
        self,
        turn_id: int,
        sequence: int,
        text: str,
        metrics: dict,
    ) -> None:
        queued_at = time.monotonic()
        logger.info(
            "TTS chunk queued: turn=%s sequence=%s queued_at=%.6f text=%r",
            turn_id,
            sequence,
            queued_at,
            text,
        )
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

    async def _handle_full_response(self, user_text: str) -> None:
        try:
            response = await asyncio.to_thread(
                self.chat_session.send_message,
                user_text,
            )
            dialogue, action = self._parse_response(response.text or "")

            if dialogue:
                await self._publish_and_yield("AI_SPEAK", dialogue)
            if action and action.upper() != "IDLE":
                await self._publish_and_yield("MOTION_CMD", action)

        except json.JSONDecodeError as exc:
            logger.error("Failed to decode Gemini JSON response: %s", exc)
            await self._publish_and_yield(
                "AI_SPEAK",
                "I had a brain glitch and couldn't process that properly.",
            )
        except Exception as exc:
            logger.error("Error handling user input with Gemini API: %s", exc)
            await self._publish_and_yield(
                "AI_SPEAK",
                "I'm having trouble connecting to my cognitive pathways right now.",
            )

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
