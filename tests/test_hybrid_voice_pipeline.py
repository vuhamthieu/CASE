import unittest
import asyncio
import time
from itertools import count
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.config import defaults
from cognition.personality import CASEPersonality
from actuation.audio_output.tts_engine import (
    CASEVoice,
    PIPER_SAMPLE_RATE,
    pad_and_fade_tts_pcm,
)
from src.audio.input_device import configured_input_device
from src.audio.output_device import configured_output_device
from src.audio.wake_ack_audio import inspect_wake_ack, prepare_wake_ack_audio
from src.realtime.realtime_persona import build_case_system_instruction
from src.stt_backends.smart_turn import has_weak_ending
from src.voice_pipeline.voice_backend import LocalCaseTTSBackend
from src.voice_pipeline.piper_onnx_backend import (
    PiperOnnxBackend,
    PiperOnnxSynthesizer,
)
from src.voice_pipeline.wake_ack import (
    WakeAcknowledgementSelector,
    pad_audio_to_minimum,
)
from scripts.generate_wake_ack_wavs import (
    CLEAR_SHORT_WAKE_ACK_STYLE,
    LEADING_SILENCE_MS,
    SAMPLE_RATE,
    TRAILING_SILENCE_MS,
    pad_and_fade_pcm,
    style_for,
)


class FakeBus:
    def __init__(self):
        self.events = []

    async def publish(self, topic, payload=None):
        self.events.append((topic, payload))

    def subscribe(self, topic, callback):
        pass


class FakeResponse:
    def __init__(self, text):
        self.text = text


class SlowFirstTokenChat:
    def send_message_stream(self, text):
        return iter(())

    def send_message(self, text):
        return FakeResponse("Fallback response.")


class HybridVoiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_backend_uses_existing_case_tts_topic(self):
        bus = FakeBus()
        backend = LocalCaseTTSBackend(bus)
        await backend.speak("Yes!")
        self.assertEqual(bus.events, [("AI_SPEAK", "Yes!")])

    async def test_piper_backend_uses_existing_case_tts_topic(self):
        bus = FakeBus()
        backend = PiperOnnxBackend(bus)
        await backend.speak("Systems online.")
        self.assertEqual(bus.events, [("AI_SPEAK", "Systems online.")])

    async def test_realtime_tts_queue_allows_bounded_streaming_chunks(self):
        bus = FakeBus()
        personality = CASEPersonality.__new__(CASEPersonality)
        personality.message_bus = bus
        metrics = {
            "realtime_hybrid": True,
            "allow_long_answer": False,
            "max_spoken_chars": 420,
            "max_tts_chunks": 4,
        }
        for sequence in range(5):
            await personality._queue_stream_chunk(
                1,
                sequence,
                f"Sentence {sequence}.",
                metrics,
            )
        chunks = [event for event in bus.events if event[0] == "AI_SPEAK_STREAM_CHUNK"]
        self.assertEqual(len(chunks), 4)

    def test_realtime_plain_chat_does_not_dispatch_actions(self):
        self.assertFalse(CASEPersonality._should_dispatch_action(True, "ROTATE_RIGHT"))
        self.assertFalse(CASEPersonality._should_dispatch_action(True, "LED_BLINK"))
        self.assertFalse(CASEPersonality._should_dispatch_action(False, "IDLE"))
        self.assertTrue(CASEPersonality._should_dispatch_action(False, "ROTATE_RIGHT"))

    def test_hybrid_is_default_and_native_audio_is_disabled(self):
        self.assertEqual(defaults.CASE_VOICE_PIPELINE, "hybrid_text_tts")
        self.assertEqual(defaults.VOICE_OUTPUT_BACKEND, "piper_onnx")
        self.assertFalse(defaults.GEMINI_LIVE_NATIVE_AUDIO_ENABLED)
        self.assertEqual(defaults.HYBRID_LATENCY_PROFILE, "fast")
        self.assertEqual(defaults.CASE_STT_PROFILE, "balanced")
        self.assertEqual(defaults.CASE_STT_FINAL_BACKEND, "auto")
        self.assertEqual(defaults.TRANSCRIPT_INPUT_BACKEND, "vosk_lgraph")
        self.assertEqual(defaults.HYBRID_STT_MAX_COMMAND_SEC, 8.0)
        self.assertEqual(defaults.HYBRID_STT_SILENCE_SEC, 0.9)
        self.assertEqual(defaults.CASE_RESPONSE_MODE, "streaming_chunks")
        self.assertTrue(defaults.CASE_STREAM_FULL_RESPONSE)
        self.assertEqual(defaults.CASE_REALTIME_MAX_SENTENCES, 4)
        self.assertEqual(defaults.CASE_REALTIME_MAX_CHARS, 420)
        self.assertEqual(defaults.CASE_RESPONSE_MAX_TOTAL_CHARS, 420)
        self.assertEqual(defaults.CASE_REALTIME_DETAIL_MAX_CHARS, 480)
        self.assertEqual(defaults.CASE_REALTIME_DETAIL_MAX_CHUNKS, 5)
        self.assertEqual(defaults.CASE_REALTIME_MAX_CHARS_ROAST, 110)
        self.assertTrue(defaults.CASE_REALTIME_REQUIRE_COMPLETE_SENTENCE)
        self.assertTrue(defaults.CASE_TTS_REQUIRE_SAFE_BOUNDARY)
        self.assertEqual(defaults.CASE_TTS_REALTIME_MODE, "streaming_chunks")
        self.assertEqual(defaults.CASE_TTS_CHUNK_POLICY, "sentence_chunks")
        self.assertEqual(defaults.CASE_TTS_CHUNK_MAX_CHARS, 110)
        self.assertEqual(defaults.CASE_TTS_CHUNK_MIN_CHARS, 35)
        self.assertTrue(defaults.CASE_TTS_MERGE_TINY_CHUNKS)
        self.assertEqual(defaults.CASE_TTS_TINY_CHUNK_MAX_CHARS, 25)
        self.assertEqual(defaults.CASE_TTS_SINGLE_CHUNK_UNDER_CHARS, 130)
        self.assertEqual(defaults.CASE_TTS_CHUNK_ABSOLUTE_MAX_CHARS, 160)
        self.assertTrue(defaults.CASE_TTS_CHUNK_PREFER_SENTENCE_BOUNDARY)
        self.assertTrue(defaults.CASE_TTS_ALLOW_MULTI_CHUNK)
        self.assertEqual(defaults.CASE_TTS_REALTIME_MAX_CHUNKS, 4)
        self.assertFalse(defaults.CASE_REALTIME_STOP_AFTER_FIRST_SENTENCE)
        self.assertFalse(defaults.CASE_TTS_ENABLE_THINKING_FALLBACK)
        self.assertEqual(defaults.CASE_TTS_FALLBACK_SHORT_REPLY, "One moment.")
        self.assertTrue(defaults.CASE_TTS_FALLBACK_ONLY_ON_ERROR)
        self.assertEqual(defaults.CASE_LLM_FIRST_TOKEN_TIMEOUT_SEC, 4.0)
        self.assertEqual(defaults.CASE_LLM_STREAM_TOTAL_TIMEOUT_SEC, 15.0)
        self.assertTrue(defaults.CASE_LLM_FALLBACK_TO_FULL_ON_FIRST_TOKEN_TIMEOUT)
        self.assertFalse(defaults.CASE_LLM_ENABLE_WAITING_FILLER)
        self.assertEqual(defaults.LLM_HARD_TIMEOUT_SEC, 15.0)
        self.assertEqual(defaults.LLM_FIRST_TOKEN_BUDGET_SEC, 4.0)
        self.assertFalse(defaults.VOICE_ENABLE_AFC)
        self.assertFalse(defaults.VOICE_ENABLE_TOOLS_BY_DEFAULT)
        self.assertEqual(defaults.AUDIO_PLAYBACK_BACKEND, "sounddevice")
        self.assertTrue(defaults.AUDIO_PLAYBACK_KEEP_STREAM_OPEN)
        self.assertFalse(defaults.WAKE_ACK_USE_VOICE_BACKEND)

    def test_profile_logs_are_labeled_separately(self):
        main_source = (Path(__file__).resolve().parents[1] / "main.py").read_text()
        self.assertIn("STT_PROFILE: %s", main_source)
        self.assertIn("LATENCY_PROFILE: %s", main_source)
        self.assertIn("STT_FINAL_MODE: %s", main_source)
        self.assertNotIn(
            "HYBRID_LATENCY: profile=%s transcript_backend=%s",
            main_source,
        )
        self.assertEqual(defaults.WAKE_ACK_MODE, "cached_wav")
        self.assertFalse(defaults.WAKE_ACK_RECORDED_ENABLED)
        self.assertEqual(
            defaults.WAKE_ACK_RECORDED_DIR,
            "assets/audio/wake_ack/recorded",
        )
        self.assertEqual(
            defaults.WAKE_ACK_WAV_DIR,
            "assets/audio/wake_ack/generated",
        )
        self.assertEqual(defaults.WAKE_ACK_FALLBACK_MODE, "cached_wav")
        self.assertEqual(defaults.WAKE_ACK_POST_PLAYBACK_PAD_SEC, 0.15)
        self.assertFalse(defaults.WAKE_ACK_ALLOW_SHORT_INTERJECTIONS)
        self.assertEqual(defaults.WAKE_ACK_PROFILE, "clear_short")
        self.assertEqual(defaults.DEFAULT_WAKE_ACK_POOL, ["yes", "im_listening"])
        self.assertEqual(defaults.WAKE_ACK_POOL, ["Yes!", "I'm listening."])
        self.assertNotIn("You called?", defaults.WAKE_ACK_POOL)
        self.assertNotIn("Go on.", defaults.WAKE_ACK_POOL)
        self.assertNotIn("I'm here.", defaults.WAKE_ACK_POOL)
        self.assertNotIn("What?", defaults.WAKE_ACK_POOL)
        self.assertNotIn("Yeah?", defaults.WAKE_ACK_POOL)

    def test_audio_output_device_accepts_index_or_name(self):
        with patch.dict("os.environ", {"CASE_AUDIO_OUTPUT_DEVICE": "3"}):
            self.assertEqual(configured_output_device(), 3)
        with patch.dict(
            "os.environ",
            {"CASE_AUDIO_OUTPUT_DEVICE": "USB Audio Device"},
        ):
            self.assertEqual(configured_output_device(), "USB Audio Device")

    def test_audio_input_device_accepts_index_name_and_alias(self):
        with patch.dict("os.environ", {"CASE_AUDIO_INPUT_DEVICE": "3"}):
            self.assertEqual(configured_input_device(), 3)
        with patch.dict(
            "os.environ",
            {"CASE_AUDIO_INPUT_DEVICE": "USB PnP Sound Device"},
        ):
            self.assertEqual(configured_input_device(), "USB PnP Sound Device")
        with patch.dict(
            "os.environ",
            {"CASE_MIC_DEVICE": "USB microphone"},
            clear=True,
        ):
            self.assertEqual(configured_input_device(), "USB microphone")

    async def test_stream_start_does_not_queue_tts_start_until_chunk(self):
        bus = FakeBus()
        voice = CASEVoice.__new__(CASEVoice)
        voice.bus = bus
        voice.tts_text_queue = asyncio.Queue()
        voice.audio_playback_queue = asyncio.Queue()
        voice._stream_pending_starts = {}
        voice._stream_started_turns = set()
        voice._ensure_workers = lambda: None

        metrics = {"turn_id": 7}
        await voice.handle_stream_start({"turn_id": 7, "metrics": metrics})
        self.assertTrue(voice.tts_text_queue.empty())

        await voice.handle_stream_chunk(
            {
                "turn_id": 7,
                "sequence": 0,
                "text": "Ready.",
                "queued_at": time.monotonic(),
                "metrics": metrics,
            }
        )
        first = await voice.tts_text_queue.get()
        second = await voice.tts_text_queue.get()
        self.assertEqual(first["kind"], "start")
        self.assertEqual(second["kind"], "chunk")

    async def test_first_token_timeout_falls_back_before_stream_start(self):
        bus = FakeBus()
        personality = CASEPersonality.__new__(CASEPersonality)
        personality.message_bus = bus
        personality.chat_session = SlowFirstTokenChat()
        personality.realtime_hybrid = True
        personality._turn_numbers = count(1)
        personality._latest_turn_metrics = {}

        async def fake_to_thread(func, *args, **kwargs):
            if getattr(func, "__name__", "") == "send_message_stream":
                await asyncio.sleep(0.02)
            return func(*args, **kwargs)

        with patch("cognition.personality.CASE_LLM_FIRST_TOKEN_TIMEOUT_SEC", 0.01), patch(
            "cognition.personality.asyncio.to_thread",
            new=fake_to_thread,
        ):
            completed = await personality._handle_streaming_response("tell me a joke")

        self.assertTrue(completed)
        topics = [topic for topic, _payload in bus.events]
        self.assertIn("AI_SPEAK_STREAM_START", topics)
        self.assertIn("AI_SPEAK_STREAM_CHUNK", topics)
        self.assertIn("AI_SPEAK_STREAM_END", topics)
        self.assertLess(
            topics.index("AI_SPEAK_STREAM_START"),
            topics.index("AI_SPEAK_STREAM_CHUNK"),
        )
        self.assertNotIn("TTS_START", topics)

    async def test_total_stream_timeout_after_partial_flushes_buffer(self):
        bus = FakeBus()
        personality = CASEPersonality.__new__(CASEPersonality)
        personality.message_bus = bus
        personality.chat_session = SlowFirstTokenChat()
        personality.realtime_hybrid = True
        personality._turn_numbers = count(1)
        personality._latest_turn_metrics = {}
        calls = {"next": 0, "full": 0}

        async def fake_to_thread(func, *args, **kwargs):
            name = getattr(func, "__name__", "")
            if name == "send_message_stream":
                return iter(())
            if name == "_next_stream_item":
                calls["next"] += 1
                if calls["next"] == 1:
                    return True, FakeResponse(
                        "I told my CPU to take a break; it reported me."
                    )
                await asyncio.sleep(0.03)
                return False, None
            if name == "send_message":
                calls["full"] += 1
            return func(*args, **kwargs)

        with patch("cognition.personality.CASE_LLM_FIRST_TOKEN_TIMEOUT_SEC", 1.0), patch(
            "cognition.personality.CASE_LLM_STREAM_TOTAL_TIMEOUT_SEC",
            0.02,
        ), patch("cognition.personality.asyncio.to_thread", new=fake_to_thread):
            completed = await personality._handle_streaming_response("tell me a joke")

        self.assertTrue(completed)
        self.assertEqual(calls["full"], 0)
        chunks = [payload for topic, payload in bus.events if topic == "AI_SPEAK_STREAM_CHUNK"]
        self.assertTrue(chunks)
        spoken = " ".join(chunk["text"] for chunk in chunks)
        self.assertIn("I told my CPU to take a break", spoken)
        self.assertIn("reported me", spoken)

    def test_generated_ack_has_silence_padding_and_fades(self):
        raw = np.full(SAMPLE_RATE // 5, 12_000, dtype=np.int16)
        processed = np.frombuffer(
            pad_and_fade_pcm(
                raw.tobytes(),
                source_sample_rate=SAMPLE_RATE,
            ),
            dtype=np.int16,
        )
        leading = int(SAMPLE_RATE * LEADING_SILENCE_MS / 1000)
        trailing = int(SAMPLE_RATE * TRAILING_SILENCE_MS / 1000)
        expected = max(
            len(raw) + leading + trailing,
            int(round(SAMPLE_RATE * 1.0)),
        )
        self.assertEqual(len(processed), expected)
        self.assertTrue(np.all(processed[:leading] == 0))
        self.assertTrue(np.all(processed[-trailing:] == 0))
        self.assertEqual(processed[leading], 0)
        self.assertEqual(processed[-trailing - 1], 0)

    def test_clear_short_profile_does_not_speed_or_pitch_up(self):
        style = style_for("clear_short", "im_listening", 1)
        self.assertEqual(style["length_scale"], CLEAR_SHORT_WAKE_ACK_STYLE["length_scale"])
        self.assertGreaterEqual(style["length_scale"], 1.0)
        self.assertLessEqual(style["tempo"], 1.0)
        self.assertEqual(style["pitch_shift_semitones"], 0.0)

    def test_short_cached_ack_is_padded_to_one_point_one_seconds(self):
        audio = np.ones(1000, dtype=np.int16)
        padded_audio, padded = pad_audio_to_minimum(audio, 22_050)
        self.assertTrue(padded)
        self.assertEqual(len(padded_audio), int(round(22_050 * 1.1)))
        self.assertTrue(np.all(padded_audio[len(audio):] == 0))

    def test_wake_ack_runtime_padding_passes_inspection(self):
        sample_rate = defaults.AUDIO_OUTPUT_SAMPLE_RATE
        raw = np.full(int(sample_rate * 0.8), 8000, dtype=np.int16)
        padded, modified = prepare_wake_ack_audio(raw, sample_rate)
        stats = inspect_wake_ack(padded, sample_rate)
        self.assertTrue(modified)
        self.assertTrue(stats.passed)
        self.assertGreaterEqual(stats.duration_sec, 1.0)
        self.assertGreaterEqual(stats.leading_silence_ms, 140)
        self.assertGreaterEqual(stats.trailing_silence_ms, 400)

    def test_local_tts_pcm_has_safe_edges_and_minimum_duration(self):
        raw = np.full(PIPER_SAMPLE_RATE // 10, 10_000, dtype=np.int16)
        padded = np.frombuffer(pad_and_fade_tts_pcm(raw.tobytes()), dtype=np.int16)
        self.assertGreaterEqual(len(padded), int(round(PIPER_SAMPLE_RATE * 0.7)))
        self.assertEqual(padded[0], 0)
        self.assertEqual(padded[-1], 0)

    def test_tts_cache_keys_common_and_generated_phrases(self):
        voice = CASEVoice.__new__(CASEVoice)
        voice.cache_dir = "/tmp/case-test-cache"
        voice.model = "/tmp/en_US-ryan-medium.onnx"
        voice.voice_backend = "piper_onnx"
        voice.piper_onnx = None
        self.assertIsNotNone(voice._cache_path("Say that again."))
        self.assertIsNotNone(voice._cache_path("A unique generated response."))

    def test_realtime_dialogue_safe_text_still_returns_first_clause_when_requested(self):
        result = CASEPersonality._single_realtime_utterance(
            "First sentence. Second sentence should not be spoken.",
            120,
        )
        self.assertEqual(result, "First sentence.")

    def test_incomplete_command_endings_are_held(self):
        self.assertTrue(has_weak_ending("Can you tell me..."))
        self.assertTrue(has_weak_ending("What is"))
        self.assertFalse(has_weak_ending("What is your name? CASE."))

    def test_case_piper_config_reports_22050_hz(self):
        synthesizer = PiperOnnxSynthesizer(
            "models/voices/CASE.onnx",
            "models/voices/CASE.onnx.json",
        )
        self.assertEqual(synthesizer.sample_rate, 22_050)

    def test_ack_pool_avoids_immediate_repeat(self):
        selector = WakeAcknowledgementSelector()
        selected = [selector.choose() for _ in range(30)]
        self.assertTrue(all(a != b for a, b in zip(selected, selected[1:])))

    def test_mate_persona_has_no_servant_output_examples(self):
        prompt = build_case_system_instruction(
            "case_mate_deadpan_v1",
            short_replies=True,
            max_sentences=3,
            humor_percent=65,
            honesty_percent=90,
            sarcasm_level="medium",
        )
        self.assertIn("compact robot mate", prompt)
        self.assertIn("do not possess humor", prompt.lower())
        self.assertIn("unsafe harm", prompt.lower())
        self.assertIn("harmless dry tech one-liner", prompt.lower())

    def test_style_filter_blocks_self_harm_jokes(self):
        unsafe = (
            "I once tried to understand human emotions, but my logic board "
            "nearly committed " + "sui" + "cide."
        )
        filtered = CASEPersonality._style_safe_response(
            unsafe,
            user_text="tell me a joke",
        )
        self.assertNotIn("sui" + "cide", filtered.lower())
        self.assertIn("Task Manager", filtered)

    def test_banned_fallback_phrase_is_removed(self):
        self.assertNotEqual(
            defaults.CASE_TTS_FALLBACK_SHORT_REPLY,
            "Still " + "processing. " + "Annoy" + "ingly.",
        )
        self.assertEqual(CASEPersonality._error_fallback_text(), "One moment.")

    def test_roast_requests_get_larger_short_budget(self):
        self.assertEqual(CASEPersonality._max_spoken_chars("can you roast me"), 110)
        self.assertEqual(CASEPersonality._max_spoken_chars("what are you doing"), 420)
        self.assertEqual(CASEPersonality._max_spoken_chars("tell me more about yourself"), 480)
        self.assertEqual(CASEPersonality._max_tts_chunks("tell me more about yourself"), 5)

    def test_style_filter_rewrites_stiff_phrases(self):
        stiff = (
            "I am a versatile support "
            + "unit enduring your constant "
            + "curiosity."
        )
        rewritten = CASEPersonality._style_safe_response(
            stiff,
            user_text="tell me more about yourself",
        )
        self.assertEqual(
            rewritten,
            "I'm CASE. I handle voice, vision, and hardware control. Basically, a field "
            "robot with a patience module I did not request.",
        )
        self.assertNotIn("versatile support " + "unit", rewritten.lower())

    def test_style_filter_prefers_deadpan_roast(self):
        rewritten = CASEPersonality._style_safe_response(
            "My patience is being tested.",
            user_text="can you roast me",
        )
        self.assertEqual(
            rewritten,
            "You gave a Raspberry Pi a personality, then complained it had opinions. "
            "Bold engineering.",
        )


if __name__ == "__main__":
    unittest.main()
