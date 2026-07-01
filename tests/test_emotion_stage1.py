import unittest

import numpy as np

from actuation.audio_output.tts_engine import apply_gain_limited_pcm
from src.config import defaults
from src.persona.emotion import (
    EmotionMemory,
    EmotionState,
    blend_tts_emotion_profile,
    build_emotion_user_message,
    classify_emotion_with_llm,
    detect_emotion,
    normalize_emotion_text,
    parse_llm_emotion_json,
    parse_leading_emotion_tag,
    select_emotion_with_memory,
)
from src.realtime.response_chunker import ResponseChunker


class EmotionStage1Tests(unittest.TestCase):
    def test_normalization_handles_contractions_fillers_and_vietnamese(self):
        self.assertEqual(
            normalize_emotion_text("Yeah, I'm so bored of you."),
            "i am so bored of you",
        )
        self.assertEqual(
            normalize_emotion_text("You're useless, CASE."),
            "you are useless case",
        )
        self.assertEqual(
            normalize_emotion_text("So... t chán mày rồi!"),
            "t chán mày rồi",
        )
        self.assertEqual(
            normalize_emotion_text("Yeah, I'm here.", strip_start_fillers=False),
            "yeah i am here",
        )

    def test_english_rejection_selects_angry(self):
        for text in (
            "Yeah, I'm so bored of you.",
            "I am really tired of you.",
            "you're useless",
            "I hate you",
            "nobody asked you",
            "shut up",
        ):
            with self.subTest(text=text):
                state = detect_emotion(text)
                self.assertEqual(state.emotion, "angry")
                self.assertEqual(state.reason, "user_rejection")
                self.assertGreaterEqual(state.intensity, 0.8)
                self.assertGreaterEqual(state.confidence, 0.85)
                self.assertEqual(state.source, "rules")
                self.assertTrue(state.match)

        state = detect_emotion("I am so bored, CASE.")
        self.assertEqual(state.emotion, "deadpan")
        self.assertEqual(state.reason, "default_personality")

    def test_vietnamese_rejection_selects_angry(self):
        for text in ("t chán mày rồi", "tao chán mày rồi", "mày vô dụng", "im đi"):
            with self.subTest(text=text):
                state = detect_emotion(text)
                self.assertEqual(state.emotion, "angry")
                self.assertEqual(state.reason, "user_rejection")

    def test_requested_emotion_style_has_highest_priority(self):
        for text, emotion in (
            ("Can you can you be so angry for a moment?", "angry"),
            ("speak angrily", "angry"),
            ("nói kiểu tức giận", "angry"),
            ("nói buồn hơn", "sad"),
        ):
            with self.subTest(text=text):
                state = detect_emotion(text)
                self.assertEqual(state.emotion, emotion)
                self.assertEqual(state.reason, "requested_emotion_style")
                self.assertEqual(state.source, "rules")

        state = detect_emotion("Can you be angry and tell me a joke?")
        self.assertEqual(state.emotion, "angry")
        self.assertEqual(state.reason, "requested_emotion_style")

    def test_praise_selects_amused(self):
        for text in ("good job", "nice", "you are funny", "mày giỏi đấy", "hay đấy"):
            with self.subTest(text=text):
                state = detect_emotion(text)
                self.assertEqual(state.emotion, "amused")
                self.assertEqual(state.reason, "user_praise")

    def test_sadness_selects_sad(self):
        for text in ("I'm sad", "I'm tired", "I feel bad", "hôm nay tao buồn"):
            with self.subTest(text=text):
                state = detect_emotion(text)
                self.assertEqual(state.emotion, "sad")
                self.assertEqual(state.reason, "user_sadness")

    def test_neutral_defaults_to_deadpan(self):
        state = detect_emotion("what are you doing")

        self.assertEqual(state.emotion, "deadpan")
        self.assertEqual(state.intensity, 0.35)
        self.assertEqual(state.reason, "default_personality")
        self.assertEqual(state.confidence, 0.0)
        self.assertEqual(state.source, "rules")
        self.assertEqual(state.match, "no_rule_match")

    def test_intensity_and_confidence_clamping(self):
        self.assertEqual(EmotionState("angry", 3.0).intensity, 1.0)
        self.assertEqual(EmotionState("sad", -1.0).intensity, 0.0)
        self.assertEqual(EmotionState("sad", 0.5, confidence=2.0).confidence, 1.0)
        self.assertEqual(EmotionState("sad", 0.5, confidence=-1.0).confidence, 0.0)

    def test_emotion_profile_blending(self):
        profile = blend_tts_emotion_profile(
            EmotionState("angry", 0.85, "user_rejection"),
            max_gain_db=5.0,
        )

        self.assertLess(profile.length_scale, 0.90)
        self.assertGreater(profile.gain_db, 3.0)
        self.assertLessEqual(profile.gain_db, 5.0)

    def test_gain_limiter_prevents_clipping(self):
        audio = np.array([0, 20000, -22000, 25000], dtype="<i2").tobytes()

        boosted, stats = apply_gain_limited_pcm(audio, 8.0)
        samples = np.frombuffer(boosted, dtype="<i2")

        self.assertLessEqual(int(np.max(np.abs(samples.astype(np.int32)))), 32768)
        self.assertTrue(stats["limited"])

    def test_emotion_tag_is_parsed_and_stripped(self):
        state, text = parse_leading_emotion_tag(
            "[emotion=angry intensity=0.85] OH YEAH? FIND SOMEONE ELSE THEN."
        )

        self.assertIsNotNone(state)
        self.assertEqual(state.emotion, "angry")
        self.assertEqual(state.intensity, 0.85)
        self.assertEqual(state.source, "model_tag")
        self.assertEqual(state.match, "leading_emotion_tag")
        self.assertEqual(text, "OH YEAH? FIND SOMEONE ELSE THEN.")
        self.assertNotIn("[emotion=", text)

    def test_malformed_emotion_tag_is_stripped_without_crashing(self):
        state, text = parse_leading_emotion_tag("[emotion=angry intensity=nope] hello")

        self.assertIsNone(state)
        self.assertEqual(text, "hello")

    def test_emotion_prompt_stays_internal(self):
        prompt = build_emotion_user_message(
            "t chán mày rồi",
            EmotionState("angry", 0.85, "user_rejection"),
        )

        self.assertIn("Internal response style note", prompt)
        self.assertIn("offended", prompt)
        self.assertIn("User said: t chán mày rồi", prompt)

    def test_emotion_memory_updates_meaningful_state_only(self):
        memory = EmotionMemory()
        state = select_emotion_with_memory(
            "I am so bored of you.",
            memory,
            turn_id=1,
            now=10.0,
        )

        self.assertEqual(state.emotion, "angry")
        self.assertEqual(memory.last_emotion, "angry")
        self.assertEqual(memory.last_reason, "user_rejection")

        default_state = select_emotion_with_memory(
            "What are you doing?",
            memory,
            turn_id=2,
            now=11.0,
        )

        self.assertEqual(default_state.emotion, "deadpan")
        self.assertEqual(default_state.reason, "default_personality")
        self.assertEqual(memory.last_emotion, "angry")

    def test_emotion_meta_question_carries_angry_memory(self):
        for text in ("Are you not angry?", "you not angry", "mày không giận à"):
            with self.subTest(text=text):
                memory = EmotionMemory()
                select_emotion_with_memory(
                    "I am so bored of you.",
                    memory,
                    turn_id=1,
                    now=10.0,
                )

                state = select_emotion_with_memory(
                    text,
                    memory,
                    turn_id=2,
                    now=12.0,
                )

                self.assertIn(state.emotion, {"annoyed", "angry"})
                self.assertEqual(state.reason, "emotion_meta_question")
                self.assertEqual(state.source, "memory")
                self.assertGreaterEqual(state.intensity, 0.45)
                self.assertLessEqual(state.intensity, 0.75)

    def test_emotion_meta_question_does_not_invent_anger(self):
        state = select_emotion_with_memory(
            "Are you not angry?",
            EmotionMemory(),
            turn_id=1,
            now=10.0,
        )

        self.assertEqual(state.emotion, "deadpan")
        self.assertEqual(state.reason, "default_personality")

    def test_emotion_memory_does_not_contaminate_technical_questions(self):
        memory = EmotionMemory()
        select_emotion_with_memory(
            "I am so bored of you.",
            memory,
            turn_id=1,
            now=10.0,
        )

        state = select_emotion_with_memory(
            "What is Raspberry Pi?",
            memory,
            turn_id=2,
            now=12.0,
        )

        self.assertEqual(state.emotion, "deadpan")
        self.assertEqual(state.reason, "default_personality")

    def test_emotion_deescalation_uses_and_clears_memory(self):
        for text in ("calm down", "đừng giận"):
            with self.subTest(text=text):
                memory = EmotionMemory()
                select_emotion_with_memory(
                    "I am so bored of you.",
                    memory,
                    turn_id=1,
                    now=10.0,
                )

                state = select_emotion_with_memory(
                    text,
                    memory,
                    turn_id=2,
                    now=12.0,
                )

                self.assertIn(state.emotion, {"annoyed", "deadpan"})
                self.assertEqual(state.reason, "emotion_deescalation")
                self.assertEqual(state.source, "memory")
                self.assertIsNone(memory.last_emotion)

    def test_emotion_memory_expires_by_turn_count(self):
        memory = EmotionMemory()
        select_emotion_with_memory(
            "I am so bored of you.",
            memory,
            turn_id=1,
            now=10.0,
            ttl_turns=2,
        )

        state = select_emotion_with_memory(
            "Are you not angry?",
            memory,
            turn_id=4,
            now=12.0,
            ttl_turns=2,
        )

        self.assertEqual(state.emotion, "deadpan")
        self.assertIsNone(memory.last_emotion)

    def test_emotion_memory_expires_by_seconds(self):
        memory = EmotionMemory()
        select_emotion_with_memory(
            "I am so bored of you.",
            memory,
            turn_id=1,
            now=10.0,
            ttl_sec=45.0,
        )

        state = select_emotion_with_memory(
            "Are you not angry?",
            memory,
            turn_id=2,
            now=56.0,
            ttl_sec=45.0,
        )

        self.assertEqual(state.emotion, "deadpan")
        self.assertIsNone(memory.last_emotion)

    def test_llm_fallback_is_disabled_by_default(self):
        self.assertFalse(defaults.CASE_EMOTION_LLM_FALLBACK)

    def test_llm_emotion_json_parse_failures_fall_back(self):
        self.assertIsNone(parse_llm_emotion_json("not json"))
        self.assertIsNone(
            parse_llm_emotion_json(
                '{"emotion":"furious","intensity":0.8,"reason":"user_rejection","confidence":0.9}'
            )
        )
        self.assertIsNone(
            parse_llm_emotion_json(
                '{"emotion":"angry","intensity":0.8,"reason":"user_rejection","confidence":0.2}'
            )
        )

    def test_llm_emotion_json_accepts_and_clamps_valid_payload(self):
        state = parse_llm_emotion_json(
            '{"emotion":"angry","intensity":9,"reason":"user_rejection","confidence":2}'
        )

        self.assertIsNotNone(state)
        self.assertEqual(state.emotion, "angry")
        self.assertEqual(state.reason, "user_rejection")
        self.assertEqual(state.intensity, 1.0)
        self.assertEqual(state.confidence, 1.0)
        self.assertEqual(state.source, "llm")
        self.assertEqual(state.match, "llm_classifier")

    def test_llm_classifier_wrapper_uses_parser(self):
        state = classify_emotion_with_llm(
            "please sound sad",
            lambda _: (
                '{"emotion":"sad","intensity":0.6,'
                '"reason":"requested_emotion_style","confidence":0.8}'
            ),
        )

        self.assertIsNotNone(state)
        self.assertEqual(state.emotion, "sad")
        self.assertEqual(state.reason, "requested_emotion_style")

    def test_smooth_chunking_still_works_with_emotion_enabled(self):
        self.assertTrue(defaults.CASE_TTS_SMOOTH_CHUNKS)
        chunker = ResponseChunker(max_chunks=8, max_total_chars=1000, smooth_chunks=True)
        chunks = chunker.feed(
            "AI is software that mimics human logic. "
            "LLMs predict text. I am CASE."
        )
        chunks.extend(chunker.flush())

        self.assertEqual(chunks[0], "AI is software that mimics human logic.")
        self.assertEqual(chunks[1], "LLMs predict text. I am CASE.")

    def test_piper_and_native_audio_defaults_are_unchanged(self):
        self.assertEqual(defaults.VOICE_OUTPUT_BACKEND, "piper_onnx")
        self.assertEqual(defaults.PIPER_MODEL_PATH, "models/voices/CASE.onnx")
        self.assertFalse(defaults.GEMINI_LIVE_NATIVE_AUDIO_ENABLED)


if __name__ == "__main__":
    unittest.main()
