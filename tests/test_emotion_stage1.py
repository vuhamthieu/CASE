import unittest

import numpy as np

from actuation.audio_output.tts_engine import apply_gain_limited_pcm
from src.config import defaults
from src.persona.emotion import (
    EmotionState,
    blend_tts_emotion_profile,
    build_emotion_user_message,
    detect_emotion,
    parse_leading_emotion_tag,
)
from src.realtime.response_chunker import ResponseChunker


class EmotionStage1Tests(unittest.TestCase):
    def test_english_rejection_selects_angry(self):
        for text in ("I'm bored of you", "you are boring", "I hate you", "shut up"):
            with self.subTest(text=text):
                state = detect_emotion(text)
                self.assertEqual(state.emotion, "angry")
                self.assertEqual(state.reason, "user_rejection")
                self.assertGreaterEqual(state.intensity, 0.8)

    def test_vietnamese_rejection_selects_angry(self):
        for text in ("t chán mày rồi", "tao chán mày rồi", "mày vô dụng"):
            with self.subTest(text=text):
                state = detect_emotion(text)
                self.assertEqual(state.emotion, "angry")
                self.assertEqual(state.reason, "user_rejection")

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

    def test_intensity_clamping(self):
        self.assertEqual(EmotionState("angry", 3.0).intensity, 1.0)
        self.assertEqual(EmotionState("sad", -1.0).intensity, 0.0)

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
