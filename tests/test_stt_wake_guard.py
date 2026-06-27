import queue
import threading
import time
import unittest
from importlib.util import find_spec
from collections import deque

import numpy as np

STT_DEPS_AVAILABLE = find_spec("sounddevice") is not None and find_spec("vosk") is not None
if STT_DEPS_AVAILABLE:
    from perception.audio import stt_engine
else:
    stt_engine = None


class FakeWakeModel:
    def __init__(self, score: float):
        self.score = score

    def predict(self, frame):
        return {"hey_case_v2": self.score}


def make_engine():
    engine = stt_engine.STTEngine.__new__(stt_engine.STTEngine)
    engine.wakeword_name = "hey_case_v2"
    engine.wakeword_model = FakeWakeModel(0.999)
    engine.wake_threshold = 0.995
    engine.wake_strong_threshold = 0.998
    engine.wake_min_hits = 1
    engine.wake_hit_window_sec = 0.7
    engine.wake_cooldown_sec = 2.0
    engine._last_wake_time = 0.0
    engine._last_tts_end_time = time.monotonic()
    engine._wake_suppressed_until = 0.0
    engine._wake_suppression_reason = ""
    engine._wake_hits = deque()
    engine._wake_scores = deque()
    engine._state = stt_engine.STATE_IDLE
    engine._state_lock = threading.Lock()
    engine._tts_active_count = 0
    engine.audio_queue = queue.Queue()
    engine._last_published_transcript = ""
    return engine


@unittest.skipUnless(STT_DEPS_AVAILABLE, "STT wake guard tests require sounddevice and vosk")
class SttWakeGuardTests(unittest.TestCase):
    def test_wake_rejected_during_post_tts_cooldown(self):
        engine = make_engine()
        engine._wake_suppressed_until = time.monotonic() + 3.0
        engine._wake_suppression_reason = "post_tts_cooldown"
        frame = np.zeros(stt_engine.WAKE_FRAME_SAMPLES, dtype=np.int16)

        with self.assertLogs(level="INFO") as logs:
            confirmed, score = engine._predict_wake_frame(frame)

        self.assertFalse(confirmed)
        self.assertGreaterEqual(score, 0.998)
        self.assertEqual(len(engine._wake_hits), 0)
        self.assertTrue(
            any("reason=post_tts_cooldown" in line for line in logs.output)
        )

    def test_short_followup_to_idle_resets_wake_history_and_starts_guard(self):
        engine = make_engine()
        engine._state = stt_engine.STATE_SHORT_FOLLOW_UP
        engine._wake_hits.append((time.monotonic(), 0.999))
        engine._wake_scores.append((time.monotonic(), 0.999))

        with self.assertLogs(level="INFO") as logs:
            engine._transition(stt_engine.STATE_IDLE)

        self.assertEqual(len(engine._wake_hits), 0)
        self.assertEqual(len(engine._wake_scores), 0)
        self.assertGreater(engine._wake_suppressed_until, time.monotonic())
        self.assertTrue(
            any(
                "WAKE_DETECTOR_RESET: reason=state_transition "
                "from=SHORT_FOLLOW_UP to=IDLE" in line
                for line in logs.output
            )
        )

    def test_repairs_malformed_common_command(self):
        engine = make_engine()
        self.assertEqual(
            engine._repair_transcript("A you doing?"),
            "What are you doing?",
        )
        self.assertEqual(
            engine._repair_transcript("The are you doing"),
            "What are you doing?",
        )
        self.assertEqual(
            engine._repair_transcript("Are you doing?"),
            "What are you doing?",
        )

    def test_one_word_garbage_is_rejected(self):
        engine = make_engine()
        self.assertEqual(engine._transcript_reject_reason("Yeah."), "too_few_words")
        self.assertEqual(engine._transcript_reject_reason("The."), "too_few_words")
        self.assertEqual(engine._transcript_reject_reason("A."), "too_short")

    def test_non_english_and_punctuation_garbage_rejected(self):
        engine = make_engine()
        self.assertEqual(
            engine._transcript_reject_reason("买咬."),
            "non_english_garbage",
        )
        self.assertEqual(engine._transcript_reject_reason("。"), "punctuation_only")

    def test_unclear_followups_are_rejected(self):
        engine = make_engine()
        for text in ("oh right boy", "the ideology", "a very funny"):
            with self.subTest(text=text):
                self.assertEqual(
                    engine._transcript_reject_reason(text, followup=True),
                    "followup_unclear",
                )

    def test_clear_followup_questions_are_accepted_with_leading_fillers(self):
        engine = make_engine()
        examples = (
            "Yeah, what is your current humor percentage.",
            "What is your current humor percentage?",
            "Can you tell me another joke?",
            "Tell me a longer joke.",
            "Do you know who I am?",
            "Are you still there?",
            "yeah can you tell me long",
        )
        for text in examples:
            with self.subTest(text=text):
                repaired = engine._repair_transcript(text)
                self.assertIsNone(engine._transcript_reject_reason(repaired, followup=True))

    def test_followup_filler_stripping_for_intent_classification(self):
        engine = make_engine()
        self.assertTrue(
            engine._has_clear_followup_intent("yeah what is your current humor percentage")
        )

    def test_feedback_followups_are_accepted(self):
        engine = make_engine()
        for text in (
            "that is very funny",
            "Very funny.",
            "not funny",
            "boring",
            "haha",
            "good one",
            "yeah that's your problem",
            "that's your problem",
            "not my problem",
        ):
            with self.subTest(text=text):
                self.assertIsNone(engine._transcript_reject_reason(text, followup=True))

    def test_more_request_followups_are_accepted(self):
        engine = make_engine()
        for text in ("again", "one more", "another one", "tell me more", "make it longer", "funnier"):
            with self.subTest(text=text):
                repaired = engine._repair_transcript(text)
                self.assertIsNone(engine._transcript_reject_reason(repaired, followup=True))

    def test_embedded_followup_command_is_repaired_and_accepted(self):
        engine = make_engine()
        repaired = engine._repair_transcript(
            "the i very from can you tell me something funny"
        )
        self.assertEqual(repaired, "can you tell me something funny")
        self.assertIsNone(engine._transcript_reject_reason(repaired, followup=True))

    def test_wake_echo_and_empty_followups_are_rejected(self):
        engine = make_engine()
        self.assertIn(
            engine._transcript_reject_reason("case", followup=True),
            {"garbage", "wake_word_only"},
        )
        self.assertEqual(engine._transcript_reject_reason("", followup=True), "empty")

    def test_joke_context_repairs_longer_job(self):
        engine = make_engine()
        engine._last_published_transcript = "tell me something funny"
        self.assertEqual(
            engine._repair_transcript("Tell me your longer job."),
            "Tell me a longer joke.",
        )


if __name__ == "__main__":
    unittest.main()
