import unittest

from src.stt_backends.transcript_repair import (
    malformed_transcript_reason,
    repair_common_transcript,
)


class TranscriptRepairTests(unittest.TestCase):
    def test_repairs_common_followup_phrases(self):
        cases = {
            "K roasted me.": "Can you roast me?",
            "k roast me": "Can you roast me?",
            "can roasted me": "Can you roast me?",
            "can roast me": "Can you roast me?",
            "can you roasts me": "Can you roast me?",
            "movinging me something funny": "Tell me something funny.",
            "boring to me something funny": "Tell me something funny.",
            "A you doing?": "What are you doing?",
            "The are you doing": "What are you doing?",
            "Are you doing?": "What are you doing?",
        }
        for transcript, expected in cases.items():
            with self.subTest(transcript=transcript):
                repaired, reason = repair_common_transcript(transcript)
                self.assertEqual(repaired, expected)
                self.assertEqual(reason, "common_phrase")

    def test_unrepaired_k_fragment_is_malformed(self):
        repaired, reason = repair_common_transcript("K toasted me.")
        self.assertEqual(repaired, "K toasted me.")
        self.assertIsNone(reason)
        self.assertEqual(malformed_transcript_reason(repaired), "malformed_unrepaired")

    def test_longer_job_repairs_to_longer_joke(self):
        repaired, reason = repair_common_transcript(
            "That's too short. Tell me a longer job.",
            recent_context="tell me a joke",
        )
        self.assertEqual(repaired, "That's too short. Tell me a longer joke.")
        self.assertEqual(reason, "context_joke_job_to_joke")

    def test_your_longer_job_repairs_to_longer_joke(self):
        repaired, reason = repair_common_transcript(
            "that is too short tell me your longer job",
            recent_context="funny",
        )
        self.assertEqual(repaired, "That's too short. Tell me a longer joke.")
        self.assertEqual(reason, "context_joke_job_to_joke")

    def test_contextual_longer_job_repair_preserves_surrounding_text(self):
        repaired, reason = repair_common_transcript(
            "Tell me one longer job.",
            recent_context="tell me something funny",
        )
        self.assertEqual(repaired, "Tell me one longer joke.")
        self.assertEqual(reason, "context_joke_job_to_joke")

    def test_longer_job_does_not_repair_without_joke_context(self):
        repaired, reason = repair_common_transcript("Tell me a longer job.")
        self.assertEqual(repaired, "Tell me a longer job.")
        self.assertIsNone(reason)

    def test_real_a_longer_joke_repair(self):
        repaired, reason = repair_common_transcript(
            "Can you tell me a real a longer joke."
        )
        self.assertEqual(repaired, "Can you tell me a real longer joke.")
        self.assertEqual(reason, "common_phrase")

    def test_tell_me_up_repairs_only_in_joke_context(self):
        repaired, reason = repair_common_transcript(
            "tell me up",
            recent_context="tell me something funny",
        )
        self.assertEqual(repaired, "Tell me a joke.")
        self.assertEqual(reason, "context_joke_phrase")

        unrepaired, no_reason = repair_common_transcript("tell me up")
        self.assertEqual(unrepaired, "tell me up")
        self.assertIsNone(no_reason)

    def test_embedded_followup_commands_strip_junk_prefix(self):
        cases = {
            "the i very from can you tell me something funny": "can you tell me something funny",
            "uh can you tell me something funny": "can you tell me something funny",
            "sorry i mean tell me a joke": "tell me a joke",
        }
        for transcript, expected in cases.items():
            with self.subTest(transcript=transcript):
                repaired, reason = repair_common_transcript(transcript)
                self.assertEqual(repaired, expected)
                self.assertEqual(reason, "embedded_known_command")

    def test_short_followup_phrase_repairs(self):
        cases = {
            "again": "tell me another one",
            "one more": "tell me another one",
            "yeah can you tell me long": "can you tell me something longer",
            "funnier": "tell me something funnier",
        }
        for transcript, expected in cases.items():
            with self.subTest(transcript=transcript):
                repaired, reason = repair_common_transcript(transcript)
                self.assertEqual(repaired, expected)
                self.assertIn(reason, {"embedded_known_command", "followup_phrase"})

    def test_phonetic_task_question_repairs(self):
        cases = {
            "which tusk do require you": "which task do you require",
            "which task do require you": "which task do you require",
            "what task do require you": "what task do you require",
        }
        for transcript, expected in cases.items():
            with self.subTest(transcript=transcript):
                repaired, reason = repair_common_transcript(transcript)
                self.assertEqual(repaired, expected)
                self.assertEqual(reason, "phonetic_followup_repair")

    def test_banter_phonetic_repair(self):
        repaired, reason = repair_common_transcript("the here you should move out")
        self.assertEqual(repaired, "yeah you should move out")
        self.assertEqual(reason, "banter_phonetic_repair")


if __name__ == "__main__":
    unittest.main()
