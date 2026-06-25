import unittest

from src.stt_backends.transcript_selection import (
    choose_final_transcript,
    dedupe_repeated_transcript,
)


class TranscriptSelectionTests(unittest.TestCase):
    def test_sensevoice_replaces_vosk_candidate(self):
        status = {"sensevoice_available": True, "sensevoice_error": None}
        selected = choose_final_transcript(
            "you tell me a joke",
            "You tell me a joke.",
            status,
        )
        self.assertEqual(selected, "You tell me a joke.")
        self.assertEqual(status["selected_source"], "sensevoice")

    def test_vosk_is_used_when_sensevoice_fails(self):
        status = {"sensevoice_available": True, "sensevoice_error": "decode failed"}
        selected = choose_final_transcript("tell me more", "", status)
        self.assertEqual(selected, "tell me more")
        self.assertEqual(status["selected_source"], "vosk_fallback")

    def test_deduplicates_identical_candidates(self):
        self.assertEqual(
            dedupe_repeated_transcript(
                "you tell me a joke You tell me a joke."
            ),
            "You tell me a joke.",
        )

    def test_prefers_fuller_repeated_candidate(self):
        self.assertEqual(
            dedupe_repeated_transcript("are you doing How are you doing?"),
            "How are you doing?",
        )
        self.assertEqual(dedupe_repeated_transcript("case Case."), "Case.")


if __name__ == "__main__":
    unittest.main()
