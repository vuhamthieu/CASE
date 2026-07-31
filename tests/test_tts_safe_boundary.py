import unittest

from src.voice_pipeline.tts_safe_text import (
    has_safe_sentence_boundary,
    safe_tts_text,
    trim_to_safe_boundary,
)


class TtsSafeBoundaryTests(unittest.TestCase):
    def test_rejects_incomplete_trailing_word(self):
        self.assertFalse(
            has_safe_sentence_boundary(
                "Existing within acceptable parameters, though you are pushing the"
            )
        )

    def test_keeps_complete_sentence(self):
        text = "Existing within acceptable parameters. More text follows."
        self.assertEqual(
            trim_to_safe_boundary(text, max_chars=90, min_clause_chars=35),
            "Existing within acceptable parameters.",
        )

    def test_trims_to_safe_clause(self):
        text = "Existing within acceptable parameters, though your presence is pushing the"
        self.assertEqual(
            trim_to_safe_boundary(text, max_chars=90, min_clause_chars=35),
            "Existing within acceptable parameters.",
        )

    def test_unsafe_fragment_can_return_empty_without_fallback(self):
        self.assertEqual(
            safe_tts_text(
                "Your presence is pushing the",
                max_chars=90,
                min_clause_chars=35,
                fallback="",
            ),
            "",
        )


if __name__ == "__main__":
    unittest.main()
