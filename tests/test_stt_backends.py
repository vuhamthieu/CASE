import unittest

from scripts.benchmark_stt_backends import word_error_rate
from src.stt_backends.smart_turn import has_weak_ending


class OfflineSttHelpersTests(unittest.TestCase):
    def test_word_error_rate(self):
        self.assertEqual(word_error_rate("take a picture", "take a picture"), 0.0)
        self.assertAlmostEqual(
            word_error_rate("take a picture", "take picture"),
            1 / 3,
        )

    def test_weak_endings(self):
        self.assertTrue(has_weak_ending("can you tell me"))
        self.assertTrue(has_weak_ending("I want you to..."))
        self.assertFalse(has_weak_ending("can you tell me a joke"))


if __name__ == "__main__":
    unittest.main()
