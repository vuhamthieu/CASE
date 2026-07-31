import unittest
from collections import deque

from src.wakeword_listener import WakeWordListener


class WakeConfirmationTests(unittest.TestCase):
    def test_confirmation_uses_window_max_for_strong_threshold(self):
        listener = WakeWordListener.__new__(WakeWordListener)
        listener.threshold = 0.995
        listener.strong_threshold = 0.998
        listener.min_hits = 3
        listener.hit_window_sec = 0.7
        listener._hit_history = {}

        listener._update_confirmation_window("hey_case_v2", 0.999, 1.00)
        listener._update_confirmation_window("hey_case_v2", 0.996, 1.08)
        result = listener._update_confirmation_window("hey_case_v2", 0.996, 1.16)

        self.assertEqual(result["hit_count"], 3.0)
        self.assertGreaterEqual(result["window_max"], 0.998)
        self.assertEqual(result["confirmed"], 1.0)

    def test_confirmation_rejects_old_strong_hit_outside_window(self):
        listener = WakeWordListener.__new__(WakeWordListener)
        listener.threshold = 0.995
        listener.strong_threshold = 0.998
        listener.min_hits = 2
        listener.hit_window_sec = 0.7
        listener._hit_history = {"hey_case_v2": deque([(1.00, 0.999)])}

        result = listener._update_confirmation_window("hey_case_v2", 0.996, 1.80)

        self.assertEqual(result["hit_count"], 1.0)
        self.assertLess(result["window_max"], 0.998)
        self.assertEqual(result["confirmed"], 0.0)


if __name__ == "__main__":
    unittest.main()
