import unittest

import numpy as np

from scripts.record_wake_ack_assets import ACK_LINES, prepare_raw_recording


class RecordWakeAckAssetsTests(unittest.TestCase):
    def test_raw_capture_is_not_trimmed_or_peak_normalized(self):
        recording = np.zeros(44_100, dtype=np.float32)
        recording[10_000:18_000] = 0.2 * np.sin(np.linspace(0, 40, 8_000))
        raw = prepare_raw_recording(recording)

        self.assertEqual(len(raw), len(recording))
        self.assertEqual(raw.dtype, np.int16)
        self.assertTrue(raw.flags.c_contiguous)
        self.assertAlmostEqual(
            int(np.max(np.abs(raw.astype(np.int32)))) / 32767.0,
            0.2,
            delta=0.01,
        )

    def test_only_approved_runtime_lines_are_recordable(self):
        self.assertEqual(set(ACK_LINES), {"yes", "im_listening"})


if __name__ == "__main__":
    unittest.main()
