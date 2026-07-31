import unittest

import numpy as np

from src.audio.recorded_wake_ack_processing import (
    WAKE_ACK_PROCESSING_PRESETS,
    process_recorded_wake_ack,
)
from src.audio.wake_ack_audio import inspect_wake_ack


class RecordedWakeAckProcessingTests(unittest.TestCase):
    def test_clean_processing_removes_dc_and_adds_runtime_edges(self):
        sample_rate = 22_050
        raw = np.full(sample_rate * 2, 0.03, dtype=np.float32)
        raw[8_000:15_000] += 0.2 * np.sin(np.linspace(0, 60, 7_000))
        processed = process_recorded_wake_ack(
            raw,
            sample_rate,
            WAKE_ACK_PROCESSING_PRESETS["clean"],
        )
        stats = inspect_wake_ack(processed, sample_rate)
        self.assertEqual(processed.dtype, np.int16)
        self.assertTrue(processed.flags.c_contiguous)
        self.assertGreaterEqual(stats.leading_silence_ms, 119)
        self.assertGreaterEqual(stats.trailing_silence_ms, 349)
        self.assertAlmostEqual(stats.peak_dbfs, -3.0, delta=0.25)
        self.assertFalse(stats.clipped)

    def test_requested_presets_exist(self):
        self.assertEqual(
            set(WAKE_ACK_PROCESSING_PRESETS),
            {"clean", "case_robot", "surprised_ack"},
        )


if __name__ == "__main__":
    unittest.main()
