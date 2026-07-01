import unittest

import numpy as np

from actuation.audio_output.tts_engine import (
    pad_and_fade_tts_pcm,
    trim_tts_silence_pcm,
)
from src.config import defaults


class TtsSmoothingTests(unittest.TestCase):
    def test_silence_trim_keeps_margin_and_audio(self):
        sample_rate = 1000
        leading = np.zeros(100, dtype="<i2")
        tone = np.full(300, 4000, dtype="<i2")
        trailing = np.zeros(120, dtype="<i2")
        audio = np.concatenate((leading, tone, trailing)).tobytes()

        trimmed, stats = trim_tts_silence_pcm(
            audio,
            sample_rate,
            threshold_db=-45,
            keep_ms=35,
        )

        self.assertTrue(stats.get("trimmed"))
        self.assertAlmostEqual(stats["lead_ms"], 65.0, delta=1.0)
        self.assertAlmostEqual(stats["tail_ms"], 85.0, delta=1.0)
        self.assertGreater(len(trimmed), len(tone.tobytes()))
        self.assertLess(len(trimmed), len(audio))

    def test_silence_trim_does_not_trim_all_silence_audio(self):
        audio = np.zeros(200, dtype="<i2").tobytes()
        trimmed, stats = trim_tts_silence_pcm(audio, 1000)

        self.assertEqual(trimmed, audio)
        self.assertFalse(stats.get("trimmed"))

    def test_safe_padding_still_extends_short_sounds(self):
        sample_rate = 1000
        audio = np.full(100, 2000, dtype="<i2").tobytes()

        padded = pad_and_fade_tts_pcm(audio, sample_rate=sample_rate)

        self.assertGreater(len(padded), len(audio))

    def test_piper_and_native_audio_defaults_are_unchanged(self):
        self.assertEqual(defaults.VOICE_OUTPUT_BACKEND, "piper_onnx")
        self.assertEqual(defaults.PIPER_MODEL_PATH, "models/voices/CASE.onnx")
        self.assertFalse(defaults.GEMINI_LIVE_NATIVE_AUDIO_ENABLED)


if __name__ == "__main__":
    unittest.main()
