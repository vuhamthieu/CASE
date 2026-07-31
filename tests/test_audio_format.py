import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from src.audio.audio_format import (
    convert_channels,
    load_wav_int16,
    normalize_for_playback,
)


class AudioFormatTests(unittest.TestCase):
    def test_22050_mono_to_44100_stereo_preserves_duration(self):
        source = np.arange(22_050, dtype=np.int16)
        converted = normalize_for_playback(source, 22_050, 44_100, 2)
        self.assertEqual(converted.shape, (44_100, 2))
        self.assertEqual(converted.dtype, np.int16)
        self.assertTrue(converted.flags.c_contiguous)
        self.assertAlmostEqual(len(source) / 22_050, len(converted) / 44_100)

    def test_mono_to_stereo_duplicates_samples(self):
        mono = np.array([[-100], [0], [100]], dtype=np.int16)
        stereo = convert_channels(mono, 2)
        self.assertEqual(stereo.shape, (3, 2))
        np.testing.assert_array_equal(stereo[:, 0], stereo[:, 1])

    def test_wav_header_rate_and_shape_are_loaded_correctly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            samples = np.zeros(22_050, dtype=np.int16)
            with wave.open(str(path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(22_050)
                output.writeframes(samples.tobytes())

            loaded, sample_rate, channels = load_wav_int16(path)
            self.assertEqual(sample_rate, 22_050)
            self.assertEqual(channels, 1)
            self.assertEqual(loaded.shape, (22_050, 1))
            self.assertEqual(loaded.dtype, np.int16)
            self.assertTrue(loaded.flags.c_contiguous)

    def test_resampling_does_not_shrink_duration(self):
        source = np.zeros((28_812, 1), dtype=np.int16)
        converted = normalize_for_playback(source, 22_050, 44_100, 2)
        duration_in = len(source) / 22_050
        duration_out = len(converted) / 44_100
        self.assertLess(abs(duration_out - duration_in) / duration_in, 0.02)


if __name__ == "__main__":
    unittest.main()
