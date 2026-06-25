import unittest

import numpy as np

from src.audio.wake_ack_fx import apply_wake_ack_fx, pitch_shift_light, tempo_scale_light
from scripts.generate_wake_ack_wavs import PROFILE_SELECTIONS, style_for


class WakeAckFxTests(unittest.TestCase):
    def test_clear_short_profile_does_not_pitch_or_speed_up(self):
        self.assertEqual(PROFILE_SELECTIONS["clear_short"]["im_listening"], 1)
        style = style_for("clear_short", "im_listening", 1)
        self.assertEqual(style["pitch_shift_semitones"], 0.0)
        self.assertGreaterEqual(style["length_scale"], 1.0)
        self.assertLessEqual(style["tempo"], 1.0)

    def test_pitch_and_tempo_make_short_reaction_quicker(self):
        audio = np.sin(np.linspace(0, 20, 22_050)).astype(np.float32) * 10_000
        pitched = pitch_shift_light(audio, 22_050, 2.5)
        faster = tempo_scale_light(pitched, 1.08)
        self.assertLess(len(faster), len(audio))

    def test_apply_fx_returns_safe_contiguous_int16(self):
        audio = np.full(8_000, 20_000, dtype=np.int16)
        styled = apply_wake_ack_fx(
            audio,
            22_050,
            {
                "pitch_shift_semitones": 1.5,
                "tempo": 1.05,
                "gain_db": 1.5,
            },
        )
        self.assertEqual(styled.dtype, np.int16)
        self.assertTrue(styled.flags.c_contiguous)
        self.assertLessEqual(int(np.max(np.abs(styled.astype(np.int32)))), 23_200)


if __name__ == "__main__":
    unittest.main()
