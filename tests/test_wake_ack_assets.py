import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.voice_pipeline import wake_ack
from src.voice_pipeline.wake_ack import (
    WakeAcknowledgementSelector,
    migrate_legacy_generated_wavs,
    play_wake_ack_asset,
)


class WakeAckAssetTests(unittest.TestCase):
    def test_default_selector_excludes_short_interjections(self):
        with patch.dict("os.environ", {}, clear=True):
            selector = WakeAcknowledgementSelector()
            selected = {selector.choose() for _ in range(80)}
        self.assertEqual(selector.pool, ["Yes!", "I'm listening."])
        self.assertTrue(selected <= {"Yes!", "I'm listening."})
        self.assertFalse(
            {
                "You called?",
                "Go on.",
                "I'm here.",
                "Say that again?",
                "Still with you.",
                "What?",
                "Yeah?",
            }
            & selected
        )

    def test_unknown_ack_is_not_in_default_pool(self):
        with patch.dict(
            "os.environ",
            {
                "WAKE_ACK_POOL": "yes,im_listening,unknown_ack",
                "WAKE_ACK_ALLOW_SHORT_INTERJECTIONS": "false",
            },
        ):
            selector = WakeAcknowledgementSelector()
            selected = {selector.choose() for _ in range(80)}
        self.assertNotIn("unknown_ack", selected)
        self.assertTrue(selected <= {"Yes!", "I'm listening."})

    def test_old_keys_are_not_selectable_even_if_configured(self):
        with patch.dict(
            "os.environ",
            {
                "WAKE_ACK_POOL": "yes,im_listening,what,yeah",
                "WAKE_ACK_ALLOW_SHORT_INTERJECTIONS": "true",
            },
        ):
            selector = WakeAcknowledgementSelector()
            selected = {selector.choose() for _ in range(80)}
        self.assertTrue(selected <= {"Yes!", "I'm listening."})
        self.assertFalse({"What?", "Yeah?"} & selected)

    def test_cached_asset_is_default_priority(self):
        with patch(
            "src.voice_pipeline.wake_ack._play_wake_ack_path",
            return_value=True,
        ) as play, patch(
            "src.voice_pipeline.wake_ack.migrate_legacy_generated_wavs"
        ):
            self.assertTrue(
                play_wake_ack_asset(
                    "Yes!",
                    mode="cached_wav",
                    recorded_directory="recorded",
                    cached_directory="generated",
                    fallback_mode="cached_wav",
                )
            )
        self.assertEqual(play.call_count, 1)
        self.assertEqual(play.call_args.args[2], "cached_wav")
        self.assertEqual(play.call_args.args[1], Path("generated") / "yes.wav")

    def test_yes_cached_wav_plays_runtime_audio_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "yes.wav"
            audio = np.zeros(22_050, dtype=np.int16)
            audio[4_000:10_000] = 8000
            with wave.open(str(path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(22_050)
                output.writeframes(audio.tobytes())
            with patch(
                "src.voice_pipeline.wake_ack.play_int16_mono",
                return_value={
                    "duration_in": 1.0,
                    "duration_out": 1.0,
                    "sample_rate": 22_050,
                    "channels": 1,
                    "frames_in": 22_050,
                    "frames_out": 22_050,
                    "resampled": False,
                    "device_name": "test",
                    "safe_mode": True,
                    "underflow": False,
                },
            ) as play:
                with self.assertLogs("src.voice_pipeline.wake_ack", level="INFO") as logs:
                    self.assertTrue(wake_ack._play_wake_ack_path("Yes!", path, "cached_wav"))
        self.assertEqual(play.call_count, 1)
        self.assertTrue(
            any("WAKE_ACK_AUDIO: source=cached_wav" in line for line in logs.output)
        )

    def test_im_listening_uses_cached_generated_wav(self):
        with patch(
            "src.voice_pipeline.wake_ack._play_wake_ack_path",
            return_value=True,
        ) as play, patch(
            "src.voice_pipeline.wake_ack.migrate_legacy_generated_wavs"
        ):
            self.assertTrue(
                play_wake_ack_asset(
                    "I'm listening.",
                    mode="cached_wav",
                    cached_directory="generated",
                    fallback_mode="cached_wav",
                )
            )
        self.assertEqual(play.call_count, 1)
        self.assertEqual(play.call_args.args[1], Path("generated") / "im_listening.wav")
        self.assertEqual(play.call_args.args[2], "cached_wav")

    def test_missing_generated_wav_logs_explicit_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            wake_ack._missing_wav_warnings.clear()
            with self.assertLogs("src.voice_pipeline.wake_ack", level="WARNING") as logs:
                self.assertFalse(
                    play_wake_ack_asset(
                        "Yes!",
                        mode="cached_wav",
                        cached_directory=str(Path(tmp) / "generated"),
                        fallback_mode="cached_wav",
                    )
                )
        self.assertTrue(
            any("WAKE_ACK_MISSING_GENERATED: key=yes" in line for line in logs.output)
        )

    def test_recorded_mode_is_ignored_when_disabled(self):
        with patch(
            "src.voice_pipeline.wake_ack._play_wake_ack_path",
            return_value=True,
        ) as play, patch(
            "src.voice_pipeline.wake_ack.migrate_legacy_generated_wavs"
        ):
            with patch.dict(
                "os.environ",
                {"WAKE_ACK_RECORDED_ENABLED": "false"},
            ):
                self.assertTrue(
                    play_wake_ack_asset(
                        "What?",
                        mode="recorded_wav",
                        recorded_directory="recorded",
                        cached_directory="generated",
                        fallback_mode="cached_wav",
                    )
                )
        self.assertEqual(
            [call.args[2] for call in play.call_args_list],
            ["cached_wav"],
        )

    def test_migration_only_copies_active_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generated = root / "generated"
            generated.mkdir()
            removed_short_ack = "hu" + "h.wav"
            for filename in (
                "you_called.wav",
                "yes.wav",
                removed_short_ack,
                "what.wav",
                "yeah.wav",
                "im_listening.wav",
            ):
                (root / filename).write_bytes(b"RIFF")
            wake_ack._migration_checked.clear()
            migrate_legacy_generated_wavs(generated)
            self.assertTrue((generated / "yes.wav").is_file())
            self.assertTrue((generated / "im_listening.wav").is_file())
            self.assertFalse((generated / "you_called.wav").exists())
            self.assertFalse((generated / removed_short_ack).exists())
            self.assertFalse((generated / "what.wav").exists())
            self.assertFalse((generated / "yeah.wav").exists())


if __name__ == "__main__":
    unittest.main()
