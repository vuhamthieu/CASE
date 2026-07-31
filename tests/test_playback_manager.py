import types
import unittest
from unittest.mock import patch

import numpy as np

from src.audio.playback_manager import AudioPlaybackManager


class FakeRawOutputStream:
    instances = []
    underflow = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.active = False
        self.closed = False
        self.start_count = 0
        self.stop_count = 0
        self.write_count = 0
        self.__class__.instances.append(self)

    def start(self):
        self.active = True
        self.start_count += 1

    def write(self, payload):
        self.write_count += 1
        self.last_payload = payload
        return self.__class__.underflow

    def stop(self):
        self.active = False
        self.stop_count += 1

    def close(self):
        self.closed = True


class PlaybackManagerTests(unittest.TestCase):
    def setUp(self):
        FakeRawOutputStream.instances.clear()
        FakeRawOutputStream.underflow = False

    def test_reuses_stream_and_closes_only_on_manager_close(self):
        fake_sounddevice = types.SimpleNamespace(
            RawOutputStream=FakeRawOutputStream,
        )
        device_info = {
            "name": "MAX98357A: bcm2835-i2s-HiFi",
            "default_samplerate": 44_100.0,
            "max_output_channels": 2,
        }
        environment = {
            "AUDIO_PLAYBACK_BACKEND": "sounddevice",
            "AUDIO_OUTPUT_SAMPLE_RATE": "44100",
            "AUDIO_OUTPUT_CHANNELS": "2",
            "AUDIO_PLAYBACK_KEEP_STREAM_OPEN": "true",
        }
        with patch.dict("os.environ", environment, clear=False), patch.dict(
            "sys.modules", {"sounddevice": fake_sounddevice}
        ), patch(
            "src.audio.playback_manager.query_output_device",
            return_value=("MAX98357A", device_info),
        ), patch(
            "src.audio.playback_manager.configured_output_device",
            return_value="MAX98357A",
        ):
            manager = AudioPlaybackManager()
            manager.start()
            audio = np.zeros(4410, dtype=np.int16)
            manager.play(audio, 44_100, tail_guard_sec=0)
            manager.play(audio, 44_100, tail_guard_sec=0)

            self.assertEqual(len(FakeRawOutputStream.instances), 1)
            stream = FakeRawOutputStream.instances[0]
            self.assertEqual(stream.start_count, 2)
            self.assertEqual(stream.stop_count, 2)
            self.assertEqual(stream.write_count, 2)
            self.assertFalse(stream.closed)
            self.assertEqual(stream.kwargs["channels"], 2)

            manager.close()
            self.assertTrue(stream.closed)

    def test_safe_mode_uses_high_latency_and_appends_runtime_tail(self):
        fake_sounddevice = types.SimpleNamespace(RawOutputStream=FakeRawOutputStream)
        device_info = {
            "name": "MAX98357A",
            "default_samplerate": 44_100.0,
            "max_output_channels": 2,
        }
        environment = {
            "AUDIO_PLAYBACK_BACKEND": "sounddevice",
            "AUDIO_OUTPUT_SAMPLE_RATE": "44100",
            "AUDIO_OUTPUT_CHANNELS": "2",
            "WAKE_ACK_PLAYBACK_LATENCY": "high",
        }
        with patch.dict("os.environ", environment, clear=False), patch.dict(
            "sys.modules", {"sounddevice": fake_sounddevice}
        ), patch(
            "src.audio.playback_manager.query_output_device",
            return_value=("MAX98357A", device_info),
        ), patch(
            "src.audio.playback_manager.configured_output_device",
            return_value="MAX98357A",
        ):
            manager = AudioPlaybackManager()
            result = manager.play(
                np.zeros(4410, dtype=np.int16),
                44_100,
                tail_guard_sec=0,
                safe_mode=True,
                extra_tail_sec=0.25,
            )

            stream = FakeRawOutputStream.instances[0]
            self.assertEqual(stream.kwargs["latency"], "high")
            self.assertAlmostEqual(result["duration"], 0.35)
            self.assertTrue(result["safe_mode"])
            self.assertEqual(len(stream.last_payload), 15_435 * 2 * 2)

    def test_underflow_forces_future_safe_mode(self):
        fake_sounddevice = types.SimpleNamespace(RawOutputStream=FakeRawOutputStream)
        device_info = {
            "name": "MAX98357A",
            "default_samplerate": 44_100.0,
            "max_output_channels": 2,
        }
        environment = {
            "AUDIO_PLAYBACK_BACKEND": "sounddevice",
            "AUDIO_OUTPUT_SAMPLE_RATE": "44100",
            "AUDIO_OUTPUT_CHANNELS": "2",
            "AUDIO_PLAYBACK_RETRY_ON_UNDERFLOW": "true",
            "AUDIO_PLAYBACK_LATENCY": "low",
            "WAKE_ACK_PLAYBACK_LATENCY": "high",
            "AUDIO_SHORT_SOUND_SAFE_MODE": "false",
        }
        with patch.dict("os.environ", environment, clear=False), patch.dict(
            "sys.modules", {"sounddevice": fake_sounddevice}
        ), patch(
            "src.audio.playback_manager.query_output_device",
            return_value=("MAX98357A", device_info),
        ), patch(
            "src.audio.playback_manager.configured_output_device",
            return_value="MAX98357A",
        ):
            manager = AudioPlaybackManager()
            FakeRawOutputStream.underflow = True
            first = manager.play(
                np.zeros(4410, dtype=np.int16), 44_100, tail_guard_sec=0
            )
            FakeRawOutputStream.underflow = False
            second = manager.play(
                np.zeros(4410, dtype=np.int16), 44_100, tail_guard_sec=0
            )

            self.assertTrue(first["underflow"])
            self.assertFalse(first["safe_mode"])
            self.assertTrue(second["safe_mode"])
            self.assertEqual(FakeRawOutputStream.instances[-1].kwargs["latency"], "high")


if __name__ == "__main__":
    unittest.main()
