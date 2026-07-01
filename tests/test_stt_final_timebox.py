import concurrent.futures
import importlib
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

STUBBED_MODULES = []
sounddevice = ModuleType("sounddevice")
sounddevice.InputStream = object
if "sounddevice" not in sys.modules:
    sys.modules["sounddevice"] = sounddevice
    STUBBED_MODULES.append("sounddevice")

vosk = ModuleType("vosk")
vosk.KaldiRecognizer = object
vosk.Model = object
vosk.SetLogLevel = lambda level: None
if "vosk" not in sys.modules:
    sys.modules["vosk"] = vosk
    STUBBED_MODULES.append("vosk")

scipy = ModuleType("scipy")
scipy_signal = ModuleType("scipy.signal")
scipy_signal.resample_poly = lambda samples, up, down: samples
if "scipy" not in sys.modules:
    sys.modules["scipy"] = scipy
    STUBBED_MODULES.append("scipy")
if "scipy.signal" not in sys.modules:
    sys.modules["scipy.signal"] = scipy_signal
    STUBBED_MODULES.append("scipy.signal")

from perception.audio.stt_engine import STTEngine

for module_name in STUBBED_MODULES:
    sys.modules.pop(module_name, None)


class SttFinalTimeboxTests(unittest.TestCase):
    def make_engine(self) -> STTEngine:
        engine = STTEngine.__new__(STTEngine)
        engine.lgraph_final_timeout_sec = 0.02
        engine.accept_fast_candidate_on_timeout = True
        engine._final_stt_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1
        )
        engine.stt_plan = SimpleNamespace(
            final_chain=("vosk_lgraph", "vosk_small")
        )
        engine.cloud_stt_final_mode = "local"
        engine.final_mode = "vosk_lgraph"
        engine.vosk_lgraph_model_path = Path(tempfile.mkdtemp())
        return engine

    def tearDown(self):
        executor = getattr(self, "executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def test_lgraph_final_timeout_falls_back_to_vosk_small(self):
        engine = self.make_engine()
        self.executor = engine._final_stt_executor

        def slow_lgraph(waveform):
            time.sleep(0.2)
            return "tell me a much better joke"

        engine._transcribe_final_vosk = slow_lgraph
        started = time.monotonic()
        text, error = engine._transcribe_lgraph_timeboxed(
            np.ones(160, dtype=np.int16),
            fallback_candidate="tell me a joke",
        )

        self.assertEqual(text, "")
        self.assertEqual(error, "lgraph_timeout")
        self.assertLess(time.monotonic() - started, 0.15)

    def test_default_lgraph_final_timebox_is_two_point_five_seconds(self):
        import src.realtime.realtime_config as realtime_config

        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("CASE_STT_FINAL_TIMEBOX_SEC", None)
            os.environ.pop("CASE_STT_LGRAPH_FINAL_TIMEOUT_SEC", None)
            importlib.reload(realtime_config)
            try:
                self.assertEqual(realtime_config.CASE_STT_FINAL_TIMEBOX_SEC, 2.5)
                self.assertEqual(
                    realtime_config.CASE_STT_LGRAPH_FINAL_TIMEOUT_SEC,
                    2.5,
                )
            finally:
                importlib.reload(realtime_config)

    def test_lgraph_final_timebox_env_override(self):
        import src.realtime.realtime_config as realtime_config

        with patch.dict(
            "os.environ",
            {"CASE_STT_FINAL_TIMEBOX_SEC": "3.75"},
            clear=False,
        ):
            importlib.reload(realtime_config)
            try:
                self.assertEqual(realtime_config.CASE_STT_FINAL_TIMEBOX_SEC, 3.75)
                self.assertEqual(
                    realtime_config.CASE_STT_LGRAPH_FINAL_TIMEOUT_SEC,
                    3.75,
                )
            finally:
                importlib.reload(realtime_config)

    def test_clear_fast_intent_bypasses_slow_lgraph(self):
        engine = self.make_engine()
        self.executor = engine._final_stt_executor
        engine._transcribe_lgraph_timeboxed = (
            lambda *args, **kwargs: self.fail("lgraph should be bypassed")
        )
        status = {
            "sensevoice_available": False,
            "sensevoice_error": None,
            "vosk_lgraph_available": True,
            "vosk_lgraph_error": None,
            "final_chain": ("vosk_lgraph", "vosk_small"),
        }

        selected, reason = engine._select_final_transcript_for_utterance(
            vosk_candidate="can you tell me a joke",
            sense_text="",
            waveform=np.ones(160, dtype=np.int16),
            backend_status=status,
        )

        self.assertEqual(selected, "can you tell me a joke")
        self.assertEqual(reason, "clear_fast_intent")
        self.assertEqual(status["selected_source"], "vosk_small")

    def test_lgraph_is_accepted_if_ready_within_timeout(self):
        engine = self.make_engine()
        self.executor = engine._final_stt_executor
        engine._transcribe_lgraph_timeboxed = (
            lambda *args, **kwargs: ("tell me a joke", None)
        )
        status = {
            "sensevoice_available": False,
            "sensevoice_error": None,
            "vosk_lgraph_available": True,
            "vosk_lgraph_error": None,
            "final_chain": ("vosk_lgraph", "vosk_small"),
        }

        selected, reason = engine._select_final_transcript_for_utterance(
            vosk_candidate="tell me up",
            sense_text="",
            waveform=np.ones(160, dtype=np.int16),
            backend_status=status,
        )

        self.assertEqual(selected, "tell me a joke")
        self.assertEqual(reason, "final_ready_within_timeout")
        self.assertEqual(status["selected_source"], "vosk_lgraph")


if __name__ == "__main__":
    unittest.main()
