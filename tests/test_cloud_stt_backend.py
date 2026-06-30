import concurrent.futures
import sys
import time
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
from src.config import defaults
from src.stt_backends.cloud_stt import CloudSttResult, GeminiCloudSttProvider
from src.stt_backends.domain_glossary import DomainGlossary

for module_name in STUBBED_MODULES:
    sys.modules.pop(module_name, None)


class FakeCloudSttProvider:
    def __init__(self, text="", delay=0.0, error=None):
        self.text = text
        self.delay = delay
        self.error = error

    def transcribe(self, waveform, sample_rate):
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        return CloudSttResult(
            text=self.text,
            provider="gemini",
            latency_sec=self.delay,
        )


class CloudSttBackendTests(unittest.TestCase):
    def make_engine(self, provider) -> STTEngine:
        engine = STTEngine.__new__(STTEngine)
        engine.samplerate = 16000
        engine.cloud_stt_final_mode = "cloud"
        engine.cloud_stt_provider_name = "gemini"
        engine.cloud_stt_provider = provider
        engine.cloud_stt_timeout_sec = 0.05
        engine.cloud_stt_fallback = "vosk_small"
        engine.cloud_stt_save_debug_audio = False
        engine.cloud_stt_debug_dir = Path("output/stt_debug")
        engine._cloud_stt_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        engine.stt_plan = SimpleNamespace(
            profile="balanced",
            final_chain=("vosk_lgraph", "vosk_small"),
        )
        engine.final_mode = "vosk_lgraph"
        engine.vosk_lgraph_model_path = Path("/path/that/does/not/exist")
        engine._last_transcript_source = ""
        engine._last_published_transcript = ""
        engine._session_turn_metrics = {}
        engine.glossary_repair_enabled = True
        engine.domain_glossary = DomainGlossary.from_file(
            Path(__file__).resolve().parents[1] / "config" / "stt_domain_glossary.json"
        )
        return engine

    def tearDown(self):
        engine = getattr(self, "engine", None)
        executor = getattr(engine, "_cloud_stt_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def test_cloud_stt_success_path_uses_cloud_transcript(self):
        self.engine = self.make_engine(FakeCloudSttProvider("do you know g p a six"))
        status = {"selected_source": "vosk_small"}

        selected, reason = self.engine._select_final_transcript_for_utterance(
            vosk_candidate="do you know gps six",
            sense_text="",
            waveform=np.ones(1600, dtype=np.int16),
            backend_status=status,
        )

        self.assertEqual(selected, "do you know g p a six")
        self.assertEqual(reason, "cloud_stt")
        self.assertEqual(status["selected_source"], "cloud_stt")

    def test_glossary_repair_runs_after_cloud_stt(self):
        self.engine = self.make_engine(FakeCloudSttProvider("do you know g p a six"))
        self.engine._last_transcript_source = "cloud_stt"

        processed = self.engine._process_transcript("do you know g p a six")

        self.assertEqual(processed.raw_text, "do you know g p a six")
        self.assertEqual(processed.repaired_text, "do you know GTA 6")
        self.assertEqual(processed.repair_reason, "domain_glossary")

    def test_cloud_stt_timeout_falls_back_to_vosk_small(self):
        self.engine = self.make_engine(FakeCloudSttProvider("late text", delay=0.2))
        status = {"selected_source": "vosk_small"}

        selected, reason = self.engine._select_final_transcript_for_utterance(
            vosk_candidate="fallback words",
            sense_text="",
            waveform=np.ones(1600, dtype=np.int16),
            backend_status=status,
        )

        self.assertEqual(selected, "fallback words")
        self.assertEqual(reason, "cloud_fallback")
        self.assertEqual(status["selected_source"], "vosk_small")
        self.assertEqual(status["cloud_stt_error"], "timeout")

    def test_empty_cloud_stt_result_falls_back_to_vosk_small(self):
        self.engine = self.make_engine(FakeCloudSttProvider(""))
        status = {"selected_source": "vosk_small"}

        selected, reason = self.engine._select_final_transcript_for_utterance(
            vosk_candidate="local fallback",
            sense_text="",
            waveform=np.ones(1600, dtype=np.int16),
            backend_status=status,
        )

        self.assertEqual(selected, "local fallback")
        self.assertEqual(reason, "cloud_fallback")
        self.assertEqual(status["selected_source"], "vosk_small")
        self.assertEqual(status["cloud_stt_error"], "empty_result")

    def test_piper_output_and_native_audio_defaults_are_unchanged(self):
        self.assertEqual(defaults.VOICE_OUTPUT_BACKEND, "piper_onnx")
        self.assertEqual(defaults.PIPER_MODEL_PATH, "models/voices/CASE.onnx")
        self.assertFalse(defaults.GEMINI_LIVE_NATIVE_AUDIO_ENABLED)

    def test_cloud_final_mode_logs_cloud_as_active_mode(self):
        self.engine = self.make_engine(FakeCloudSttProvider("hello"))
        self.engine.vad_gate = None
        self.engine.final_fallback_mode = "vosk_small"

        with self.assertLogs(level="INFO") as logs:
            self.engine._log_stt_profile_runtime()

        output = "\n".join(logs.output)
        self.assertIn("STT_FINAL_MODE: cloud", output)
        self.assertIn("STT_FINAL_FALLBACK: vosk_small", output)
        self.assertNotIn("STT_FINAL_MODE: vosk_lgraph", output)

    def test_gemini_cloud_stt_prompt_includes_case_name_context(self):
        provider = GeminiCloudSttProvider(api_key="test-key", model="test-model")

        self.assertIn("robot's name is CASE", provider.prompt)
        self.assertIn("UK case", provider.prompt)
        self.assertIn("GTA 6", provider.prompt)
        self.assertIn("PCA9685", provider.prompt)


if __name__ == "__main__":
    unittest.main()
