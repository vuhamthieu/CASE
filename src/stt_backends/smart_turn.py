"""Optional Smart Turn v3 semantic end-of-turn inference."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)
WEAK_ENDINGS = (
    "can you tell me",
    "can you tell",
    "tell me about",
    "what is",
    "who is",
    "can you",
    "do you know",
    "i want you to",
)


def has_weak_ending(text: str) -> bool:
    normalized = " ".join(text.lower().strip(" ,.?!").split())
    return any(normalized.endswith(ending) for ending in WEAK_ENDINGS)


class SmartTurnDetector:
    def __init__(
        self,
        model_path: str | Path,
        *,
        threshold: float = 0.55,
        sample_rate: int = 16_000,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.threshold = float(threshold)
        self.sample_rate = sample_rate
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Smart Turn model missing: {self.model_path}")
        try:
            import onnxruntime as ort
            from transformers import WhisperFeatureExtractor
        except ImportError as exc:
            raise RuntimeError(
                "Smart Turn requires onnxruntime and transformers"
            ) from exc
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._extractor = WhisperFeatureExtractor(chunk_length=8)

    def completion_probability(self, pcm16: np.ndarray) -> float:
        audio = np.asarray(pcm16, dtype=np.int16).reshape(-1).astype(np.float32)
        audio /= 32768.0
        maximum = 8 * self.sample_rate
        if len(audio) > maximum:
            audio = audio[-maximum:]
        elif len(audio) < maximum:
            audio = np.pad(audio, (maximum - len(audio), 0))
        inputs = self._extractor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="np",
            padding="max_length",
            max_length=maximum,
            truncation=True,
            do_normalize=True,
        )
        features = np.expand_dims(
            inputs.input_features.squeeze(0).astype(np.float32),
            axis=0,
        )
        output = self._session.run(None, {"input_features": features})
        return float(np.asarray(output[0]).reshape(-1)[0])

    def is_complete(self, pcm16: np.ndarray) -> tuple[bool, float]:
        probability = self.completion_probability(pcm16)
        return probability >= self.threshold, probability
