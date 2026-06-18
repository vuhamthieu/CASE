from __future__ import annotations

import queue
import threading
import time
import wave
from collections import deque
from math import ceil, gcd
from pathlib import Path
from typing import Callable, Iterable, TypedDict

import numpy as np


SAMPLE_RATE = 16_000
FRAME_DURATION_SECONDS = 0.08
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_DURATION_SECONDS)
NATIVE_CHUNK_SECONDS = 0.5
FALSE_POSITIVE_CONTEXT_SECONDS = 1.0


class PendingFalsePositiveClip(TypedDict):
    frames: list[np.ndarray]
    remaining_post_frames: int
    model_name: str
    score: float


class WakeWordListener:
    """Continuous openWakeWord listener for local ONNX wake word models."""

    def __init__(
        self,
        model_paths: Iterable[str | Path],
        threshold: float = 0.995,
        strong_threshold: float = 0.998,
        min_hits: int = 3,
        hit_window_sec: float = 0.7,
        cooldown_seconds: float = 2.0,
        print_scores: bool = False,
        input_gain: float = 1.0,
        debug_audio: bool = False,
        save_debug_wav: str | Path | None = None,
        save_debug_seconds: float = 5.0,
        frame_scores: bool = False,
        record_false_positive_dir: str | Path | None = None,
    ) -> None:
        self.model_paths = [Path(path).expanduser().resolve() for path in model_paths]
        self.threshold = threshold
        self.strong_threshold = strong_threshold
        self.min_hits = min_hits
        self.hit_window_sec = hit_window_sec
        self.cooldown_seconds = cooldown_seconds
        self.print_scores = print_scores
        self.input_gain = input_gain
        self.debug_audio = debug_audio
        self.save_debug_wav = (
            Path(save_debug_wav).expanduser() if save_debug_wav else None
        )
        self.save_debug_seconds = save_debug_seconds
        self.frame_scores = frame_scores
        self.record_false_positive_dir = (
            Path(record_false_positive_dir).expanduser()
            if record_false_positive_dir
            else None
        )
        self._running = False
        self._last_trigger_time = 0.0
        self._hit_history: dict[str, deque[tuple[float, float]]] = {}
        self._debug_wav_samples: deque[np.ndarray] = deque()
        self._debug_wav_sample_count = 0
        self._false_positive_pre_frames: deque[np.ndarray] = deque(
            maxlen=ceil(FALSE_POSITIVE_CONTEXT_SECONDS / FRAME_DURATION_SECONDS)
        )
        self._false_positive_post_frames = ceil(
            FALSE_POSITIVE_CONTEXT_SECONDS / FRAME_DURATION_SECONDS
        )
        self._false_positive_pending: list[PendingFalsePositiveClip] = []
        self._false_positive_next_number = 1
        if self.record_false_positive_dir:
            self.record_false_positive_dir.mkdir(parents=True, exist_ok=True)
            self._false_positive_next_number = self._next_false_positive_number()

        self._validate_confirmation_settings()
        self._log_selected_models()
        self._validate_models()

        try:
            from openwakeword.model import Model
        except ImportError as exc:
            raise RuntimeError(
                "openWakeWord is not installed. Install it with: "
                "python3 -m pip install openwakeword"
            ) from exc

        self._model_names_by_key = {
            model_path.stem: model_path.stem for model_path in self.model_paths
        }
        self.model = Model(
            wakeword_models=[str(path) for path in self.model_paths],
            inference_framework="onnx",
        )

    def _log_selected_models(self) -> None:
        print(f"Wake word threshold: {self.threshold}", flush=True)
        print(f"Wake word strong threshold: {self.strong_threshold}", flush=True)
        print(
            "Wake word confirmation: "
            f"min_hits={self.min_hits}, "
            f"hit_window_sec={self.hit_window_sec}, "
            f"cooldown_sec={self.cooldown_seconds}",
            flush=True,
        )
        if self.record_false_positive_dir:
            print(
                "False-positive recording enabled: "
                f"{self.record_false_positive_dir}",
                flush=True,
            )
        for model_path in self.model_paths:
            data_path = Path(f"{model_path}.data")
            print(f"Selected wake word model: {model_path.stem}", flush=True)
            print(f"Resolved ONNX path: {model_path}", flush=True)
            print(f"ONNX exists: {model_path.is_file()}", flush=True)
            print(f"ONNX data path: {data_path}", flush=True)
            print(f"ONNX data exists: {data_path.is_file()}", flush=True)

    def _validate_confirmation_settings(self) -> None:
        if self.min_hits < 1:
            raise ValueError("min_hits must be at least 1")
        if self.hit_window_sec <= 0:
            raise ValueError("hit_window_sec must be greater than 0")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must not be negative")

    def listen_forever(self, on_wakeword: Callable[[str, float], None]) -> None:
        """Listen on the default microphone until stop() or Ctrl+C is called."""
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is not installed. Install it with: "
                "python3 -m pip install sounddevice"
            ) from exc
        try:
            from scipy.signal import resample_poly
        except ImportError as exc:
            raise RuntimeError(
                "scipy is not installed. Install it with: python3 -m pip install scipy"
            ) from exc

        audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=4)
        status_queue: queue.Queue[str] = queue.Queue(maxsize=1)
        device_info = sd.query_devices(kind="input")
        device_sample_rate = int(round(device_info["default_samplerate"]))
        input_channels = max(1, min(2, int(device_info["max_input_channels"])))
        native_chunk_samples = max(
            1,
            int(round(device_sample_rate * NATIVE_CHUNK_SECONDS)),
        )
        divisor = gcd(device_sample_rate, SAMPLE_RATE)
        resample_up = SAMPLE_RATE // divisor
        resample_down = device_sample_rate // divisor
        resampled_buffer = np.empty(0, dtype=np.int16)
        last_status_print = 0.0
        predict_stats = self._new_predict_stats()

        def put_latest_audio(audio_bytes: bytes) -> None:
            try:
                audio_queue.put_nowait(audio_bytes)
                return
            except queue.Full:
                pass

            try:
                audio_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                audio_queue.put_nowait(audio_bytes)
            except queue.Full:
                pass

        def put_status(status: str) -> None:
            try:
                status_queue.put_nowait(status)
            except queue.Full:
                pass

        def capture_audio(stream) -> None:
            while self._running:
                try:
                    audio_bytes, overflowed = stream.read(native_chunk_samples)
                except Exception as exc:
                    if self._running:
                        put_status(f"microphone read failed: {exc}")
                    break

                if overflowed:
                    put_status("input overflow")
                put_latest_audio(bytes(audio_bytes))

        self._running = True
        print(
            "Listening for wake words "
            f"from '{device_info['name']}' at {device_sample_rate} Hz. "
            f"Resampling to {SAMPLE_RATE} Hz mono int16, "
            f"{FRAME_SAMPLES}-sample wake word frames.",
            flush=True,
        )

        try:
            with sd.RawInputStream(
                samplerate=device_sample_rate,
                blocksize=0,
                channels=input_channels,
                dtype="int16",
                latency="high",
            ) as stream:
                capture_thread = threading.Thread(
                    target=capture_audio,
                    args=(stream,),
                )
                capture_thread.start()

                try:
                    while self._running:
                        try:
                            stream_status = status_queue.get_nowait()
                        except queue.Empty:
                            stream_status = None

                        now = time.monotonic()
                        if stream_status and now - last_status_print >= 2.0:
                            last_status_print = now
                            print(
                                f"Microphone stream warning: {stream_status}",
                                flush=True,
                            )

                        while audio_queue.qsize() > 2:
                            try:
                                audio_queue.get_nowait()
                            except queue.Empty:
                                break

                        try:
                            audio_bytes = audio_queue.get(timeout=0.25)
                        except queue.Empty:
                            continue

                        raw_input = np.frombuffer(audio_bytes, dtype=np.int16)
                        if input_channels > 1:
                            raw_input = raw_input.reshape(-1, input_channels)
                        self._debug_audio_array("raw input", raw_input)
                        audio = self._raw_audio_to_mono(audio_bytes, input_channels)
                        self._debug_audio_array("raw mono input", audio)
                        resampled = resample_poly(
                            audio.astype(np.float32),
                            resample_up,
                            resample_down,
                        )
                        if self.input_gain != 1.0:
                            resampled *= self.input_gain

                        resampled = np.clip(
                            np.rint(resampled),
                            np.iinfo(np.int16).min,
                            np.iinfo(np.int16).max,
                        ).astype(np.int16)
                        self._debug_audio_array("resampled int16", resampled)
                        resampled_buffer = np.concatenate(
                            (resampled_buffer, resampled)
                        )

                        while len(resampled_buffer) >= FRAME_SAMPLES:
                            frame = resampled_buffer[:FRAME_SAMPLES]
                            resampled_buffer = resampled_buffer[FRAME_SAMPLES:]
                            self._predict_frame(frame, on_wakeword, predict_stats)
                finally:
                    self._running = False
                    capture_thread.join(timeout=NATIVE_CHUNK_SECONDS + 0.5)
        finally:
            self._running = False
            self._print_final_stats(predict_stats)
            self._flush_false_positive_pending()
            self._write_debug_wav()

    def predict_resampled_audio(
        self,
        audio: np.ndarray,
        on_wakeword: Callable[[str, float], None],
    ) -> list[dict[str, float | str]]:
        """Feed already-resampled 16 kHz mono int16 audio through openWakeWord."""
        if not isinstance(audio, np.ndarray):
            raise TypeError("audio must be a numpy.ndarray")

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if self.input_gain != 1.0:
            audio *= self.input_gain
        audio = np.clip(
            np.rint(audio),
            np.iinfo(np.int16).min,
            np.iinfo(np.int16).max,
        ).astype(np.int16)
        predict_stats = self._new_predict_stats()
        predict_stats["score_print_interval"] = 0.0
        frame_results: list[dict[str, float | str]] = []

        self._debug_audio_array("resampled int16", audio)
        usable_samples = len(audio) - (len(audio) % FRAME_SAMPLES)
        for offset in range(0, usable_samples, FRAME_SAMPLES):
            frame = audio[offset:offset + FRAME_SAMPLES]
            frame_results.append(
                self._predict_frame(frame, on_wakeword, predict_stats)
            )

        self._print_final_stats(predict_stats)
        self._flush_false_positive_pending()
        self._write_debug_wav()
        return frame_results

    def stop(self) -> None:
        self._running = False

    def reset_model_state(self) -> None:
        reset = getattr(self.model, "reset", None)
        if callable(reset):
            reset()
        self._hit_history.clear()
        self._false_positive_pre_frames.clear()
        self._false_positive_pending.clear()

    def _validate_models(self) -> None:
        if not self.model_paths:
            raise ValueError("No wake word model paths were provided.")

        missing_files: list[str] = []
        for model_path in self.model_paths:
            if model_path.suffix != ".onnx":
                raise ValueError(f"Wake word model must be a .onnx file: {model_path}")

            if not model_path.is_file():
                missing_files.append(str(model_path))

            data_path = Path(f"{model_path}.data")
            if not data_path.is_file():
                missing_files.append(str(data_path))

        if missing_files:
            formatted = "\n  - ".join(missing_files)
            raise FileNotFoundError(
                "Missing wake word model file(s):\n"
                f"  - {formatted}\n"
                "Each .onnx file must be in the same folder as its .onnx.data file."
            )

    def _cooldown_elapsed(self) -> bool:
        elapsed = time.monotonic() - self._last_trigger_time
        return elapsed >= self.cooldown_seconds

    def _raw_audio_to_mono(self, indata, input_channels: int) -> np.ndarray:
        audio = np.frombuffer(indata, dtype=np.int16)
        if input_channels == 1:
            return audio.copy()

        audio = audio.reshape(-1, input_channels).astype(np.int32)
        return (audio.sum(axis=1) // input_channels).astype(np.int16)

    def _new_predict_stats(self) -> dict[str, float]:
        return {
            "last_score_print": 0.0,
            "predict_count": 0.0,
            "predict_seconds_total": 0.0,
            "score_print_interval": 1.0,
            "window_best_score": 0.0,
            "window_best_name": "unknown",
            "all_time_best_score": 0.0,
            "all_time_best_name": "unknown",
            "max_peak": 0.0,
            "clipped_frames": 0.0,
            "confirmed_wakes": 0.0,
        }

    def _predict_frame(
        self,
        frame: np.ndarray,
        on_wakeword: Callable[[str, float], None],
        stats: dict[str, float],
    ) -> dict[str, float | str]:
        frame = np.asarray(frame)
        if frame.dtype != np.int16 or frame.shape != (FRAME_SAMPLES,):
            raise ValueError(
                "openWakeWord frame must be np.int16 with shape "
                f"({FRAME_SAMPLES},); got dtype={frame.dtype}, shape={frame.shape}"
            )

        self._track_false_positive_audio(frame)
        self._save_debug_frame(frame)
        self._debug_frame(frame)
        rms, peak = self._audio_level(frame)
        frame_index = int(stats["predict_count"])
        timestamp = frame_index * FRAME_DURATION_SECONDS
        if peak > stats["max_peak"]:
            stats["max_peak"] = peak
        if peak >= 0.98:
            stats["clipped_frames"] += 1.0

        predict_started = time.monotonic()
        predictions = self.model.predict(frame)
        stats["predict_seconds_total"] += time.monotonic() - predict_started
        stats["predict_count"] += 1.0

        if self.debug_audio:
            print(f"[debug] prediction dict: {predictions}", flush=True)

        model_name, score = self._highest_score(predictions)
        prediction_scores = self._prediction_scores(predictions)
        confirmations = {
            name: self._update_confirmation_window(name, value, timestamp)
            for name, value in prediction_scores.items()
        }
        top_confirmation = confirmations.get(
            model_name,
            {
                "hit_count": 0.0,
                "window_max": 0.0,
                "confirmed": 0.0,
            },
        )
        confirmed_candidates = [
            (name, prediction_scores[name], confirmation)
            for name, confirmation in confirmations.items()
            if confirmation["confirmed"] >= 1.0
        ]
        confirmed_model = None
        if confirmed_candidates:
            confirmed_model = max(confirmed_candidates, key=lambda item: item[1])

        result: dict[str, float | str] = {
            "frame_index": float(frame_index),
            "timestamp": timestamp,
            "model_name": model_name,
            "score": score,
            "rms": rms,
            "peak": peak,
            "hit_count": top_confirmation["hit_count"],
            "window_max": top_confirmation["window_max"],
            "confirmed": top_confirmation["confirmed"],
        }
        if score > stats["window_best_score"]:
            stats["window_best_score"] = score
            stats["window_best_name"] = model_name
        if score > stats["all_time_best_score"]:
            stats["all_time_best_score"] = score
            stats["all_time_best_name"] = model_name

        now = time.monotonic()

        if (
            self.print_scores
            and now - stats["last_score_print"] >= stats["score_print_interval"]
        ):
            stats["last_score_print"] = now
            avg_predict_ms = (
                stats["predict_seconds_total"] / stats["predict_count"] * 1000.0
            )
            print(
                "Top wake score: "
                f"{model_name}={score:.3f} "
                f"[{self._format_prediction_scores(prediction_scores)}] "
                f"(last {stats['score_print_interval']:.1f}s max "
                f"{stats['window_best_name']}="
                f"{stats['window_best_score']:.3f}, "
                f"hit_count={int(top_confirmation['hit_count'])}, "
                f"window_max={top_confirmation['window_max']:.3f}, "
                f"confirmed={bool(top_confirmation['confirmed'])}, "
                f"session max {stats['all_time_best_name']}="
                f"{stats['all_time_best_score']:.3f}, "
                f"rms={rms:.3f}, peak={peak:.3f}, "
                f"avg predict {avg_predict_ms:.1f} ms/frame)",
                flush=True,
            )
            stats["window_best_score"] = 0.0
            stats["window_best_name"] = "unknown"

        if confirmed_model and self._cooldown_elapsed():
            confirmed_name, confirmed_score, _ = confirmed_model
            stats["confirmed_wakes"] += 1.0
            self._last_trigger_time = time.monotonic()
            print(
                f"Wake word detected: {confirmed_name} score={confirmed_score:.3f}",
                flush=True,
            )
            self._start_false_positive_clip(confirmed_name, confirmed_score)
            on_wakeword(confirmed_name, confirmed_score)

        if self.frame_scores:
            print(
                "Frame "
                f"{frame_index:05d} "
                f"t={timestamp:.2f}s "
                f"score={score:.6f} "
                f"hits={int(top_confirmation['hit_count'])} "
                f"window_max={top_confirmation['window_max']:.6f} "
                f"confirmed={bool(top_confirmation['confirmed'])} "
                f"rms={rms:.4f} "
                f"peak={peak:.4f}",
                flush=True,
            )

        return result

    def _prediction_scores(self, predictions) -> dict[str, float]:
        if not isinstance(predictions, dict):
            return {}

        scores: dict[str, float] = {}
        for raw_name, raw_score in predictions.items():
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue

            model_name = self._model_names_by_key.get(str(raw_name), str(raw_name))
            scores[model_name] = score

        return scores

    def _update_confirmation_window(
        self,
        model_name: str,
        score: float,
        timestamp: float,
    ) -> dict[str, float]:
        history = self._hit_history.setdefault(model_name, deque())

        if score >= self.threshold:
            history.append((timestamp, score))

        window_start = timestamp - self.hit_window_sec
        while history and history[0][0] < window_start:
            history.popleft()

        hit_count = len(history)
        window_max = max((entry_score for _, entry_score in history), default=0.0)
        confirmed = (
            score >= self.threshold
            and hit_count >= self.min_hits
            and window_max >= self.strong_threshold
        )
        return {
            "hit_count": float(hit_count),
            "window_max": float(window_max),
            "confirmed": 1.0 if confirmed else 0.0,
        }

    def _format_prediction_scores(self, scores: dict[str, float]) -> str:
        if not scores:
            return "no model scores"

        return ", ".join(
            f"{model_name}={score:.3f}"
            for model_name, score in sorted(scores.items())
        )

    def _track_false_positive_audio(self, frame: np.ndarray) -> None:
        if not self.record_false_positive_dir:
            return

        frame_copy = frame.copy()
        self._false_positive_pre_frames.append(frame_copy)

        completed_clips: list[PendingFalsePositiveClip] = []
        for pending in self._false_positive_pending:
            pending["frames"].append(frame_copy)
            pending["remaining_post_frames"] -= 1
            if pending["remaining_post_frames"] <= 0:
                completed_clips.append(pending)

        for pending in completed_clips:
            self._false_positive_pending.remove(pending)
            self._write_false_positive_clip(pending)

    def _start_false_positive_clip(self, model_name: str, score: float) -> None:
        if not self.record_false_positive_dir:
            return

        pending: PendingFalsePositiveClip = {
            "frames": [frame.copy() for frame in self._false_positive_pre_frames],
            "remaining_post_frames": self._false_positive_post_frames,
            "model_name": model_name,
            "score": score,
        }
        self._false_positive_pending.append(pending)

    def _flush_false_positive_pending(self) -> None:
        if not self.record_false_positive_dir:
            return

        while self._false_positive_pending:
            pending = self._false_positive_pending.pop(0)
            self._write_false_positive_clip(pending)

    def _next_false_positive_number(self) -> int:
        if not self.record_false_positive_dir:
            return 1

        prefix = self.record_false_positive_dir.name
        highest = 0
        for wav_path in self.record_false_positive_dir.glob(f"{prefix}_*.wav"):
            suffix = wav_path.stem.removeprefix(f"{prefix}_")
            if suffix.isdigit():
                highest = max(highest, int(suffix))

        return highest + 1

    def _write_false_positive_clip(self, pending: PendingFalsePositiveClip) -> None:
        if not self.record_false_positive_dir or not pending["frames"]:
            return

        prefix = self.record_false_positive_dir.name
        wav_path = (
            self.record_false_positive_dir
            / f"{prefix}_{self._false_positive_next_number:04d}.wav"
        )
        self._false_positive_next_number += 1

        audio = np.concatenate(pending["frames"]).astype(np.int16)
        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio.tobytes())

        print(
            "Saved false-positive clip: "
            f"{wav_path} "
            f"({len(audio) / SAMPLE_RATE:.1f}s, "
            f"model={pending['model_name']}, score={pending['score']:.3f})",
            flush=True,
        )

    def _print_final_stats(self, stats: dict[str, float]) -> None:
        if not self.print_scores or stats["predict_count"] == 0:
            return

        avg_predict_ms = (
            stats["predict_seconds_total"] / stats["predict_count"] * 1000.0
        )
        print(
            "Final wake score stats: "
            f"session max {stats['all_time_best_name']}="
            f"{stats['all_time_best_score']:.3f}, "
            f"max peak={stats['max_peak']:.3f}, "
            f"clipped frames={int(stats['clipped_frames'])}, "
            f"confirmed wakes={int(stats['confirmed_wakes'])}, "
            f"frames={int(stats['predict_count'])}, "
            f"avg predict {avg_predict_ms:.1f} ms/frame",
            flush=True,
        )
        if stats["confirmed_wakes"] < 1:
            print(
                "No wake confirmation fired. A wake needs score >= threshold, "
                "enough hits inside the window, and a strong enough window max.",
                flush=True,
            )

    def _audio_level(self, audio: np.ndarray) -> tuple[float, float]:
        if len(audio) == 0:
            return 0.0, 0.0

        normalized = audio.astype(np.float32) / np.iinfo(np.int16).max
        rms = float(np.sqrt(np.mean(normalized * normalized)))
        peak = float(np.max(np.abs(normalized)))
        return rms, peak

    def _debug_audio_array(self, label: str, audio: np.ndarray) -> None:
        if not self.debug_audio:
            return

        if len(audio) == 0:
            print(
                f"[debug] {label}: dtype={audio.dtype}, shape={audio.shape}, empty",
                flush=True,
            )
            return

        rms, peak = self._audio_level(audio)
        print(
            f"[debug] {label}: dtype={audio.dtype}, shape={audio.shape}, "
            f"min={int(audio.min())}, max={int(audio.max())}, "
            f"rms={rms:.4f}, peak={peak:.4f}",
            flush=True,
        )

    def _debug_frame(self, frame: np.ndarray) -> None:
        if not self.debug_audio:
            return

        rms, peak = self._audio_level(frame)
        print(
            f"[debug] frame before predict: dtype={frame.dtype}, "
            f"shape={frame.shape}, min={int(frame.min())}, "
            f"max={int(frame.max())}, rms={rms:.4f}, peak={peak:.4f}",
            flush=True,
        )

    def _save_debug_frame(self, frame: np.ndarray) -> None:
        if not self.save_debug_wav:
            return

        max_samples = max(1, int(SAMPLE_RATE * self.save_debug_seconds))
        saved = frame.copy()
        self._debug_wav_samples.append(saved)
        self._debug_wav_sample_count += len(saved)
        while self._debug_wav_sample_count > max_samples:
            removed = self._debug_wav_samples.popleft()
            self._debug_wav_sample_count -= len(removed)

    def _write_debug_wav(self) -> None:
        if not self.save_debug_wav or not self._debug_wav_samples:
            return

        audio = np.concatenate(self._debug_wav_samples).astype(np.int16)
        self.save_debug_wav.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(self.save_debug_wav), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio.tobytes())

        print(
            f"Saved debug wav: {self.save_debug_wav} "
            f"({len(audio) / SAMPLE_RATE:.1f}s, {SAMPLE_RATE} Hz, int16 mono)",
            flush=True,
        )
        self._debug_wav_samples.clear()
        self._debug_wav_sample_count = 0

    def _highest_score(self, predictions) -> tuple[str, float]:
        if not isinstance(predictions, dict) or not predictions:
            return "unknown", 0.0

        model_name = "unknown"
        best_score = 0.0
        for raw_name, raw_score in predictions.items():
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue

            if score > best_score:
                model_name = self._model_names_by_key.get(str(raw_name), str(raw_name))
                best_score = score

        return model_name, best_score
