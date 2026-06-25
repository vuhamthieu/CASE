from __future__ import annotations

import argparse
import sys
from math import gcd
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.wakeword_listener import SAMPLE_RATE, WakeWordListener


MODEL_DIR = ROOT / "models" / "wakewords"
MODEL_NAMES = [
    "hey_case_v2",
]
SUMMARY_THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.85)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test CASE openWakeWord models.")
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        default="hey_case_v2",
        help="Wake word model to test. Default: hey_case_v2.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.995,
        help="Wake word trigger threshold. Default: 0.995.",
    )
    parser.add_argument(
        "--strong-threshold",
        type=float,
        default=0.998,
        help="Required max score inside the hit window. Default: 0.998.",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=3,
        help="Minimum threshold-crossing frames inside the hit window. Default: 3.",
    )
    parser.add_argument(
        "--hit-window-sec",
        type=float,
        default=0.7,
        help="Seconds used for wake hit confirmation. Default: 0.7.",
    )
    parser.add_argument(
        "--cooldown-sec",
        type=float,
        default=2.0,
        help="Seconds to suppress repeated wake triggers. Default: 2.0.",
    )
    parser.add_argument(
        "--scores",
        action="store_true",
        help="Print live top wake score and average prediction time.",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=1.0,
        help="Multiply microphone audio before wake word inference. Default: 1.0.",
    )
    parser.add_argument(
        "--device",
        help="Input device index or name from `python3 -m sounddevice`.",
    )
    parser.add_argument(
        "--debug-audio",
        action="store_true",
        help="Print raw audio, resampled audio, frame, and prediction diagnostics.",
    )
    parser.add_argument(
        "--save-debug-wav",
        type=Path,
        help="Save recent 16 kHz int16 mono audio sent to openWakeWord.",
    )
    parser.add_argument(
        "--save-debug-seconds",
        type=float,
        default=5.0,
        help="Seconds of recent openWakeWord input audio to save. Default: 5.",
    )
    parser.add_argument(
        "--wav",
        type=Path,
        help="Run wake word detection on a WAV file instead of the microphone.",
    )
    parser.add_argument(
        "--frame-scores",
        action="store_true",
        help="Print score, hit state, RMS, and peak for every 80 ms frame.",
    )
    parser.add_argument(
        "--test-training-clips",
        action="store_true",
        help="Test synthetic training output WAV folders for the selected model.",
    )
    parser.add_argument(
        "--list-wavs",
        action="store_true",
        help="List WAV files under output/<model> and the project root, then exit.",
    )
    parser.add_argument(
        "--record-false-positive-dir",
        type=Path,
        help="Live mode only: save confirmed wake context clips into this folder.",
    )
    return parser.parse_args()


def load_wav_as_16khz_int16(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"WAV file not found: {path}")

    try:
        from scipy.io import wavfile
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for --wav. Install it with: python3 -m pip install scipy"
        ) from exc

    sample_rate, audio = wavfile.read(path)
    if audio.ndim > 1:
        audio = audio.astype(np.float32).mean(axis=1)

    if np.issubdtype(audio.dtype, np.floating):
        audio = np.clip(audio, -1.0, 1.0) * np.iinfo(np.int16).max
    elif audio.dtype != np.int16:
        dtype_info = np.iinfo(audio.dtype)
        audio = audio.astype(np.float32) / max(abs(dtype_info.min), dtype_info.max)
        audio *= np.iinfo(np.int16).max

    audio = audio.astype(np.float32).reshape(-1)

    if sample_rate != SAMPLE_RATE:
        divisor = gcd(int(sample_rate), SAMPLE_RATE)
        audio = resample_poly(
            audio,
            SAMPLE_RATE // divisor,
            int(sample_rate) // divisor,
        )

    return np.clip(
        np.rint(audio),
        np.iinfo(np.int16).min,
        np.iinfo(np.int16).max,
    ).astype(np.int16)


def print_wav_missing_help(path: Path, model_name: str) -> None:
    print(f"WAV file not found: {path}", flush=True)
    if "YOUR_FILE" in str(path):
        print(
            "Replace YOUR_FILE.wav with the name of a real WAV file. "
            "That was only a placeholder.",
            flush=True,
        )

    training_dir = ROOT / "output" / model_name
    debug_wav = ROOT / "debug_input.wav"
    print("\nUseful checks:", flush=True)
    print(f"  python3 scripts/test_wakeword.py --model {model_name} --list-wavs", flush=True)
    print(f"  find {training_dir} -type f -name '*.wav' | head", flush=True)
    print(
        "  python3 scripts/test_wakeword.py "
        f"--model {model_name} --scores "
        "--save-debug-wav debug_input.wav --save-debug-seconds 12",
        flush=True,
    )
    print(f"\nExpected debug capture path: {debug_wav}", flush=True)


def list_wavs(model_name: str, limit: int = 40) -> None:
    ignored_parts = {
        ".git",
        ".pytest_cache",
        "__pycache__",
        "venv",
        ".venv",
    }
    search_roots = [
        ROOT / "output" / model_name,
        ROOT,
    ]
    seen: set[Path] = set()
    wav_paths: list[Path] = []

    for search_root in search_roots:
        if not search_root.exists():
            continue

        for wav_path in sorted(search_root.rglob("*.wav")):
            relative_parts = wav_path.relative_to(ROOT).parts
            if any(part in ignored_parts for part in relative_parts):
                continue
            if wav_path in seen:
                continue
            seen.add(wav_path)
            wav_paths.append(wav_path)

    if not wav_paths:
        print("No WAV files found in this CASE checkout.", flush=True)
        print(
            f"Expected synthetic clips under: {ROOT / 'output' / model_name}",
            flush=True,
        )
        print(
            "Copy the Colab/training output folders here, or create debug_input.wav "
            "with --save-debug-wav.",
            flush=True,
        )
        return

    print(f"Found {len(wav_paths)} WAV file(s). Showing up to {limit}:", flush=True)
    for wav_path in wav_paths[:limit]:
        print(f"  {wav_path.relative_to(ROOT)}", flush=True)

    print("\nExample:", flush=True)
    print(
        "  python3 scripts/test_wakeword.py "
        f"--wav {wav_paths[0].relative_to(ROOT)} --model {model_name} "
        "--scores --frame-scores",
        flush=True,
    )


def build_listener(args: argparse.Namespace, model_paths: list[Path]) -> WakeWordListener:
    return WakeWordListener(
        model_paths=model_paths,
        threshold=args.threshold,
        strong_threshold=args.strong_threshold,
        min_hits=args.min_hits,
        hit_window_sec=args.hit_window_sec,
        cooldown_seconds=args.cooldown_sec,
        print_scores=args.scores,
        input_gain=args.gain,
        debug_audio=args.debug_audio,
        save_debug_wav=args.save_debug_wav,
        save_debug_seconds=args.save_debug_seconds,
        frame_scores=args.frame_scores,
        record_false_positive_dir=args.record_false_positive_dir,
        input_device=args.device,
    )


def run_wav_test(
    listener: WakeWordListener,
    wav_path: Path,
    on_wakeword,
    quiet: bool = False,
) -> list[dict[str, float | str]]:
    audio = load_wav_as_16khz_int16(wav_path)
    if not quiet:
        print(
            f"Loaded WAV as {SAMPLE_RATE} Hz mono int16: "
            f"{wav_path} ({len(audio) / SAMPLE_RATE:.1f}s)",
            flush=True,
        )
    return listener.predict_resampled_audio(audio, on_wakeword)


def summarize_scores(label: str, file_scores: list[tuple[Path, float]]) -> None:
    scores = np.array([score for _, score in file_scores], dtype=np.float32)
    if len(scores) == 0:
        print(f"{label}: no WAV files tested", flush=True)
        return

    print(
        f"\n{label}: count={len(scores)}, "
        f"min={float(np.min(scores)):.6f}, "
        f"max={float(np.max(scores)):.6f}, "
        f"mean={float(np.mean(scores)):.6f}, "
        f"median={float(np.median(scores)):.6f}",
        flush=True,
    )
    threshold_counts = ", ".join(
        f">={threshold:.2f}: {int(np.sum(scores >= threshold))}"
        for threshold in SUMMARY_THRESHOLDS
    )
    print(f"Threshold counts: {threshold_counts}", flush=True)
    print("Top 10 highest scoring files:", flush=True)
    for path, score in sorted(file_scores, key=lambda item: item[1], reverse=True)[:10]:
        print(f"  {score:.6f}  {path}", flush=True)


def test_training_clips(args: argparse.Namespace, model_paths: list[Path]) -> None:
    base_dir = ROOT / "output" / args.model
    folder_names = [
        "positive_train",
        "positive_test",
        "negative_train",
        "negative_test",
    ]
    folders = [base_dir / name for name in folder_names]
    missing = [folder for folder in folders if not folder.is_dir()]
    if missing:
        print("Training output WAV folders are missing:", flush=True)
        for folder in missing:
            print(f"  - {folder}", flush=True)
        print(
            "Copy the output WAV folders from Colab to this CASE project, "
            "or run this test on the training machine.",
            flush=True,
        )
        print(
            f"Expected layout: {base_dir}/positive_test/*.wav "
            "and the matching train/negative folders.",
            flush=True,
        )
        return

    print(f"Testing synthetic training clips under {base_dir}", flush=True)
    results_by_folder: dict[str, list[tuple[Path, float]]] = {}

    def on_wakeword(model_name: str, score: float) -> None:
        pass

    for folder_name, folder in zip(folder_names, folders):
        wav_files = sorted(folder.glob("*.wav"))[:50]
        file_scores: list[tuple[Path, float]] = []

        for wav_path in wav_files:
            listener = WakeWordListener(
                model_paths=model_paths,
                threshold=1.1,
                cooldown_seconds=0.0,
                print_scores=False,
                input_gain=args.gain,
                debug_audio=False,
                frame_scores=False,
            )
            frame_results = run_wav_test(listener, wav_path, on_wakeword, quiet=True)
            max_score = max(
                (float(result["score"]) for result in frame_results),
                default=0.0,
            )
            file_scores.append((wav_path, max_score))

        results_by_folder[folder_name] = file_scores
        summarize_scores(folder_name, file_scores)

    debug_score = None
    debug_wav = ROOT / "debug_input.wav"
    if debug_wav.is_file():
        listener = WakeWordListener(
            model_paths=model_paths,
            threshold=1.1,
            cooldown_seconds=0.0,
            print_scores=False,
            input_gain=args.gain,
            debug_audio=False,
            frame_scores=False,
        )
        frame_results = run_wav_test(listener, debug_wav, on_wakeword, quiet=True)
        debug_score = max(
            (float(result["score"]) for result in frame_results),
            default=0.0,
        )
        print(f"\ndebug_input.wav max score: {debug_score:.6f}", flush=True)

    positive_scores = [
        score
        for folder in ("positive_train", "positive_test")
        for _, score in results_by_folder.get(folder, [])
    ]
    negative_scores = [
        score
        for folder in ("negative_train", "negative_test")
        for _, score in results_by_folder.get(folder, [])
    ]
    positive_max = max(positive_scores, default=0.0)
    negative_max = max(negative_scores, default=0.0)
    positive_median = float(np.median(np.array(positive_scores))) if positive_scores else 0.0
    negative_median = float(np.median(np.array(negative_scores))) if negative_scores else 0.0

    print("\nDiagnosis:", flush=True)
    print(
        f"Positive max={positive_max:.6f}, median={positive_median:.6f}; "
        f"negative max={negative_max:.6f}, median={negative_median:.6f}",
        flush=True,
    )
    if positive_max >= args.threshold and debug_score is not None and debug_score < args.threshold:
        print(
            "Model works on synthetic clips but does not generalize to this real voice/audio.",
            flush=True,
        )
    elif positive_max < args.threshold:
        print("Model or prediction path is likely wrong.", flush=True)
    else:
        print("Positive synthetic clips cross the threshold.", flush=True)

    if negative_max >= args.threshold:
        print("Model has false positive risk.", flush=True)
    if negative_max >= positive_median:
        print(
            "Negative outliers overlap the positive distribution. "
            "Inspect the top negative WAVs; they may be mislabeled, contaminated, "
            "or too similar to the wake phrase.",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    if args.record_false_positive_dir and (args.wav or args.test_training_clips):
        print(
            "--record-false-positive-dir is intended for live microphone testing only.",
            flush=True,
        )
        sys.exit(1)

    if args.list_wavs:
        list_wavs(args.model)
        return

    model_names = [args.model]

    model_paths = [MODEL_DIR / f"{name}.onnx" for name in model_names]
    print("Loading wake word model(s):", flush=True)
    for model_path in model_paths:
        print(f"  - {model_path.name}", flush=True)

    def on_wakeword(model_name: str, score: float) -> None:
        print(f"Wake word detected ({model_name}, score={score:.3f})", flush=True)

    if args.test_training_clips:
        test_training_clips(args, model_paths)
    elif args.wav:
        if not args.wav.is_file():
            print_wav_missing_help(args.wav, args.model)
            sys.exit(1)

        listener = build_listener(args, model_paths)
        run_wav_test(listener, args.wav, on_wakeword)
    else:
        listener = build_listener(args, model_paths)
        try:
            listener.listen_forever(on_wakeword)
        except KeyboardInterrupt:
            print("\nStopping wake word listener.", flush=True)
        finally:
            listener.stop()


if __name__ == "__main__":
    main()
