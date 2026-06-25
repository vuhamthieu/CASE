#!/usr/bin/env python3
"""Generate, audition, and select expressive CASE wake acknowledgements."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.audio_format import convert_channels, load_wav_int16, resample_audio
from src.audio.output_device import play_int16_mono
from src.audio.playback_manager import close_playback_manager
from src.audio.wake_ack_audio import inspect_wake_ack, prepare_wake_ack_audio
from src.audio.wake_ack_fx import apply_wake_ack_fx
from src.config import defaults
from src.voice_pipeline.piper_onnx_backend import PiperOnnxSynthesizer


PIPER = ROOT / "ai" / "tts" / "piper" / "piper"
MODEL = ROOT / "ai" / "tts" / "en_US-ryan-medium.onnx"
OUTPUT_DIR = ROOT / "assets" / "audio" / "wake_ack"
PIPER_SAMPLE_RATE = 22_050
SAMPLE_RATE = PIPER_SAMPLE_RATE
LEADING_SILENCE_MS = defaults.WAKE_ACK_PRE_SILENCE_MS
TRAILING_SILENCE_MS = defaults.WAKE_ACK_POST_SILENCE_MS
FADE_IN_MS = defaults.WAKE_ACK_FADE_IN_MS
FADE_OUT_MS = defaults.WAKE_ACK_FADE_OUT_MS
MIN_DURATION_MS = defaults.WAKE_ACK_MIN_DURATION_MS
MIN_VOICED_MS = defaults.WAKE_ACK_MIN_VOICED_MS
DEFAULT_WAKE_ACK_POOL = tuple(defaults.DEFAULT_WAKE_ACK_POOL)
LEGACY_WAKE_ACK_KEYS = (
    "you_called",
    "im_listening",
    "go_on",
    "im_here",
    "say_that_again",
    "still_with_you",
    "what",
    "yeah",
)

WAKE_ACK_VARIANTS = {
    "yes": [defaults.WAKE_ACK_TEXTS["yes"], "Yes."],
    "im_listening": [defaults.WAKE_ACK_TEXTS["im_listening"], "Listening."],
    "you_called": ["You called?", "You called."],
    "go_on": ["Go on.", "Go on, I'm listening."],
    "im_here": ["I'm here.", "Right here."],
    "say_that_again": ["Say that again?", "One more time?"],
    "still_with_you": ["Still with you.", "I'm still here."],
    "what": ["What?", "What is it?"],
    "yeah": ["Yeah?", "Yeah."],
}

CLEAR_SHORT_WAKE_ACK_STYLE = {
    "length_scale": 1.08,
    "noise_scale": 0.55,
    "noise_w": 0.65,
    "pitch_shift_semitones": 0.0,
    "tempo": 0.98,
    "gain_db": 0.0,
}
SLOW_CLEAR_WAKE_ACK_STYLE = {
    "length_scale": 1.18,
    "noise_scale": 0.55,
    "noise_w": 0.65,
    "pitch_shift_semitones": 0.0,
    "tempo": 0.96,
    "gain_db": 0.0,
}
DEFAULT_WAKE_ACK_STYLE = dict(SLOW_CLEAR_WAKE_ACK_STYLE)
NEUTRAL_WAKE_ACK_STYLE = {
    "length_scale": 1.0,
    "noise_scale": defaults.PIPER_NOISE_SCALE,
    "noise_w": defaults.PIPER_NOISE_W,
    "pitch_shift_semitones": 0.0,
    "tempo": 1.0,
    "gain_db": 0.0,
}
VARIANT_STYLE_OVERRIDES = {
    ("what", 1): {"tempo": 0.90},
    ("yeah", 1): {"tempo": 0.90},
}
PROFILE_SELECTIONS = {
    "neutral": {name: 1 for name in WAKE_ACK_VARIANTS},
    "clear_short": {
        "yes": 1,
        "im_listening": 1,
    },
    "slow_clear": {
        "yes": 1,
        "im_listening": 1,
    },
    "natural": {
        "what": 1,
        "yeah": 1,
        "you_called": 1,
        "im_listening": 1,
        "go_on": 1,
        "im_here": 1,
        "say_that_again": 1,
        "still_with_you": 1,
    },
    "surprised": {
        "what": 1,
        "yeah": 1,
        "you_called": 2,
        "im_listening": 1,
        "go_on": 1,
        "im_here": 1,
        "say_that_again": 1,
        "still_with_you": 1,
    },
}
_cli_style_warning_logged = False


def style_for(profile: str, canonical: str, candidate_number: int) -> dict[str, float]:
    if profile == "neutral":
        style = dict(NEUTRAL_WAKE_ACK_STYLE)
    elif profile == "clear_short":
        style = dict(CLEAR_SHORT_WAKE_ACK_STYLE)
    elif profile == "slow_clear":
        style = dict(SLOW_CLEAR_WAKE_ACK_STYLE)
    else:
        style = dict(DEFAULT_WAKE_ACK_STYLE)
    style.update(VARIANT_STYLE_OVERRIDES.get((canonical, candidate_number), {}))
    style["length_scale"] = max(1.0, float(style["length_scale"]))
    style["tempo"] = min(1.0, float(style["tempo"]))
    style["pitch_shift_semitones"] = float(style.get("pitch_shift_semitones", 0.0))
    return style


def pad_and_fade_pcm(
    raw_pcm: bytes,
    sample_rate: int = SAMPLE_RATE,
    source_sample_rate: int = PIPER_SAMPLE_RATE,
) -> bytes:
    samples = np.frombuffer(raw_pcm, dtype="<i2")
    if samples.size == 0:
        return raw_pcm
    if source_sample_rate != sample_rate:
        samples = resample_audio(samples, source_sample_rate, sample_rate)[:, 0]
    padded, _ = prepare_wake_ack_audio(samples, sample_rate)
    return padded.tobytes()


def write_and_verify(path: Path, audio: np.ndarray, sample_rate: int) -> dict:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(np.ascontiguousarray(audio, dtype="<i2").tobytes())
    verified, header_rate, channels = load_wav_int16(path)
    mono = convert_channels(verified, 1)[:, 0]
    stats = inspect_wake_ack(mono, header_rate)
    if header_rate != sample_rate or channels != 1 or not stats.passed:
        raise RuntimeError(
            f"wake ack verification failed path={path} rate={header_rate} "
            f"channels={channels} voiced_ms={stats.voiced_ms:.0f} "
            f"passed={stats.passed}"
        )
    return {
        "sample_rate": header_rate,
        "channels": channels,
        "duration_sec": stats.duration_sec,
        "voiced_ms": stats.voiced_ms,
        "leading_silence_ms": stats.leading_silence_ms,
        "trailing_silence_ms": stats.trailing_silence_ms,
        "peak_dbfs": stats.peak_dbfs,
        "clipped": stats.clipped,
    }


def _stretch_to_min_voiced(
    speech: np.ndarray,
    sample_rate: int,
    *,
    canonical: str,
) -> np.ndarray:
    padded, _ = prepare_wake_ack_audio(speech, sample_rate)
    stats = inspect_wake_ack(padded, sample_rate)
    if stats.voiced_ms >= MIN_VOICED_MS:
        return speech
    factor = min(2.5, max(1.0, (MIN_VOICED_MS / max(stats.voiced_ms, 1.0)) * 1.03))
    target_frames = int(round(len(speech) * factor))
    try:
        from scipy.signal import resample

        stretched = resample(speech.astype(np.float32), target_frames)
    except ImportError:
        source_positions = np.arange(len(speech), dtype=np.float64)
        target_positions = np.linspace(0.0, len(speech) - 1.0, target_frames)
        stretched = np.interp(target_positions, source_positions, speech)
    return np.ascontiguousarray(
        np.clip(np.rint(stretched), -32768, 32767).astype("<i2")
    )


def synthesize_cli(text: str, style: dict[str, float]) -> tuple[bytes, int]:
    global _cli_style_warning_logged
    styled_command = [
        str(PIPER),
        "--model",
        str(MODEL),
        "--output_raw",
        "--length-scale",
        str(style["length_scale"]),
        "--noise-scale",
        str(style["noise_scale"]),
        "--noise-w",
        str(style["noise_w"]),
    ]
    result = subprocess.run(
        styled_command,
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        if not _cli_style_warning_logged:
            print("WAKE_ACK_CACHE: Piper CLI style flags unavailable; using defaults")
            _cli_style_warning_logged = True
        result = subprocess.run(
            [str(PIPER), "--model", str(MODEL), "--output_raw"],
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    if result.returncode != 0 or not result.stdout:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Piper CLI failed for {text!r}: {error}")
    return result.stdout, PIPER_SAMPLE_RATE


def generate_candidate(
    path: Path,
    text: str,
    style: dict[str, float],
    *,
    synthesizer: PiperOnnxSynthesizer | None,
    output_sample_rate: int,
) -> dict:
    if synthesizer is not None:
        raw_pcm, source_rate = synthesizer.synthesize(
            text,
            length_scale=style["length_scale"],
            noise_scale=style["noise_scale"],
            noise_w=style["noise_w"],
        )
    else:
        raw_pcm, source_rate = synthesize_cli(text, style)
    speech = np.frombuffer(raw_pcm, dtype="<i2").copy()
    speech = apply_wake_ack_fx(speech, source_rate, style)
    if source_rate != output_sample_rate:
        speech = resample_audio(speech, source_rate, output_sample_rate)[:, 0]
    speech = _stretch_to_min_voiced(
        speech,
        output_sample_rate,
        canonical=path.stem.split("__", 1)[0],
    )
    padded, _ = prepare_wake_ack_audio(speech, output_sample_rate)
    stats = write_and_verify(path, padded, output_sample_rate)
    print(
        f"WAKE_ACK_CANDIDATE: path={path} text={text!r} "
        f"rate={stats['sample_rate']} duration={stats['duration_sec']:.3f}s "
        f"voiced={stats['voiced_ms']:.0f}ms "
        f"lead={stats['leading_silence_ms']:.0f}ms "
        f"tail={stats['trailing_silence_ms']:.0f}ms "
        f"peak={stats['peak_dbfs']:.1f}dBFS clipped={stats['clipped']}"
    )
    return stats


def audition_candidates(output_dir: Path, canonical_keys: list[str]) -> dict[str, int]:
    selections: dict[str, int] = {}
    for canonical in canonical_keys:
        variants = WAKE_ACK_VARIANTS[canonical]
        print(f"\n[{canonical}]")
        for index, text in enumerate(variants, 1):
            path = output_dir / "candidates" / f"{canonical}__{index:02d}.wav"
            print(f"{index}) {text}")
            audio, sample_rate, _ = load_wav_int16(path)
            play_int16_mono(audio, sample_rate, post_guard_sec=0.20, safe_mode=True)
        while True:
            answer = input(f"Choose best [1-{len(variants)}]: ").strip()
            if answer.isdigit() and 1 <= int(answer) <= len(variants):
                selections[canonical] = int(answer)
                break
            print("Enter one of the listed candidate numbers.")
    close_playback_manager()
    return selections


def install_selections(
    output_dir: Path,
    selections: dict[str, int],
    profile: str,
) -> None:
    manifest = {"profile": profile, "selections": {}}
    generated_dir = output_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    for canonical, candidate_number in selections.items():
        source = output_dir / "candidates" / f"{canonical}__{candidate_number:02d}.wav"
        destination = generated_dir / f"{canonical}.wav"
        shutil.copy2(source, destination)
        text = WAKE_ACK_VARIANTS[canonical][candidate_number - 1]
        manifest["selections"][canonical] = {
            "candidate": source.name,
            "selected_wav": destination.name,
            "text": text,
            "style": style_for(profile, canonical, candidate_number),
        }
        print(
            f"WAKE_ACK_SELECTED: canonical={canonical} candidate={source.name} "
            f"text={text!r} path={destination}"
        )
    manifest_path = output_dir / "wake_ack_selection.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"WAKE_ACK_SELECTED: manifest={manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--play-test", action="store_true")
    parser.add_argument("--audition", action="store_true")
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Also generate old optional wake acknowledgements outside the active pool.",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_SELECTIONS),
        default=defaults.WAKE_ACK_PROFILE,
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--backend",
        choices=("piper_onnx", "local_case_tts"),
        default=defaults.VOICE_OUTPUT_BACKEND,
    )
    parser.add_argument("--model", type=Path, default=Path(defaults.PIPER_MODEL_PATH))
    parser.add_argument("--config", type=Path, default=Path(defaults.PIPER_CONFIG_PATH))
    parser.add_argument(
        "--output-sample-rate",
        type=int,
        default=None,
        help="WAV output rate; defaults to the loaded Piper model's native rate",
    )
    args = parser.parse_args()
    if args.output_sample_rate is not None and args.output_sample_rate <= 0:
        parser.error("--output-sample-rate must be positive")

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    protected_dirs = tuple(
        (ROOT / configured).resolve()
        for configured in (
            defaults.WAKE_ACK_RECORDED_RAW_DIR,
            defaults.WAKE_ACK_RECORDED_PROCESSED_DIR,
            defaults.WAKE_ACK_RECORDED_DIR,
        )
    )
    resolved_output = output_dir.resolve()
    protected = next(
        (
            directory
            for directory in protected_dirs
            if resolved_output == directory or directory in resolved_output.parents
        ),
        None,
    )
    if protected is not None:
        parser.error(
            "generated wake acknowledgements cannot be written inside the "
            f"protected recorded asset directory: {protected}"
        )
    if args.dry_run:
        dry_run_keys = list(DEFAULT_WAKE_ACK_POOL)
        if args.include_legacy:
            dry_run_keys.extend(LEGACY_WAKE_ACK_KEYS)
        for canonical in dry_run_keys:
            variants = WAKE_ACK_VARIANTS[canonical]
            for index, text in enumerate(variants, 1):
                print(f"{canonical}__{index:02d}.wav <- {text!r}")
        return

    model_path = args.model if args.model.is_absolute() else ROOT / args.model
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    required = (model_path, config_path) if args.backend == "piper_onnx" else (PIPER, MODEL)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise SystemExit("\n".join(f"WAKE_ACK_CACHE: missing {path}" for path in missing))

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = output_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    synthesizer = None
    output_sample_rate = args.output_sample_rate or PIPER_SAMPLE_RATE
    if args.backend == "piper_onnx":
        synthesizer = PiperOnnxSynthesizer(
            model_path,
            config_path,
            length_scale=defaults.PIPER_LENGTH_SCALE,
            noise_scale=defaults.PIPER_NOISE_SCALE,
            noise_w=defaults.PIPER_NOISE_W,
        )
        try:
            synthesizer.load()
        except Exception as exc:
            raise SystemExit(f"WAKE_ACK_CACHE: Piper ONNX load failed: {exc}") from exc
        if args.output_sample_rate is None:
            output_sample_rate = synthesizer.sample_rate

    generation_keys = list(DEFAULT_WAKE_ACK_POOL)
    if args.include_legacy:
        generation_keys.extend(LEGACY_WAKE_ACK_KEYS)
    for canonical in generation_keys:
        variants = WAKE_ACK_VARIANTS[canonical]
        for index, text in enumerate(variants, 1):
            path = candidates_dir / f"{canonical}__{index:02d}.wav"
            if path.exists() and not args.force:
                print(f"WAKE_ACK_CANDIDATE: keeping {path} (use --force to replace)")
                continue
            generate_candidate(
                path,
                text,
                style_for(args.profile, canonical, index),
                synthesizer=synthesizer,
                output_sample_rate=output_sample_rate,
            )

    selections = (
        audition_candidates(output_dir, generation_keys)
        if args.audition
        else {
            key: value
            for key, value in PROFILE_SELECTIONS[args.profile].items()
            if key in generation_keys
        }
    )
    install_selections(output_dir, selections, args.profile)

    if args.play_test:
        for canonical in generation_keys:
            path = output_dir / "generated" / f"{canonical}.wav"
            audio, sample_rate, _ = load_wav_int16(path)
            print(f"WAKE_ACK_CACHE: play-test {canonical} path={path}")
            play_int16_mono(audio, sample_rate, post_guard_sec=0.20, safe_mode=True)
        close_playback_manager()


if __name__ == "__main__":
    main()
