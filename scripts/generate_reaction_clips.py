#!/usr/bin/env python3
"""Generate pre-baked CASE reaction clips with the local Piper ONNX voice.

Examples:
  python3 scripts/generate_reaction_clips.py
  python3 scripts/generate_reaction_clips.py --force
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actuation.audio_output.tts_engine import (  # noqa: E402
    apply_gain_limited_pcm,
    fade_tts_pcm,
    trim_tts_silence_pcm,
)
from src.config import defaults  # noqa: E402
from src.persona.emotion import EmotionState, blend_tts_emotion_profile  # noqa: E402
from src.persona.reaction_clips import resolve_runtime_path  # noqa: E402
from src.voice_pipeline.piper_onnx_backend import PiperOnnxSynthesizer  # noqa: E402


REACTION_LENGTH_SCALE = {
    "angry": 1.05,
    "annoyed": 1.03,
    "sarcastic": 1.05,
    "amused": 1.03,
}


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_wav(path: Path, audio: bytes, sample_rate: int) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(audio)
    return len(audio) / 2.0 / float(sample_rate)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=defaults.CASE_REACTION_CLIPS_MANIFEST)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest_path = resolve_runtime_path(args.manifest, root=ROOT)
    if not manifest_path.is_file():
        raise SystemExit(f"reaction manifest missing: {manifest_path}")

    synthesizer = PiperOnnxSynthesizer(
        resolve_runtime_path(defaults.PIPER_MODEL_PATH, root=ROOT),
        resolve_runtime_path(defaults.PIPER_CONFIG_PATH, root=ROOT),
        length_scale=defaults.PIPER_LENGTH_SCALE,
        noise_scale=defaults.PIPER_NOISE_SCALE,
        noise_w=defaults.PIPER_NOISE_W,
    )
    manifest = load_manifest(manifest_path)
    clips = manifest.get("clips", {})
    if not isinstance(clips, dict):
        raise SystemExit("invalid reaction manifest: clips must be an object")

    for clip_id, item in clips.items():
        if item.get("enabled", True) is False:
            print(f"skip disabled clip={clip_id}")
            continue
        text = str(item.get("text", "")).strip()
        tts_text = str(item.get("tts_text", text)).strip()
        emotion = str(item.get("emotion", "deadpan")).strip().lower()
        path = resolve_runtime_path(str(item.get("path", "")), root=ROOT)
        if not text or not tts_text or not path:
            continue
        if str(clip_id).strip().lower() == "one_sec" or text.casefold().strip(".!? ") == "one sec" or tts_text.casefold().strip(".!? ") == "one sec":
            print(f"skip disabled clip={clip_id}")
            continue
        if path.exists() and not args.force:
            print(f"skip existing clip={clip_id} path={path}")
            continue

        state = EmotionState(
            emotion=emotion,
            intensity=0.85 if emotion in {"angry", "annoyed"} else 0.65,
            reason="ambiguous",
            confidence=1.0,
            source="reaction_generator",
            match=str(clip_id),
        )
        profile = blend_tts_emotion_profile(
            state,
            max_gain_db=defaults.CASE_TTS_EMOTION_MAX_GAIN_DB,
        )
        length_scale = REACTION_LENGTH_SCALE.get(emotion, max(1.0, profile.length_scale))
        audio, sample_rate = synthesizer.synthesize(
            tts_text,
            length_scale=length_scale,
        )
        audio, _gain_stats = apply_gain_limited_pcm(audio, profile.gain_db)
        audio, _trim_stats = trim_tts_silence_pcm(
            audio,
            sample_rate,
            threshold_db=defaults.CASE_TTS_TRIM_THRESHOLD_DB,
            keep_ms=defaults.CASE_TTS_TRIM_KEEP_MS,
        )
        audio = fade_tts_pcm(audio, sample_rate=sample_rate)
        duration = write_wav(path, audio, sample_rate)
        print(
            f'generated clip={clip_id} text="{text}" tts_text="{tts_text}" '
            f"emotion={emotion} length_scale={length_scale:.2f} "
            f"path={path} duration={duration:.2f}s"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
