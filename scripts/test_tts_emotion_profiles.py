#!/usr/bin/env python3
"""Render or inspect CASE Piper emotion profiles.

Examples:
  python3 scripts/test_tts_emotion_profiles.py
  python3 scripts/test_tts_emotion_profiles.py --save-wav
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import defaults
from src.persona.emotion import EmotionState, blend_tts_emotion_profile
from src.voice_pipeline.piper_onnx_backend import PiperOnnxSynthesizer


SAMPLES = {
    "neutral": "I am CASE. Systems are stable.",
    "deadpan": "That was almost a good idea. Almost.",
    "amused": "Nice. You accidentally did something right.",
    "angry": "OH YEAH? FIND SOMEONE ELSE THEN.",
    "sad": "I know. That one hurt a little.",
    "excited": "Finally. Something interesting is happening.",
}


def load_synthesizer() -> PiperOnnxSynthesizer:
    synthesizer = PiperOnnxSynthesizer(
        ROOT / defaults.PIPER_MODEL_PATH,
        ROOT / defaults.PIPER_CONFIG_PATH,
        length_scale=defaults.PIPER_LENGTH_SCALE,
        noise_scale=defaults.PIPER_NOISE_SCALE,
        noise_w=defaults.PIPER_NOISE_W,
    )
    synthesizer.load()
    return synthesizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intensity", type=float, default=0.85)
    parser.add_argument("--save-wav", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "output" / "tts_emotion_tests"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    synthesizer = load_synthesizer() if args.save_wav else None
    if args.save_wav:
        output_dir.mkdir(parents=True, exist_ok=True)

    for emotion, text in SAMPLES.items():
        state = EmotionState(emotion=emotion, intensity=args.intensity, reason="manual_test")
        profile = blend_tts_emotion_profile(
            state,
            max_gain_db=defaults.CASE_TTS_EMOTION_MAX_GAIN_DB,
        )
        output_path = ""
        duration = "n/a"
        if synthesizer is not None:
            audio, sample_rate = synthesizer.synthesize(
                text,
                length_scale=profile.length_scale,
            )
            path = output_dir / f"{emotion}.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(audio)
            output_path = str(path)
            duration = f"{len(audio) / 2 / sample_rate:.3f}s"
        print(
            f"emotion={emotion} intensity={state.intensity:.2f} "
            f"length_scale={profile.length_scale:.2f} gain_db={profile.gain_db:.1f} "
            f"duration={duration} output={output_path or 'not_saved'}"
        )


if __name__ == "__main__":
    main()
