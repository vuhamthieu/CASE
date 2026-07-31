#!/usr/bin/env python3
"""Generate offline Gemini TTS voice samples for subjective CASE research."""

import argparse
import json
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.voice_fx import VoiceFX
from src.config.env import get_str
from src.realtime.realtime_config import CASE_RECOMMENDED_GEMINI_VOICES

SAMPLE_LINES = [
    "Honesty: ninety percent. Humor: seventy-five percent. Self-preservation: unfortunately still online.",
    "That was not a command. That was a noise with ambition.",
    "I see you. Centered. Mostly.",
    "Running diagnostics. Nothing is on fire. Yet.",
    "Yes, I can explain quantum physics. No, I cannot make it emotionally healthy.",
]


def write_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(pcm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voices", nargs="+", default=CASE_RECOMMENDED_GEMINI_VOICES)
    parser.add_argument("--fx-preset", default="cinematic_robot_v1")
    parser.add_argument("--apply-fx", action="store_true")
    parser.add_argument("--model", default="gemini-3.1-flash-tts-preview")
    parser.add_argument("--output-dir", type=Path, default=Path("research/voice_samples"))
    parser.add_argument("--text")
    parser.add_argument("--wake-ack", action="store_true")
    parser.add_argument("--voice")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    api_key = get_str("GEMINI_API_KEY", "")
    if not api_key:
        print("VOICE_RESEARCH: GEMINI_API_KEY is missing; no samples generated.")
        return
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("VOICE_RESEARCH: google-genai is not installed.")
        return

    text = args.text or ("I'm listening." if args.wake_ack else " ".join(SAMPLE_LINES))
    voices = [args.voice] if args.wake_ack and args.voice else args.voices
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    client = genai.Client(api_key=api_key)
    manifest = []
    for voice in voices:
        print(f"VOICE_RESEARCH: generating voice={voice}")
        response = client.models.generate_content(
            model=args.model,
            contents=(
                "Read exactly this text in a calm, low-energy, dry, precise onboard-computer style. "
                + text
            ),
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                    )
                ),
            ),
        )
        pcm = response.candidates[0].content.parts[0].inline_data.data
        raw_path = output_dir / f"{voice.lower()}_raw.wav"
        if args.wake_ack and args.output:
            raw_path = args.output if args.output.is_absolute() else ROOT / args.output
        write_wav(raw_path, pcm)
        manifest.append({
            "voice": voice,
            "preset": "raw",
            "sample_text": text,
            "output_path": str(raw_path.relative_to(ROOT)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "notes": "",
        })
        if args.apply_fx:
            fx_pcm = VoiceFX(24_000, args.fx_preset).process_int16_mono(pcm)
            fx_path = output_dir / f"{voice.lower()}_{args.fx_preset}.wav"
            if args.wake_ack and args.output:
                fx_path = args.output if args.output.is_absolute() else ROOT / args.output
            write_wav(fx_path, fx_pcm)
            manifest.append({**manifest[-1], "preset": args.fx_preset, "output_path": str(fx_path.relative_to(ROOT))})
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"VOICE_RESEARCH: manifest={output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
