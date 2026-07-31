#!/usr/bin/env python3
"""Exercise cached acknowledgements and padded CASE TTS PCM playback."""

import argparse
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from actuation.audio_output.tts_engine import PIPER_SAMPLE_RATE, pad_and_fade_tts_pcm
from src.audio.playback_manager import close_playback_manager
from src.audio.output_device import play_int16_mono

PIPER = ROOT / "ai" / "tts" / "piper" / "piper"
MODEL = ROOT / "ai" / "tts" / "en_US-ryan-medium.onnx"
WAKE_DIR = ROOT / "assets" / "audio" / "wake_ack"
WAKE_FILES = (
    "what.wav", "yeah.wav", "im_listening.wav",
    "you_called.wav", "go_on.wav", "im_here.wav",
    "say_that_again.wav", "still_with_you.wav",
)
TEST_PHRASES = {
    "short_phrase.wav": "Yeah?",
    "normal_sentence.wav": "My sensors were already calibrated for your boredom.",
    "two_sentences.wav": "Operational. Mildly offended by the room acoustics.",
}


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2 or source.getnchannels() != 1:
            raise ValueError(f"expected mono 16-bit PCM: {path}")
        rate = source.getframerate()
        audio = np.frombuffer(
            source.readframes(source.getnframes()), dtype="<i2"
        ).copy()
    return rate, audio


def write_wav(path: Path, audio: bytes, rate: int = PIPER_SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(audio)


def synthesize(text: str) -> bytes | None:
    if not PIPER.is_file() or not MODEL.is_file():
        return None
    try:
        result = subprocess.run(
            [str(PIPER), "--model", str(MODEL), "--output_raw"],
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        print(f"AUDIO_TEST: local Piper unavailable on this machine: {exc}")
        return None
    return result.stdout if result.returncode == 0 and result.stdout else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-play", action="store_true")
    parser.add_argument("--write-padded", action="store_true")
    parser.add_argument(
        "--device",
        help="PortAudio output device index or name from `python3 -m sounddevice`.",
    )
    args = parser.parse_args()
    if args.device:
        os.environ["CASE_AUDIO_OUTPUT_DEVICE"] = args.device

    samples: list[tuple[str, int, np.ndarray]] = []
    for filename in WAKE_FILES:
        path = WAKE_DIR / filename
        if not path.is_file():
            print(f"AUDIO_TEST: {filename} missing")
            continue
        rate, audio = read_wav(path)
        duration = len(audio) / rate
        status = "ok" if duration >= 1.1 else "too short"
        print(f"AUDIO_TEST: {filename} duration={duration:.2f}s {status}")
        samples.append((filename, rate, audio))

    debug_dir = ROOT / "output" / "tts_debug"
    for filename, text in TEST_PHRASES.items():
        raw = synthesize(text)
        if raw is None:
            continue
        padded = pad_and_fade_tts_pcm(raw)
        audio = np.frombuffer(padded, dtype="<i2").copy()
        path = debug_dir / filename
        if args.write_padded:
            write_wav(path, padded)
        print(f"AUDIO_TEST: {filename} duration={len(audio) / PIPER_SAMPLE_RATE:.2f}s ok")
        samples.append((filename, PIPER_SAMPLE_RATE, audio))

    if args.no_play or not samples:
        print("AUDIO_TEST: playback skipped")
        return
    underflow = False
    for name, rate, audio in samples:
        result = play_int16_mono(
            audio,
            rate,
            post_guard_sec=0.15,
        )
        underflow = bool(result["underflow"]) or underflow
        print(
            f"AUDIO_TEST: played {name} device={result['device_name']!r} "
            f"rate={result['sample_rate']} channels={result['channels']}"
        )
    print(
        "AUDIO_TEST: underflow detected"
        if underflow
        else "AUDIO_TEST: no underflow detected"
    )
    close_playback_manager()


if __name__ == "__main__":
    main()
