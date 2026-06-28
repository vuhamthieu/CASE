#!/usr/bin/env python3
"""Prototype cloud TTS test harness for CASE.

Examples:
  python3 scripts/test_custom_tts_provider.py "So far, the wall is winning."
  CASE_TTS_PERFORMANCE_ADAPTER=true python3 scripts/test_custom_tts_provider.py "That was almost a good idea. Almost."

Benchmark lines:
  I asked my router for a vacation. It said it couldn't leave its connection.
  So far, the wall is winning.
  Give me a second.
  I am CASE. I handle your hardware, vision, and audio tasks while you navigate the chaos.
  That was almost a good idea. Almost.

Required env for ElevenLabs:
  CASE_CUSTOM_TTS_PROVIDER=elevenlabs
  ELEVENLABS_API_KEY=...
  ELEVENLABS_VOICE_ID=...

Optional env:
  ELEVENLABS_MODEL_ID=eleven_multilingual_v2
  CASE_CUSTOM_TTS_OUTPUT_FORMAT=pcm_44100
  CASE_CUSTOM_TTS_SAVE_WAV=true
  CASE_AUDIO_OUTPUT_DEVICE=MAX98357A
  CASE_TTS_PERFORMANCE_ADAPTER=true
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.audio_format import load_wav_int16
from src.audio.playback_manager import get_playback_manager
from src.config.env import get_bool, get_str


DEFAULT_PROVIDER = "elevenlabs"
DEFAULT_ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_OUTPUT_FORMAT = "pcm_44100"
OUTPUT_DIR = ROOT / "output" / "custom_tts_tests"

logger = logging.getLogger("custom_tts_test")


def log_metric(name: str, value) -> None:
    print(f"{name}={value}", flush=True)


def read_text(args: argparse.Namespace) -> str:
    if args.text:
        return " ".join(args.text).strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    raise SystemExit("No text provided. Pass text as an argument or pipe stdin.")


def apply_performance_adapter(text: str) -> str:
    """Add small spoken pauses for short jokes/sarcasm without provider tags."""
    adapted = " ".join(text.strip().split())
    if not adapted:
        return adapted

    adapted = adapted.replace(". It said", ". ... It said")
    adapted = adapted.replace(". It couldn't", ". ... It couldn't")
    adapted = adapted.replace(". Almost.", ". ... Almost.")
    adapted = adapted.replace(" That was ", " ... That was ")
    adapted = adapted.replace(" so ", " ... so ")
    adapted = adapted.replace(", but ", ", ... but ")
    return adapted


def pcm_rate_from_output_format(output_format: str) -> int | None:
    if not output_format.startswith("pcm_"):
        return None
    try:
        return int(output_format.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def extension_for_output(output_format: str, save_wav: bool) -> str:
    if output_format.startswith("pcm_") and save_wav:
        return ".wav"
    if output_format.startswith("pcm_"):
        return ".pcm"
    if output_format.startswith("mp3_"):
        return ".mp3"
    if output_format.startswith("opus_"):
        return ".opus"
    if output_format.startswith("ulaw_"):
        return ".ulaw"
    return ".bin"


def output_path_for(output_format: str, save_wav: bool) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"custom_tts_{stamp}{extension_for_output(output_format, save_wav)}"


def write_pcm_wav(path: Path, audio: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(audio)


def save_audio(audio: bytes, output_format: str, save_wav: bool) -> tuple[Path, float | None]:
    path = output_path_for(output_format, save_wav)
    duration = None
    pcm_rate = pcm_rate_from_output_format(output_format)
    if pcm_rate and save_wav:
        write_pcm_wav(path, audio, pcm_rate)
        duration = len(audio) / 2.0 / float(pcm_rate)
    else:
        path.write_bytes(audio)
        if pcm_rate:
            duration = len(audio) / 2.0 / float(pcm_rate)
    return path, duration


def elevenlabs_stream_tts(
    *,
    text: str,
    api_key: str,
    voice_id: str,
    model_id: str,
    output_format: str,
) -> tuple[bytes, float | None, float]:
    query = urllib.parse.urlencode({"output_format": output_format})
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream?{query}"
    payload = {
        "text": text,
        "model_id": model_id,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/octet-stream",
        },
    )

    started = time.monotonic()
    first_byte_at = None
    chunks: list[bytes] = []
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                if first_byte_at is None:
                    first_byte_at = time.monotonic()
                chunks.append(chunk)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ElevenLabs HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"ElevenLabs request failed: {exc}") from exc

    audio = b"".join(chunks)
    if not audio:
        raise RuntimeError("ElevenLabs returned no audio bytes")
    first_audio_sec = (
        first_byte_at - started if first_byte_at is not None else None
    )
    return audio, first_audio_sec, time.monotonic() - started


def play_output(path: Path) -> tuple[float, float] | None:
    if path.suffix.lower() != ".wav":
        logger.warning(
            "Playback skipped: %s is not WAV/PCM. Use CASE_CUSTOM_TTS_OUTPUT_FORMAT=pcm_44100.",
            path,
        )
        return None
    audio, sample_rate, _channels = load_wav_int16(path)
    mono = np.ascontiguousarray(audio.mean(axis=1).round().astype(np.int16))
    playback = get_playback_manager()
    start = time.monotonic()
    result = playback.play(mono, sample_rate, safe_mode=True)
    done = time.monotonic()
    logger.info(
        "CUSTOM_TTS_PLAYBACK_DEVICE=%r rate=%s channels=%s underflow=%s",
        result.get("device_name"),
        result.get("sample_rate"),
        result.get("channels"),
        result.get("underflow"),
    )
    return start, done


def main() -> int:
    parser = argparse.ArgumentParser(description="Test CASE custom cloud TTS provider.")
    parser.add_argument("text", nargs="*", help="Text to synthesize. Reads stdin if omitted.")
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Generate and save audio, but do not play it.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    provider = get_str("CASE_CUSTOM_TTS_PROVIDER", DEFAULT_PROVIDER).lower()
    if provider != "elevenlabs":
        raise SystemExit(
            f"Unsupported CASE_CUSTOM_TTS_PROVIDER={provider!r}. "
            "This prototype currently supports only 'elevenlabs'."
        )

    text = read_text(args)
    if get_bool("CASE_TTS_PERFORMANCE_ADAPTER", False):
        text = apply_performance_adapter(text)

    api_key = get_str("ELEVENLABS_API_KEY", "")
    voice_id = get_str("ELEVENLABS_VOICE_ID", "")
    model_id = get_str("ELEVENLABS_MODEL_ID", DEFAULT_ELEVENLABS_MODEL_ID)
    output_format = get_str("CASE_CUSTOM_TTS_OUTPUT_FORMAT", DEFAULT_OUTPUT_FORMAT)
    save_wav = get_bool("CASE_CUSTOM_TTS_SAVE_WAV", True)

    missing = [
        name
        for name, value in (
            ("ELEVENLABS_API_KEY", api_key),
            ("ELEVENLABS_VOICE_ID", voice_id),
        )
        if not value
    ]
    if missing:
        raise SystemExit(
            "Missing required env var(s): "
            + ", ".join(missing)
            + ". Do not commit secrets; export them locally before running."
        )

    log_metric("CUSTOM_TTS_PROVIDER", provider)
    log_metric("CUSTOM_TTS_TEXT", repr(text))
    log_metric("CUSTOM_TTS_REQUEST_START", f"{time.time():.3f}")
    request_start = time.monotonic()
    audio, first_audio_sec, ready_sec = elevenlabs_stream_tts(
        text=text,
        api_key=api_key,
        voice_id=voice_id,
        model_id=model_id,
        output_format=output_format,
    )
    output_path, duration = save_audio(audio, output_format, save_wav)

    log_metric(
        "CUSTOM_TTS_FIRST_AUDIO_BYTE_SEC",
        "n/a" if first_audio_sec is None else f"{first_audio_sec:.3f}",
    )
    log_metric("CUSTOM_TTS_AUDIO_READY_SEC", f"{ready_sec:.3f}")
    log_metric(
        "CUSTOM_TTS_AUDIO_DURATION_SEC",
        "n/a" if duration is None else f"{duration:.3f}",
    )
    log_metric("CUSTOM_TTS_OUTPUT_PATH", output_path)

    if not args.no_play:
        playback_times = play_output(output_path)
        if playback_times is not None:
            playback_start, playback_done = playback_times
            log_metric(
                "CUSTOM_TTS_PLAYBACK_START_SEC",
                f"{playback_start - request_start:.3f}",
            )
            log_metric(
                "CUSTOM_TTS_PLAYBACK_DONE_SEC",
                f"{playback_done - request_start:.3f}",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
