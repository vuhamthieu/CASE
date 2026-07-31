# CASE Voice Lab

The Voice Lab develops an original CASE cinematic-robot voice: calm, dry,
precise, low-energy, and intelligible. It does not clone or imitate actors,
celebrities, or copyrighted characters.

## Three Separate Layers

1. **Gemini Live voice** controls realtime conversation timbre.
2. **Gemini TTS samples** produce repeatable offline lines for A/B evaluation.
3. **VoiceFX DSP** applies lightweight local EQ, compression, filtering, and
   saturation to 24 kHz mono PCM.

Hybrid text plus local CASE TTS is the current runtime default. Gemini remains
the text brain while the established local CASE voice handles every spoken
reply. Gemini Live native voice remains an optional fast/debug path.

The future research direction is a fully streamed chain:

```text
streaming STT -> Gemini text LLM -> expressive streaming TTS
```

That pipeline can remain responsive when every stage streams, but it is not a
production runtime option yet. Selecting `streaming_tts_research` without a
configured provider safely falls back to `gemini_live_native`.

Realtime v1 intentionally excludes pitch/formant conversion and neural voice
conversion. RVC, OpenVoice, and LLVC may be investigated later on a laptop or
server, but are not Raspberry Pi runtime dependencies.

## CASE Voice Identity

CASE is an original voice and personality: calm, precise, dry, useful, and
slightly sarcastic. It does not clone real actors or imitate copyrighted
characters. The default `case_mate_deadpan_v1` persona has configurable humor,
honesty, sarcasm, and response length rather than denying that humor exists.

Test it without starting the robot runtime:

```bash
python3 scripts/test_case_persona.py --preset case_mate_deadpan_v1
```

## Runtime Voice Pipelines

Optional fast Gemini native audio:

```bash
CASE_VOICE_PIPELINE=gemini_live_native python3 main.py
```

Consistent wake and conversation voice through local CASE TTS:

```bash
CASE_VOICE_PIPELINE=hybrid_text_tts \
VOICE_OUTPUT_BACKEND=piper_onnx \
TRANSCRIPT_INPUT_BACKEND=sherpa_sensevoice \
HYBRID_LATENCY_PROFILE=fast \
CASE_VOICE_PRESET=case_mate_deadpan_v1 \
python3 main.py
```

Random acknowledgements use `,` or `|` separators in `WAKE_ACK_POOL`.
Matching cached files belong in `assets/audio/wake_ack/generated/`; missing
variants fall back to live Piper TTS and finally a beep.

Setting `WAKE_ACK_USE_VOICE_BACKEND=true` makes random acknowledgements use
live `piper_onnx`. The fast profile instead defaults to cached WAV
acknowledgements generated with the same
local voice, avoiding live synthesis latency. Native Gemini audio is not opened
or played.

The stable runtime wake acknowledgement set is intentionally small and
generated as cached WAVs:

```bash
WAKE_ACK_MODE=cached_wav
WAKE_ACK_RECORDED_ENABLED=false
WAKE_ACK_ALLOW_SHORT_INTERJECTIONS=false
WAKE_ACK_PROFILE=clear_short
WAKE_ACK_POOL=yes,im_listening
python3 scripts/generate_wake_ack_wavs.py --backend piper_onnx --profile clear_short --force
python3 scripts/inspect_wake_ack_wavs.py
python3 scripts/test_wake_ack_runtime.py --all-default --play
```

The folder contract is:

```text
generated/           Piper fallback WAVs
candidates/          Piper audition candidates
_archive_recorded_experiment/
                     archived recorded experiment assets, if preserved
```

Runtime tries generated Piper cache, live Piper TTS, and finally a beep.
Recorded assets are not attempted unless explicitly re-enabled. Archive old
recorded files without deleting them:

```bash
python3 scripts/cleanup_wake_ack_experiments.py --archive
```

A future expressive CASE voice fine-tune may reduce the need for separate
interjection handling.

The default runtime pool is exactly `Yes!` and `I'm listening.`.
Old keys such as `You called?`, `Go on.`, `I'm here.`, `Say that again?`,
`Still with you.`, `What?`, and `Yeah?` are not selected by default.

The fast profile also limits ordinary replies to two short local-TTS chunks.
Explicit requests for a detailed explanation or story may exceed that limit.

## Realtime FX

Enable the default effect:

```bash
GEMINI_LIVE_VOICE_NAME=Schedar
CASE_VOICE_PRESET=case_cinematic_robot
CASE_VOICE_FX_ENABLED=true
CASE_VOICE_FX_PRESET=cinematic_robot_v1
python3 main.py
```

Disable it for an exact PCM bypass:

```bash
CASE_VOICE_FX_ENABLED=false python3 main.py
```

Available presets are `bypass`, `cinematic_robot_v1`, `dry_computer_v1`, and
`dark_robot_v1`.

To dump the last raw and processed model response:

```bash
CASE_VOICE_FX_ENABLED=true CASE_VOICE_FX_DUMP_WAV=true python3 main.py
```

Files are saved under `output/voice_fx_debug/`. Dumps are disabled by default.

## Offline Listening Tests

Apply an effect to an existing realtime response:

```bash
python3 scripts/test_voice_fx.py \
  --input output/realtime_debug/last_model_response.wav \
  --preset cinematic_robot_v1 \
  --output output/voice_fx_debug/test_fx.wav
```

Generate comparable Gemini TTS samples:

```bash
python3 scripts/research_voice_samples.py \
  --voices Schedar Algenib Orus Kore Charon \
  --fx-preset cinematic_robot_v1 --apply-fx
```

Rate samples with `research/voice_samples/evaluation_template.md`.

## Wake Acknowledgement Consistency

A local Piper acknowledgement may sound different from Gemini Live. Use a beep
to avoid that mismatch:

```bash
REALTIME_WAKE_ACK_MODE=beep python3 main.py
```

Or generate a matching cached acknowledgement with Gemini TTS:

```bash
python3 scripts/research_voice_samples.py --wake-ack --voice Schedar \
  --text "I'm listening." \
  --output assets/audio/realtime_wake_ack.wav \
  --apply-fx --fx-preset cinematic_robot_v1
```

For a future expressive provider, generate the cached acknowledgement with the
same selected voice. Avoid mixing local TTS with an expressive runtime voice
unless the mismatch is acceptable.

## Expressive TTS Research

The interfaces in `src/voice_pipeline/` reserve clean boundaries for streaming
STT, LLM, TTS, and events without adding paid-provider dependencies. Inspect a
provider configuration safely with:

```bash
python3 scripts/research_expressive_tts.py --provider none
python3 scripts/research_expressive_tts.py --provider elevenlabs \
  --text "Operational. Mildly disappointed, but stable."
```

Recommended sequence: improve the persona, compare Gemini voices, evaluate
external TTS in research scripts, and only then consider production integration.

## Voice Stability Fixes

### Guillotine Effect (Audio Cutoff)
When playing PCM audio chunks using the `sounddevice` backend, calling `.stop()` on the output stream immediately after `write()` returns would cut off the end of speech (e.g. "expensive" -> "expensi..."). This happens because `write()` is a blocking write to the sound device's internal ring buffer, but does not block until the physical hardware (DAC/amplifier) finishes playing the queued samples.

To fix this, we modified the `AudioPlaybackManager.play()` method in `src/audio/playback_manager.py` to query the actual stream output latency and ensure that the playback thread waits (`time.sleep()`) for the full duration of the latency plus a 30ms safety buffer before calling `stop()`.

### Caffeine Rush (Punctuation and Fast Speech)
Streamed LLM responses can sometimes omit spaces after punctuation (e.g., `learning.It` or `market.They`). Because the sentence-splitting regex in `ResponseChunker` and `clean_tts_text` relies on whitespace to identify sentence boundaries, these merged phrases were treated as a single continuous text chunk. This caused Piper to omit natural pauses and speak sentences too quickly.

To fix this, we added regex-based preprocessing that inserts missing spaces between punctuation marks and subsequent capital letters (e.g., `learning.It` -> `learning. It`), while leaving abbreviation patterns (e.g., `U.S.`, `A.B.C.`) untouched. This allows both streaming chunks and pre-synthesis texts to split cleanly at sentence boundaries and maintain standard speech cadences.

