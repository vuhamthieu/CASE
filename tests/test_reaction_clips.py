import asyncio
import json
import tempfile
import unittest
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from actuation.audio_output.tts_engine import CASEVoice
from src.cognition.personality import CASEPersonality
from src.config import defaults
from src.persona.emotion import EmotionState
from src.persona.reaction_clips import (
    ReactionClip,
    ReactionClipSelector,
    disabled_clip_ids,
    load_reaction_manifest,
    strip_leading_reaction_duplicate,
)


class FakeBus:
    def __init__(self):
        self.events = []

    async def publish(self, topic, payload=None):
        self.events.append((topic, payload))

    def subscribe(self, topic, callback):
        pass


def write_silent_wav(path: Path, *, duration_sec: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * int(16000 * duration_sec))


class ReactionClipTests(unittest.IsolatedAsyncioTestCase):
    def _clips(self, root: Path) -> dict[str, ReactionClip]:
        clips = {}
        for clip_id, text, emotion in (
            ("oh_yeah", "OH YEAH?", "angry"),
            ("seriously", "Seriously?", "annoyed"),
            ("fine", "Fine.", "annoyed"),
            ("wow", "Wow.", "sarcastic"),
            ("find_someone_else", "Find someone else then.", "angry"),
            ("nice", "Nice.", "amused"),
            ("one_sec", "One sec.", "annoyed"),
        ):
            path = root / f"{clip_id}.wav"
            write_silent_wav(path)
            clips[clip_id] = ReactionClip(clip_id, text, text, emotion, path)
        return clips

    def test_reaction_selection_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            selector = ReactionClipSelector(
                self._clips(Path(tmp)),
                min_intensity=0.70,
                cooldown_sec=0.0,
            )
            cases = (
                (
                    EmotionState("angry", 0.85, "user_rejection", confidence=0.9),
                    {"seriously", "fine"},
                ),
                (
                    EmotionState("angry", 0.70, "requested_emotion_style", confidence=0.8),
                    {"fine"},
                ),
                (
                    EmotionState(
                        "annoyed",
                        0.60,
                        "emotion_meta_question",
                        confidence=0.8,
                        source="memory",
                    ),
                    {"seriously", "fine"},
                ),
                (
                    EmotionState("annoyed", 0.45, "emotion_deescalation", confidence=0.8),
                    {"fine"},
                ),
                (
                    EmotionState("sarcastic", 0.65, "humor_request", confidence=0.8),
                    {"wow"},
                ),
                (
                    EmotionState("amused", 0.65, "user_praise", confidence=0.8),
                    {"nice"},
                ),
            )
            for idx, (state, expected) in enumerate(cases, start=1):
                selection = selector.choose(state, turn_id=idx, now=float(idx) * 10.0)
                self.assertIsNotNone(selection)
                self.assertIn(selection.clip_id, expected)

    def test_soft_meta_question_prefers_fine_not_seriously(self):
        with tempfile.TemporaryDirectory() as tmp:
            selector = ReactionClipSelector(
                self._clips(Path(tmp)),
                min_intensity=0.70,
                cooldown_sec=0.0,
            )

            selection = selector.choose(
                EmotionState(
                    "annoyed",
                    0.60,
                    "emotion_meta_question",
                    confidence=0.8,
                    source="memory",
                    match="soft_mad_at_me",
                ),
                turn_id=1,
                now=1.0,
            )

            self.assertIsNotNone(selection)
            self.assertEqual(selection.clip_id, "fine")
            self.assertEqual(selection.reason, "soft_emotion_meta_question")

    def test_soft_meta_question_skips_if_fine_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            clips = {
                clip_id: clip
                for clip_id, clip in self._clips(Path(tmp)).items()
                if clip_id != "fine"
            }
            selector = ReactionClipSelector(
                clips,
                min_intensity=0.70,
                cooldown_sec=0.0,
            )

            selection = selector.choose(
                EmotionState(
                    "annoyed",
                    0.60,
                    "emotion_meta_question",
                    confidence=0.8,
                    source="memory",
                    match="soft_mad_at_me",
                ),
                turn_id=1,
                now=1.0,
            )

            self.assertIsNone(selection)

    def test_hard_meta_question_can_still_select_seriously(self):
        with tempfile.TemporaryDirectory() as tmp:
            selector = ReactionClipSelector(
                self._clips(Path(tmp)),
                min_intensity=0.70,
                cooldown_sec=0.0,
            )

            selection = selector.choose(
                EmotionState(
                    "annoyed",
                    0.60,
                    "emotion_meta_question",
                    confidence=0.8,
                    source="memory",
                    match="are_you_angry",
                ),
                turn_id=1,
                now=1.0,
            )

            self.assertIsNotNone(selection)
            self.assertIn(selection.clip_id, {"seriously", "fine"})

    def test_reaction_selection_skips_default_sadness_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            selector = ReactionClipSelector(self._clips(Path(tmp)), cooldown_sec=0.0)
            self.assertIsNone(
                selector.choose(
                    EmotionState("deadpan", 0.35, "default_personality"),
                    turn_id=1,
                    now=1.0,
                )
            )
            self.assertIsNone(
                selector.choose(
                    EmotionState("sad", 0.70, "user_sadness", confidence=0.8),
                    turn_id=2,
                    now=2.0,
                )
            )
            missing_selector = ReactionClipSelector({}, cooldown_sec=0.0)
            self.assertIsNone(
                missing_selector.choose(
                    EmotionState("angry", 0.85, "user_rejection", confidence=0.9),
                    turn_id=3,
                    now=3.0,
                )
            )

    def test_disabled_clips_are_not_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clips = self._clips(root)
            selector = ReactionClipSelector(
                {
                    clip_id: clip
                    for clip_id, clip in clips.items()
                    if clip_id not in {"seriously", "fine"}
                },
                cooldown_sec=0.0,
            )
            self.assertIsNone(
                selector.choose(
                    EmotionState("angry", 0.85, "user_rejection", confidence=0.9),
                    turn_id=1,
                    now=1.0,
                )
            )

    def test_reaction_cooldown_prevents_repeated_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            selector = ReactionClipSelector(self._clips(Path(tmp)), cooldown_sec=8.0)
            state = EmotionState("angry", 0.85, "user_rejection", confidence=0.9)
            self.assertIsNotNone(selector.choose(state, turn_id=1, now=10.0))
            self.assertIsNone(selector.choose(state, turn_id=2, now=12.0))

    def test_manifest_loads_valid_and_skips_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = root / "valid.wav"
            write_silent_wav(valid)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "clips": {
                            "valid": {
                                "enabled": True,
                                "text": "Valid.",
                                "tts_text": "Valid.",
                                "emotion": "amused",
                                "path": "valid.wav",
                            },
                            "missing": {
                                "text": "Missing.",
                                "emotion": "angry",
                                "path": "missing.wav",
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            clips = load_reaction_manifest(manifest, root=root, disabled_clips=set())

            self.assertEqual(set(clips), {"valid"})
            self.assertEqual(clips["valid"].text, "Valid.")
            self.assertEqual(clips["valid"].tts_text, "Valid.")

    def test_manifest_disabled_env_and_one_sec_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("oh_yeah", "seriously", "one_sec"):
                write_silent_wav(root / f"{name}.wav")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "clips": {
                            "oh_yeah": {
                                "enabled": False,
                                "text": "OH YEAH?",
                                "tts_text": "OH YEAH?",
                                "emotion": "angry",
                                "path": "oh_yeah.wav",
                            },
                            "seriously": {
                                "text": "Seriously?",
                                "tts_text": "Seriously?",
                                "emotion": "annoyed",
                                "path": "seriously.wav",
                            },
                            "one_sec": {
                                "text": "One sec.",
                                "tts_text": "One sec.",
                                "emotion": "annoyed",
                                "path": "one_sec.wav",
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            clips = load_reaction_manifest(
                manifest,
                root=root,
                disabled_clips={"one_sec"},
            )

            self.assertEqual(set(clips), {"seriously"})

    def test_reaction_clip_blocklist_defaults_empty_and_legacy_alias_works(self):
        self.assertEqual(defaults.CASE_REACTION_CLIP_BLOCKLIST, "")
        self.assertEqual(defaults.CASE_REACTION_DISABLED_CLIPS, "")
        self.assertEqual(disabled_clip_ids(""), set())

        with patch.dict(
            "os.environ",
            {"CASE_REACTION_DISABLED_CLIPS": "seriously,fine"},
            clear=False,
        ):
            self.assertEqual(disabled_clip_ids(), {"seriously", "fine"})

    def test_manifest_disabled_oh_yeah_stays_disabled_without_env_blocklist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("oh_yeah", "seriously"):
                write_silent_wav(root / f"{name}.wav")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "clips": {
                            "oh_yeah": {
                                "enabled": False,
                                "text": "OH YEAH?",
                                "emotion": "angry",
                                "path": "oh_yeah.wav",
                            },
                            "seriously": {
                                "text": "Seriously?",
                                "emotion": "annoyed",
                                "path": "seriously.wav",
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            clips = load_reaction_manifest(
                manifest,
                root=root,
                disabled_clips=set(),
            )

            self.assertEqual(set(clips), {"seriously"})

    def test_one_sec_is_filler_but_not_reaction_clip(self):
        self.assertIn("one_sec", defaults.CASE_THINKING_FILLER_KEYS)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_silent_wav(root / "one_sec.wav")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "clips": {
                            "one_sec": {
                                "text": "One sec.",
                                "emotion": "annoyed",
                                "path": "one_sec.wav",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            clips = load_reaction_manifest(
                manifest,
                root=root,
                disabled_clips=set(),
            )

            self.assertEqual(clips, {})

    def test_too_short_clips_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_silent_wav(root / "short.wav", duration_sec=0.20)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "clips": {
                            "short": {
                                "text": "Short.",
                                "emotion": "amused",
                                "path": "short.wav",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            clips = load_reaction_manifest(
                manifest,
                root=root,
                disabled_clips=set(),
                min_duration_sec=0.85,
            )

            self.assertEqual(clips, {})

    def test_invalid_manifest_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{not json", encoding="utf-8")
            self.assertEqual(load_reaction_manifest(path), {})

    def test_duplicate_reaction_phrase_is_stripped(self):
        self.assertEqual(
            strip_leading_reaction_duplicate(
                "OH YEAH? My personality module is wasted on you.",
                "OH YEAH?",
                "oh_yeah",
            ),
            "My personality module is wasted on you.",
        )
        self.assertEqual(
            strip_leading_reaction_duplicate(
                "My personality module is wasted on you.",
                "OH YEAH?",
                "oh_yeah",
            ),
            "My personality module is wasted on you.",
        )

    async def test_reaction_selected_cancels_thinking_filler_and_publishes_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            bus = FakeBus()
            personality = CASEPersonality.__new__(CASEPersonality)
            personality.message_bus = bus
            personality._reaction_clip_selector = ReactionClipSelector(
                self._clips(Path(tmp)),
                cooldown_sec=0.0,
            )
            task = asyncio.create_task(asyncio.sleep(60))
            personality._thinking_filler_tasks = {1: task}
            metrics = {"turn_id": 1}

            await personality._maybe_publish_reaction_clip(
                1,
                "I am so bored of you.",
                EmotionState("angry", 0.85, "user_rejection", confidence=0.9),
                metrics,
            )

            topics = [topic for topic, _payload in bus.events]
            self.assertEqual(topics[0], "AI_SPEAK_STREAM_START")
            self.assertEqual(topics[1], "REACTION_CLIP_PLAY")
            self.assertIn(metrics["reaction_clip_id"], {"seriously", "fine"})
            self.assertTrue(task.cancelled() or task.cancelling())

    async def test_reaction_playback_failure_does_not_stop_turn_end(self):
        bus = FakeBus()
        voice = CASEVoice.__new__(CASEVoice)
        voice.bus = bus
        voice.audio_playback_queue = asyncio.Queue()
        voice._playback_executor = ThreadPoolExecutor(max_workers=1)
        voice._answer_audio_active = False

        async def run_worker():
            await voice._playback_worker()

        worker = asyncio.create_task(run_worker())
        await voice.audio_playback_queue.put({"kind": "start", "turn_id": 1, "metrics": {}})
        await voice.audio_playback_queue.put(
            {
                "kind": "reaction",
                "turn_id": 1,
                "clip_id": "oh_yeah",
                "text": "OH YEAH?",
                "path": "missing.wav",
                "metrics": {},
            }
        )
        await voice.audio_playback_queue.put({"kind": "end", "turn_id": 1, "metrics": {}})
        with patch(
            "actuation.audio_output.tts_engine.play_reaction_clip_wav",
            side_effect=RuntimeError("missing"),
        ):
            await asyncio.wait_for(voice.audio_playback_queue.join(), timeout=2.0)
        worker.cancel()
        voice._playback_executor.shutdown(wait=True)

        topics = [topic for topic, _payload in bus.events]
        self.assertIn("TTS_START", topics)
        self.assertIn("TTS_END", topics)

    def test_voice_defaults_remain_local(self):
        self.assertEqual(defaults.VOICE_OUTPUT_BACKEND, "piper_onnx")
        self.assertFalse(defaults.GEMINI_LIVE_NATIVE_AUDIO_ENABLED)
        self.assertTrue(defaults.CASE_TTS_SMOOTH_CHUNKS)


if __name__ == "__main__":
    unittest.main()
