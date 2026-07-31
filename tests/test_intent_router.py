import unittest
from pathlib import Path

from src.cognition.intent_router import IntentRouter, IntentType


class FakeBus:
    def __init__(self) -> None:
        self.events = []
        self.subscribers = {}

    def subscribe(self, topic, callback) -> None:
        self.subscribers.setdefault(topic, []).append(callback)

    async def publish(self, topic, payload=None) -> None:
        self.events.append((topic, payload))


class FakeScheduler:
    async def run_user_requested_burst(self, **kwargs):
        return {
            "status": "STABLE",
            "target": {"direction": "CENTER", "stable": True},
        }


class FakeVisionEngine:
    async def capture_scene_snapshot(self):
        return Path("output/vision_snapshots/snapshot_test.jpg")


class FakeCapture:
    def __init__(self, ok=False, status="ERROR", faces=None, path=None):
        self.ok = ok
        self.status = status
        self.faces = [] if faces is None else faces
        self.path = path


class IntentClassificationTests(unittest.TestCase):
    def test_expected_phrases(self) -> None:
        cases = {
            "can you see me": IntentType.VISION_SEE_ME,
            "can you see me can you see": IntentType.VISION_SEE_ME,
            "can you actually seem": IntentType.CHAT,
            "can you take a picture of me": IntentType.VISION_TAKE_PICTURE,
            "take my picture": IntentType.VISION_TAKE_PICTURE,
            "look at me": IntentType.VISION_SEE_ME,
            "look around": IntentType.VISION_SEE_ME,
            "camera check": IntentType.VISION_SEE_ME,
            "vision check": IntentType.VISION_SEE_ME,
            "what is your name": IntentType.CHAT,
            "can you roast me": IntentType.CHAT,
            "tell me a joke": IntentType.CHAT,
        }
        for transcript, expected in cases.items():
            with self.subTest(transcript=transcript):
                self.assertEqual(IntentRouter.classify(transcript).type, expected)


class IntentRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.bus = FakeBus()
        self.vision_calls = []

        async def fake_vision_once(reason, mode="single_frame", **kwargs):
            self.vision_calls.append((reason, mode, kwargs))
            if mode == "snapshot":
                return FakeCapture(
                    ok=True,
                    status="OK",
                    path=Path("output/vision_snapshots/snapshot_test.jpg"),
                )
            return FakeCapture(
                ok=True,
                status="OK",
                faces=[{"direction": "CENTER", "stable": True}],
            )

        self.router = IntentRouter(
            self.bus,
            vision_scheduler=FakeScheduler(),
            vision_engine=FakeVisionEngine(),
            vision_once=fake_vision_once,
        )

    async def test_see_me_bypasses_chat(self) -> None:
        await self.router.handle_transcript("can you see me can you see")
        self.assertEqual(
            self.bus.events,
            [("AI_SPEAK", "Yeah. You're centered.")],
        )
        self.assertEqual(self.vision_calls[0][0], IntentType.VISION_SEE_ME)
        self.assertTrue(self.vision_calls[0][2]["allow_during_thinking"])

    async def test_take_picture_bypasses_chat(self) -> None:
        await self.router.handle_transcript("can you take a picture of me")
        self.assertEqual(
            self.bus.events,
            [("AI_SPEAK", "Done. I saved the snapshot.")],
        )
        self.assertEqual(self.vision_calls[0][0], IntentType.VISION_TAKE_PICTURE)
        self.assertTrue(self.vision_calls[0][2]["allow_during_thinking"])

    async def test_chat_is_forwarded(self) -> None:
        await self.router.handle_transcript("what is your name")
        self.assertEqual(
            self.bus.events,
            [("CHAT_USER_SPOKE", "what is your name")],
        )

    async def test_see_me_uses_on_demand_vision_bypass_without_scheduler(self) -> None:
        calls = []

        async def fake_vision_once(reason, mode="single_frame", **kwargs):
            calls.append((reason, mode, kwargs))
            return FakeCapture()

        router = IntentRouter(self.bus, vision_once=fake_vision_once)
        await router.handle_transcript("Can you see me?")
        self.assertEqual(calls[0][0], IntentType.VISION_SEE_ME)
        self.assertTrue(calls[0][2]["allow_during_thinking"])
        self.assertEqual(
            self.bus.events[-1],
            ("AI_SPEAK", "I tried, but the camera did not give me a clean frame."),
        )


if __name__ == "__main__":
    unittest.main()
