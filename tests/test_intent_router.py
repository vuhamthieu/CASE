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


class IntentClassificationTests(unittest.TestCase):
    def test_expected_phrases(self) -> None:
        cases = {
            "can you see me": IntentType.VISION_SEE_ME,
            "can you see me can you see": IntentType.VISION_SEE_ME,
            "can you actually seem": IntentType.CHAT,
            "can you take a picture of me": IntentType.VISION_TAKE_PICTURE,
            "take my picture": IntentType.VISION_TAKE_PICTURE,
            "what is your name": IntentType.CHAT,
        }
        for transcript, expected in cases.items():
            with self.subTest(transcript=transcript):
                self.assertEqual(IntentRouter.classify(transcript).type, expected)


class IntentRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.bus = FakeBus()
        self.router = IntentRouter(
            self.bus,
            vision_scheduler=FakeScheduler(),
            vision_engine=FakeVisionEngine(),
        )

    async def test_see_me_bypasses_chat(self) -> None:
        await self.router.handle_transcript("can you see me can you see")
        self.assertEqual(
            self.bus.events,
            [("AI_SPEAK", "Yes, boss. You're centered.")],
        )

    async def test_take_picture_bypasses_chat(self) -> None:
        await self.router.handle_transcript("can you take a picture of me")
        self.assertEqual(
            self.bus.events,
            [("AI_SPEAK", "Done, boss. I saved the snapshot.")],
        )

    async def test_chat_is_forwarded(self) -> None:
        await self.router.handle_transcript("what is your name")
        self.assertEqual(
            self.bus.events,
            [("CHAT_USER_SPOKE", "what is your name")],
        )


if __name__ == "__main__":
    unittest.main()
