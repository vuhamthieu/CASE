import unittest

from src.config import defaults
from src.realtime.realtime_tools import RealtimeToolRouter
from src.vision import vision_engine, vision_scheduler


class VisionOnDemandDefaultsTests(unittest.TestCase):
    def test_camera_is_hard_off_by_default(self) -> None:
        self.assertEqual(defaults.VISION_MODE, "on_demand")
        self.assertTrue(defaults.VISION_ON_DEMAND_ONLY)
        self.assertFalse(defaults.VISION_OPEN_CAMERA_ON_BOOT)
        self.assertFalse(defaults.VISION_BACKGROUND_TASK_ENABLED)
        self.assertFalse(defaults.VISION_IDLE_GLANCE_ENABLED)
        self.assertFalse(defaults.VISION_SOCIAL_TRACKING_ENABLED)
        self.assertFalse(defaults.VISION_IDLE_USER_GREETING_ENABLED)
        self.assertFalse(defaults.VISION_AUTO_FACE_TRACKING_ENABLED)

    def test_runtime_modules_inherit_hard_off_defaults(self) -> None:
        self.assertEqual(vision_engine.VISION_MODE, "on_demand")
        self.assertTrue(vision_engine.VISION_ON_DEMAND_ONLY)
        self.assertFalse(vision_engine.VISION_OPEN_CAMERA_ON_BOOT)
        self.assertFalse(vision_engine.VISION_BACKGROUND_TASK_ENABLED)
        self.assertFalse(vision_scheduler.VISION_SCHEDULER_ENABLED)
        self.assertFalse(vision_scheduler.VISION_STARTUP_ENABLED)
        self.assertTrue(vision_scheduler.VISION_ON_DEMAND_ONLY)
        self.assertFalse(vision_scheduler.VISION_IDLE_GLANCE_ENABLED)
        self.assertFalse(vision_scheduler.VISION_SOCIAL_TRACKING_ENABLED)

    def test_tool_router_accepts_on_demand_vision_without_scheduler(self) -> None:
        router = RealtimeToolRouter()
        self.assertIsNone(router.vision_scheduler)
        self.assertIsNone(router.vision_engine)
        self.assertIsNotNone(router.vision_once)

    def test_on_demand_vision_still_blocks_thinking_without_explicit_bypass(self) -> None:
        result = __import__("asyncio").run(
            vision_engine.run_vision_once(
                "background_check",
                case_state_provider=lambda: "THINKING",
            )
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "BLOCKED")


if __name__ == "__main__":
    unittest.main()
