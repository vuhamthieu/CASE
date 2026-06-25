import asyncio
import time
from pathlib import Path

from src.realtime.realtime_tools import RealtimeToolRouter
from src.realtime import realtime_config


class Scheduler:
    latest_target = {"direction": "LEFT", "stable": True, "timestamp": time.time()}

    async def run_user_requested_burst(self, **kwargs):
        return {"status": "STABLE", "target": self.latest_target}


class Vision:
    async def capture_scene_snapshot(self, output_dir):
        return Path(output_dir) / "snapshot.jpg"


def test_tools_return_structured_results_and_motion_is_disabled():
    router = RealtimeToolRouter(vision_scheduler=Scheduler(), vision_engine=Vision())

    async def exercise():
        seen = await router.execute("case_vision_see_me", {})
        state = await router.execute("case_get_vision_state", {})
        motion = await router.execute("case_motion_request", {"command": "forward"})
        return seen, state, motion

    seen, state, motion = asyncio.run(exercise())
    assert seen["face_detected"] is True
    assert state["has_recent_target"] is True
    assert motion == {
        "ok": True,
        "accepted": False,
        "command": "FORWARD",
        "reason": "motor control disabled in realtime v1",
    }


def test_snapshot_rejects_paths_outside_output():
    router = RealtimeToolRouter(vision_engine=Vision())
    result = asyncio.run(
        router.execute("case_take_picture", {"save_dir": "/tmp/not-case-output"})
    )
    assert result["ok"] is False


def test_realtime_tools_are_disabled_by_default_for_plain_chat():
    assert realtime_config.REALTIME_ENABLE_TOOLS is False
