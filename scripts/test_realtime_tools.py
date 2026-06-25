#!/usr/bin/env python3
"""Exercise realtime tool contracts with deterministic local test doubles."""

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.realtime.realtime_tools import RealtimeToolRouter


class FakeScheduler:
    latest_target = {
        "direction": "CENTER",
        "stable": True,
        "bbox": [10, 20, 100, 120],
        "timestamp": time.time(),
    }

    async def run_user_requested_burst(self, **kwargs):
        return {"status": "STABLE", "target": dict(self.latest_target)}


class FakeVisionEngine:
    async def capture_scene_snapshot(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "realtime_tool_test.jpg"
        path.write_bytes(b"test snapshot placeholder")
        return path


async def run() -> None:
    router = RealtimeToolRouter(
        vision_scheduler=FakeScheduler(),
        vision_engine=FakeVisionEngine(),
    )
    cases = [
        ("case_vision_see_me", {}),
        ("case_get_vision_state", {}),
        ("case_take_picture", {"save_dir": "output/realtime_tool_test"}),
        ("case_motion_request", {"command": "TURN_LEFT"}),
        ("unknown_tool", {}),
    ]
    for name, arguments in cases:
        result = await router.execute(name, arguments)
        print(name, json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(run())
