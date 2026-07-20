"""Safe local function tools exposed to Gemini Live."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Optional

from .realtime_config import (
    REALTIME_TOOL_TIMEOUT_SEC,
    SNAPSHOT_TOOL_TIMEOUT_SEC,
    VISION_TOOL_TIMEOUT_SEC,
)
from src.vision.vision_engine import run_vision_once
from src.memory.core_memory import case_memory


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = (PROJECT_ROOT / "output").resolve()


def update_core_memory(key: str, value: str) -> str:
    """Updates the core memory with a key-value pair.

    Use this tool to save critical user facts, preferences, names, and other enduring information
    about the user or environment that should be remembered across sessions.

    Args:
        key: The memory key or category (e.g., 'user_name', 'favorite_color', 'user_birthday').
        value: The memory value or detail to be stored (e.g., 'Alice', 'blue', 'October 5th').

    Returns:
        A confirmation message indicating success or failure.
    """
    return case_memory.update_memory(key, value)


TOOL_DECLARATIONS = [
    {
        "name": "case_vision_see_me",
        "description": (
            "Use CASE's local camera for a short forced face-tracking burst. "
            "Call this before claiming that CASE can see or locate the user."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "duration_sec": {"type": "NUMBER"},
                "fps": {"type": "NUMBER"},
                "wait_for_stable": {"type": "BOOLEAN"},
            },
        },
    },
    {
        "name": "case_vision_capture",
        "description": "Capture and save one on-demand image using CASE's local Pi camera.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "reason": {"type": "STRING"},
                "save_dir": {"type": "STRING"},
            },
        },
    },
    {
        "name": "case_take_picture",
        "description": "Capture and save a real image using CASE's local Pi camera.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "reason": {"type": "STRING"},
                "save_dir": {"type": "STRING"},
            },
        },
    },
    {
        "name": "case_get_vision_state",
        "description": "Return the most recent cached face-tracking state.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "case_motion_request",
        "description": (
            "Log a requested high-level motion intent. Motor execution is disabled."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {"command": {"type": "STRING"}},
            "required": ["command"],
        },
    },
    {
        "name": "update_core_memory",
        "description": (
            "Updates CASE's core memory with a key-value pair. Use this to save critical "
            "user facts, preferences, names, and other enduring information about the user "
            "or environment that should be remembered across sessions."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "key": {
                    "type": "STRING",
                    "description": "The memory key or category (e.g., 'user_name', 'favorite_color').",
                },
                "value": {
                    "type": "STRING",
                    "description": "The memory value or detail to be stored (e.g., 'Alice', 'blue').",
                },
            },
            "required": ["key", "value"],
        },
    },
]


class RealtimeToolRouter:
    def __init__(
        self,
        *,
        vision_scheduler: Optional[Any] = None,
        vision_engine: Optional[Any] = None,
        vision_once: Optional[Any] = None,
    ) -> None:
        self.vision_scheduler = vision_scheduler
        self.vision_engine = vision_engine
        self.vision_once = vision_once or run_vision_once

    @property
    def declarations(self) -> list[dict[str, Any]]:
        return TOOL_DECLARATIONS

    async def execute(self, name: str, arguments: Optional[dict]) -> dict[str, Any]:
        args = dict(arguments or {})
        logger.info("REALTIME: tool_call name=%s arguments=%s", name, args)
        handlers = {
            "case_vision_see_me": (
                self._vision_see_me,
                VISION_TOOL_TIMEOUT_SEC,
            ),
            "case_take_picture": (
                self._take_picture,
                SNAPSHOT_TOOL_TIMEOUT_SEC,
            ),
            "case_vision_capture": (
                self._take_picture,
                SNAPSHOT_TOOL_TIMEOUT_SEC,
            ),
            "case_get_vision_state": (
                self._get_vision_state,
                REALTIME_TOOL_TIMEOUT_SEC,
            ),
            "case_motion_request": (
                self._motion_request,
                REALTIME_TOOL_TIMEOUT_SEC,
            ),
            "update_core_memory": (
                self._update_core_memory,
                REALTIME_TOOL_TIMEOUT_SEC,
            ),
        }
        if name not in handlers:
            return {"ok": False, "error": f"unknown tool: {name}"}

        handler, timeout = handlers[name]
        try:
            result = await asyncio.wait_for(handler(args), timeout=timeout)
        except asyncio.TimeoutError:
            result = {"ok": False, "error": f"tool timed out after {timeout:.1f}s"}
        except Exception as exc:
            logger.exception("REALTIME: tool failed name=%s", name)
            result = {"ok": False, "error": str(exc)}

        logger.info("REALTIME: tool_result name=%s ok=%s", name, result.get("ok"))
        return result

    async def _vision_see_me(self, args: dict) -> dict[str, Any]:
        if self.vision_scheduler is not None:
            duration = max(0.5, min(float(args.get("duration_sec", 4.0)), 6.0))
            fps = max(0.2, min(float(args.get("fps", 1.0)), 2.0))
            wait_for_stable = bool(args.get("wait_for_stable", True))
            result = await self.vision_scheduler.run_user_requested_burst(
                duration_sec=duration,
                fps=fps,
                wait_for_stable=wait_for_stable,
                timeout_sec=min(VISION_TOOL_TIMEOUT_SEC, duration + 1.0),
            )
            target = result.get("target") or {}
            status = result.get("status")
            error = str(result.get("error", "Camera error."))
        else:
            capture = await self.vision_once("tool_case_vision_see_me")
            target = capture.faces[0] if capture.faces else {}
            status = "TARGET" if capture.ok and target else capture.status
            error = capture.error or "Camera error."
        if status in {"STABLE", "TARGET"} and target:
            direction = target.get("direction", "UNKNOWN")
            return {
                "ok": True,
                "face_detected": True,
                "direction": direction,
                "stable": bool(target.get("stable", False)),
                "bbox": target.get("bbox"),
                "message": (
                    "The user is centered."
                    if direction == "CENTER"
                    else f"The user is to the {direction.lower()}."
                ),
            }
        if status == "ERROR":
            return {
                "ok": False,
                "face_detected": False,
                "message": error,
            }
        return {
            "ok": True,
            "face_detected": False,
            "message": "No clean visual lock.",
        }

    async def _take_picture(self, args: dict) -> dict[str, Any]:
        save_dir = str(args.get("save_dir", "output/vision_snapshots"))
        destination = Path(save_dir).expanduser()
        if not destination.is_absolute():
            destination = PROJECT_ROOT / destination
        destination = destination.resolve()
        if destination != OUTPUT_ROOT and OUTPUT_ROOT not in destination.parents:
            return {"ok": False, "error": "save_dir must be under CASE/output"}

        logger.info(
            "REALTIME: snapshot requested reason=%s directory=%s",
            args.get("reason", "user_request"),
            destination,
        )
        if self.vision_engine is not None:
            path = await self.vision_engine.capture_scene_snapshot(destination)
        else:
            capture = await self.vision_once(
                str(args.get("reason", "tool_case_take_picture")),
                mode="snapshot",
                output_dir=destination,
            )
            path = capture.path if capture.ok else None
        if path is None:
            return {"ok": False, "error": "Camera capture failed."}
        try:
            display_path = str(Path(path).resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            display_path = str(path)
        return {"ok": True, "path": display_path}

    async def _get_vision_state(self, args: dict) -> dict[str, Any]:
        target = None
        if self.vision_scheduler is not None:
            target = self.vision_scheduler.latest_target
        if not target:
            return {
                "ok": True,
                "has_recent_target": False,
                "message": "No recent vision target.",
            }

        timestamp = target.get("timestamp")
        age_sec = max(0.0, time.time() - timestamp) if timestamp else None
        has_recent = age_sec is not None and age_sec <= 10.0
        return {
            "ok": True,
            "has_recent_target": has_recent,
            "direction": target.get("direction"),
            "stable": bool(target.get("stable", False)),
            "age_sec": round(age_sec, 2) if age_sec is not None else None,
        }

    async def _motion_request(self, args: dict) -> dict[str, Any]:
        command = str(args.get("command", "")).strip().upper()
        logger.info("REALTIME: motion requested but disabled command=%s", command)
        return {
            "ok": True,
            "accepted": False,
            "command": command,
            "reason": "motor control disabled in realtime v1",
        }

    async def _update_core_memory(self, args: dict) -> dict[str, Any]:
        key = str(args.get("key", "")).strip()
        value = str(args.get("value", "")).strip()
        if not key or not value:
            return {"ok": False, "error": "key and value are required"}
        try:
            result = await asyncio.to_thread(update_core_memory, key, value)
            return {"ok": True, "message": result}
        except Exception as exc:
            logger.exception("REALTIME: update_core_memory failed")
            return {"ok": False, "error": str(exc)}
