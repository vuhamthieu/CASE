"""Attention-gated burst scheduler for CASE's low-priority Pi vision."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Callable, Optional

from src.config import defaults
from .vision_engine import VisionEngine


logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


VISION_SCHEDULER_ENABLED = _env_bool(
    "VISION_SCHEDULER_ENABLED", defaults.VISION_SCHEDULER_ENABLED
)
VISION_STARTUP_ENABLED = _env_bool(
    "VISION_STARTUP_ENABLED", defaults.VISION_STARTUP_ENABLED
)
VISION_ON_DEMAND_ONLY = _env_bool(
    "VISION_ON_DEMAND_ONLY", defaults.VISION_ON_DEMAND_ONLY
)
VISION_IDLE_GLANCE_ENABLED = _env_bool(
    "VISION_IDLE_GLANCE_ENABLED", defaults.VISION_IDLE_GLANCE_ENABLED
)
VISION_SOCIAL_TRACKING_ENABLED = _env_bool(
    "VISION_SOCIAL_TRACKING_ENABLED", defaults.VISION_SOCIAL_TRACKING_ENABLED
)
VISION_PAUSE_DURING_LISTENING = _env_bool(
    "VISION_PAUSE_DURING_LISTENING", True
)
VISION_PAUSE_DURING_THINKING = _env_bool("VISION_PAUSE_DURING_THINKING", True)
VISION_PAUSE_DURING_SPEAKING = _env_bool("VISION_PAUSE_DURING_SPEAKING", True)
VISION_ALLOW_WHEN_IDLE = _env_bool("VISION_ALLOW_WHEN_IDLE", False)

VISION_IDLE_GLANCE_INTERVAL_SEC = float(
    os.getenv("VISION_IDLE_GLANCE_INTERVAL_SEC", "45.0")
)
VISION_IDLE_GLANCE_DURATION_SEC = float(
    os.getenv("VISION_IDLE_GLANCE_DURATION_SEC", "4.0")
)
VISION_IDLE_GLANCE_FPS = float(os.getenv("VISION_IDLE_GLANCE_FPS", "0.5"))
VISION_BOREDOM_SCAN_DURATION_SEC = float(
    os.getenv("VISION_BOREDOM_SCAN_DURATION_SEC", "6.0")
)
VISION_BOREDOM_SCAN_FPS = float(os.getenv("VISION_BOREDOM_SCAN_FPS", "1.0"))
VISION_SOCIAL_TRACKING_DURATION_SEC = float(
    os.getenv("VISION_SOCIAL_TRACKING_DURATION_SEC", "20.0")
)
VISION_SOCIAL_TRACKING_FPS = float(
    os.getenv("VISION_SOCIAL_TRACKING_FPS", "0.5")
)
VISION_USER_REQUESTED_DURATION_SEC = float(
    os.getenv("VISION_USER_REQUESTED_DURATION_SEC", "8.0")
)
VISION_USER_REQUESTED_FPS = float(os.getenv("VISION_USER_REQUESTED_FPS", "1.0"))
VISION_MOVE_PRECHECK_DURATION_SEC = float(
    os.getenv("VISION_MOVE_PRECHECK_DURATION_SEC", "4.0")
)
VISION_MOVE_PRECHECK_FPS = float(os.getenv("VISION_MOVE_PRECHECK_FPS", "1.0"))

BOREDOM_ENABLED = _env_bool("BOREDOM_ENABLED", False)
BOREDOM_SCORE_MIN = float(os.getenv("BOREDOM_SCORE_MIN", "0.0"))
BOREDOM_SCORE_MAX = float(os.getenv("BOREDOM_SCORE_MAX", "100.0"))
BOREDOM_TRIGGER_THRESHOLD = float(os.getenv("BOREDOM_TRIGGER_THRESHOLD", "60.0"))
BOREDOM_IDLE_GAIN_PER_SEC = float(os.getenv("BOREDOM_IDLE_GAIN_PER_SEC", "0.02"))
BOREDOM_NO_FACE_BONUS = float(os.getenv("BOREDOM_NO_FACE_BONUS", "5.0"))
BOREDOM_AFTER_SCAN_RESET = float(os.getenv("BOREDOM_AFTER_SCAN_RESET", "20.0"))
BOREDOM_USER_INTERACTION_PENALTY = float(
    os.getenv("BOREDOM_USER_INTERACTION_PENALTY", "30.0")
)
BOREDOM_FACE_SEEN_PENALTY = float(
    os.getenv("BOREDOM_FACE_SEEN_PENALTY", "20.0")
)
BOREDOM_SPEAKING_PAUSE = _env_bool("BOREDOM_SPEAKING_PAUSE", True)


class VisionMode:
    OFF = "OFF"
    IDLE_GLANCE = "IDLE_GLANCE"
    BOREDOM_SCAN = "BOREDOM_SCAN"
    SOCIAL_TRACKING = "SOCIAL_TRACKING"
    MOVE_PRECHECK = "MOVE_PRECHECK"
    USER_REQUESTED = "USER_REQUESTED"


class VisionScheduler:
    """Open the VisionEngine capture gate only for attention-worthy bursts."""

    def __init__(
        self,
        message_bus: Any,
        vision_engine: VisionEngine,
        case_state_provider: Callable[[], str],
        *,
        enabled: bool = VISION_SCHEDULER_ENABLED,
        startup_enabled: bool = VISION_STARTUP_ENABLED,
        idle_glance_interval_sec: float = VISION_IDLE_GLANCE_INTERVAL_SEC,
        idle_glance_duration_sec: float = VISION_IDLE_GLANCE_DURATION_SEC,
        idle_glance_fps: float = VISION_IDLE_GLANCE_FPS,
        boredom_scan_duration_sec: float = VISION_BOREDOM_SCAN_DURATION_SEC,
        boredom_scan_fps: float = VISION_BOREDOM_SCAN_FPS,
        social_tracking_duration_sec: float = VISION_SOCIAL_TRACKING_DURATION_SEC,
        social_tracking_fps: float = VISION_SOCIAL_TRACKING_FPS,
        user_requested_duration_sec: float = VISION_USER_REQUESTED_DURATION_SEC,
        user_requested_fps: float = VISION_USER_REQUESTED_FPS,
        poll_interval_sec: float = 0.2,
        initial_idle_glance_due: bool = False,
        initial_boredom_score: float = BOREDOM_SCORE_MIN,
    ) -> None:
        timing_values = (
            idle_glance_interval_sec,
            idle_glance_duration_sec,
            boredom_scan_duration_sec,
            social_tracking_duration_sec,
            user_requested_duration_sec,
            poll_interval_sec,
        )
        if any(value <= 0 for value in timing_values):
            raise ValueError("Vision scheduler durations and intervals must be positive")
        if min(
            idle_glance_fps,
            boredom_scan_fps,
            social_tracking_fps,
            user_requested_fps,
        ) <= 0:
            raise ValueError("Vision scheduler FPS values must be positive")

        self.message_bus = message_bus
        self.vision_engine = vision_engine
        self.case_state_provider = case_state_provider
        self.enabled = enabled
        self.startup_enabled = startup_enabled
        self.idle_glance_interval_sec = idle_glance_interval_sec
        self.idle_glance_duration_sec = idle_glance_duration_sec
        self.idle_glance_fps = idle_glance_fps
        self.boredom_scan_duration_sec = boredom_scan_duration_sec
        self.boredom_scan_fps = boredom_scan_fps
        self.social_tracking_duration_sec = social_tracking_duration_sec
        self.social_tracking_fps = social_tracking_fps
        self.user_requested_duration_sec = user_requested_duration_sec
        self.user_requested_fps = user_requested_fps
        self.poll_interval_sec = poll_interval_sec

        now = time.monotonic()
        self.mode = VisionMode.OFF
        self.boredom_score = self._clamp_boredom(initial_boredom_score)
        self.latest_target: Optional[dict[str, Any]] = None
        self._burst_remaining_sec = 0.0
        self._burst_fps: Optional[float] = None
        self._face_seen_during_burst = False
        self._pending_user_requested = False
        self._pending_move_precheck = False
        self._pending_social_tracking = False
        self._paused_case_state: Optional[str] = None
        self._stop_event = asyncio.Event()
        self._forced_burst_lock = asyncio.Lock()
        self._forced_burst_active = False
        self._forced_stable_event = asyncio.Event()
        self._forced_error_event = asyncio.Event()
        self._forced_error: Optional[str] = None
        self._last_tick_at = now
        self._last_idle_glance_at = (
            now - idle_glance_interval_sec if initial_idle_glance_due else now
        )
        self._camera_off_logged = False

        message_bus.subscribe("USER_SPOKE", self._on_user_spoke)
        message_bus.subscribe("VISION_USER_REQUESTED", self._on_user_requested)
        message_bus.subscribe("VISION_USER_DETECTED", self._on_user_detected)
        message_bus.subscribe("VISION_USER_LOST", self._on_user_lost)
        message_bus.subscribe("VISION_TARGET_UPDATE", self._on_target_update)
        message_bus.subscribe("VISION_ERROR", self._on_vision_error)

    async def run(self) -> None:
        self.vision_engine.set_scheduler_gate(False)
        self._log_camera_off()
        if VISION_ON_DEMAND_ONLY:
            logger.info("VISION_SCHEDULER: disabled because VISION_ON_DEMAND_ONLY=true")
            return
        if not self.enabled or not self.startup_enabled:
            logger.info("VISION_SCHEDULER: disabled by configuration")
            return

        self._last_tick_at = time.monotonic()
        try:
            while not self._stop_event.is_set():
                now = time.monotonic()
                elapsed = max(0.0, now - self._last_tick_at)
                self._last_tick_at = now
                if self._forced_burst_active:
                    await self._wait_or_stop(self.poll_interval_sec)
                    continue
                case_state = self._current_case_state()

                if self._should_pause(case_state):
                    self._pause_for_state(case_state)
                    await self._wait_or_stop(self.poll_interval_sec)
                    continue

                self._resume_for_state(case_state)
                if self.mode == VisionMode.OFF:
                    self._increase_boredom(elapsed, case_state)
                    self._start_next_burst_if_due(now)
                else:
                    self._burst_remaining_sec -= elapsed
                    if (
                        self._pending_user_requested
                        and self.mode != VisionMode.USER_REQUESTED
                    ):
                        self._finish_burst(interrupted=True)
                        self._start_next_burst_if_due(now)
                    elif self._burst_remaining_sec <= 0:
                        self._finish_burst()

                await self._wait_or_stop(self.poll_interval_sec)
        finally:
            self.vision_engine.set_scheduler_gate(False)
            self.mode = VisionMode.OFF
            self._log_camera_off(force=True)

    def stop(self) -> None:
        self._stop_event.set()
        self.vision_engine.set_scheduler_gate(False)

    def request_mode(self, mode: str) -> None:
        if mode == VisionMode.USER_REQUESTED:
            self._pending_user_requested = True
        elif mode == VisionMode.MOVE_PRECHECK:
            self._pending_move_precheck = True
        elif mode == VisionMode.SOCIAL_TRACKING:
            self._pending_social_tracking = True
        else:
            raise ValueError(f"Vision mode cannot be requested directly: {mode}")

    async def run_user_requested_burst(
        self,
        duration_sec: float = 4.0,
        fps: float = 1.0,
        wait_for_stable: bool = True,
        timeout_sec: float = 5.0,
    ) -> dict[str, Any]:
        """Force one quiet local-command burst and return a fresh target result."""
        if min(duration_sec, fps, timeout_sec) <= 0:
            raise ValueError(
                "Forced vision duration, FPS, and timeout must be positive"
            )

        async with self._forced_burst_lock:
            if self.vision_engine.camera_available is False:
                return {"status": "ERROR", "error": "camera unavailable"}
            if self.mode != VisionMode.OFF:
                self._finish_burst(interrupted=True)
            self._pending_user_requested = False
            self._forced_burst_active = True
            self._forced_stable_event.clear()
            self._forced_error_event.clear()
            self._forced_error = None
            self.latest_target = None
            self.mode = VisionMode.USER_REQUESTED
            self.vision_engine.set_scheduler_gate(True, fps, force=True)
            logger.info(
                "VISION_SCHEDULER: starting forced USER_REQUESTED burst "
                "duration=%.1f fps=%.1f",
                duration_sec,
                fps,
            )

            deadline = time.monotonic() + min(duration_sec, timeout_sec)
            try:
                while not self._stop_event.is_set() and time.monotonic() < deadline:
                    if self._forced_error_event.is_set():
                        return {
                            "status": "ERROR",
                            "error": self._forced_error or "camera error",
                        }
                    if wait_for_stable and self._forced_stable_event.is_set():
                        target = dict(self.latest_target or {})
                        logger.info(
                            "VISION: stable target result direction=%s",
                            target.get("direction", "UNKNOWN"),
                        )
                        return {"status": "STABLE", "target": target}
                    if not wait_for_stable and self.latest_target is not None:
                        return {
                            "status": "TARGET",
                            "target": dict(self.latest_target),
                        }
                    await self._wait_or_stop(0.05)
                return {"status": "NOFACE", "target": None}
            finally:
                self.vision_engine.set_scheduler_gate(False)
                self.mode = VisionMode.OFF
                self._forced_burst_active = False
                self._pending_user_requested = False
                self._log_camera_off(force=True)

    async def _on_user_spoke(self, payload: Any) -> None:
        self._adjust_boredom(-BOREDOM_USER_INTERACTION_PENALTY)

    async def _on_user_requested(self, payload: Any) -> None:
        self._pending_user_requested = True

    async def _on_user_detected(self, payload: Any) -> None:
        self._adjust_boredom(-BOREDOM_FACE_SEEN_PENALTY)
        if self.mode in {VisionMode.IDLE_GLANCE, VisionMode.BOREDOM_SCAN}:
            self._face_seen_during_burst = True

    async def _on_user_lost(self, payload: Any) -> None:
        self.latest_target = None

    async def _on_target_update(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        self.latest_target = dict(payload)
        if self._forced_burst_active and payload.get("stable"):
            self._forced_stable_event.set()
        if payload.get("stable") and self.mode in {
            VisionMode.IDLE_GLANCE,
            VisionMode.BOREDOM_SCAN,
        }:
            self._face_seen_during_burst = True

    async def _on_vision_error(self, payload: Any) -> None:
        if not self._forced_burst_active:
            return
        if isinstance(payload, dict):
            self._forced_error = str(payload.get("error", "camera error"))
        else:
            self._forced_error = str(payload or "camera error")
        self._forced_error_event.set()

    def _current_case_state(self) -> str:
        try:
            return str(self.case_state_provider() or "UNKNOWN").upper()
        except Exception as exc:
            logger.warning("VISION_SCHEDULER: could not read CASE state: %s", exc)
            return "UNKNOWN"

    @staticmethod
    def _state_label(state: str) -> str:
        return "LISTENING" if state == "LISTEN_COMMAND" else state

    def _should_pause(self, state: str) -> bool:
        idle_states = {"IDLE", "WAKEWORD_MODE"}
        if state in idle_states:
            return not VISION_ALLOW_WHEN_IDLE
        listening_states = {
            "LISTENING",
            "LISTEN_COMMAND",
            "SHORT_FOLLOW_UP",
            "LONG_CONVERSATION",
        }
        if state in listening_states:
            return VISION_PAUSE_DURING_LISTENING
        if state == "THINKING":
            return VISION_PAUSE_DURING_THINKING
        if state in {"SPEAKING", "WAKE_ACK"}:
            return VISION_PAUSE_DURING_SPEAKING
        # Unknown states fail closed so a new voice state cannot accidentally
        # enable camera work in a latency-sensitive period.
        return True

    def _pause_for_state(self, state: str) -> None:
        self.vision_engine.set_scheduler_gate(False)
        if self._paused_case_state != state:
            logger.info(
                "VISION_SCHEDULER: paused because CASE is %s",
                self._state_label(state),
            )
            self._paused_case_state = state

    def _resume_for_state(self, state: str) -> None:
        if self._paused_case_state is not None:
            logger.info(
                "VISION_SCHEDULER: resumed because CASE is %s",
                self._state_label(state),
            )
            self._paused_case_state = None
        if self.mode != VisionMode.OFF and self._burst_fps is not None:
            self.vision_engine.set_scheduler_gate(True, self._burst_fps)

    def _increase_boredom(self, elapsed: float, state: str) -> None:
        if not BOREDOM_ENABLED:
            return
        if state in {"IDLE", "WAKEWORD_MODE"}:
            self._adjust_boredom(BOREDOM_IDLE_GAIN_PER_SEC * elapsed)
        elif not BOREDOM_SPEAKING_PAUSE and state == "SPEAKING":
            self._adjust_boredom(BOREDOM_IDLE_GAIN_PER_SEC * elapsed)

    def _start_next_burst_if_due(self, now: float) -> None:
        if self._pending_user_requested:
            self._pending_user_requested = False
            self._start_burst(
                VisionMode.USER_REQUESTED,
                self.user_requested_duration_sec,
                self.user_requested_fps,
            )
            return
        if self._pending_move_precheck:
            self._pending_move_precheck = False
            self._start_burst(
                VisionMode.MOVE_PRECHECK,
                VISION_MOVE_PRECHECK_DURATION_SEC,
                VISION_MOVE_PRECHECK_FPS,
            )
            return
        if self._pending_social_tracking:
            self._pending_social_tracking = False
            if not VISION_SOCIAL_TRACKING_ENABLED:
                logger.info("VISION_SCHEDULER: social tracking disabled")
                return
            self._start_burst(
                VisionMode.SOCIAL_TRACKING,
                self.social_tracking_duration_sec,
                self.social_tracking_fps,
            )
            logger.info("VISION_SCHEDULER: social tracking started")
            return
        if BOREDOM_ENABLED and self.boredom_score >= BOREDOM_TRIGGER_THRESHOLD:
            logger.info(
                "VISION_SCHEDULER: boredom=%.1f trigger BOREDOM_SCAN",
                self.boredom_score,
            )
            self._start_burst(
                VisionMode.BOREDOM_SCAN,
                self.boredom_scan_duration_sec,
                self.boredom_scan_fps,
            )
            return
        if (
            VISION_IDLE_GLANCE_ENABLED
            and now - self._last_idle_glance_at >= self.idle_glance_interval_sec
        ):
            self._start_burst(
                VisionMode.IDLE_GLANCE,
                self.idle_glance_duration_sec,
                self.idle_glance_fps,
            )

    def _start_burst(self, mode: str, duration: float, fps: float) -> None:
        self.mode = mode
        self._burst_remaining_sec = duration
        self._burst_fps = fps
        self._face_seen_during_burst = False
        self._camera_off_logged = False
        self.vision_engine.set_scheduler_gate(True, fps)
        logger.info(
            "VISION_SCHEDULER: starting %s duration=%.1f fps=%.1f",
            mode,
            duration,
            fps,
        )

    def _finish_burst(self, interrupted: bool = False) -> None:
        finished_mode = self.mode
        face_seen = self._face_seen_during_burst
        self.vision_engine.set_scheduler_gate(False)
        self.mode = VisionMode.OFF
        self._burst_remaining_sec = 0.0
        self._burst_fps = None
        self._face_seen_during_burst = False

        if finished_mode == VisionMode.BOREDOM_SCAN and not interrupted:
            self.boredom_score = self._clamp_boredom(BOREDOM_AFTER_SCAN_RESET)
        if (
            not interrupted
            and not face_seen
            and finished_mode in {
                VisionMode.IDLE_GLANCE,
                VisionMode.BOREDOM_SCAN,
            }
        ):
            self._adjust_boredom(BOREDOM_NO_FACE_BONUS)
        if finished_mode == VisionMode.IDLE_GLANCE:
            self._last_idle_glance_at = time.monotonic()
        if face_seen and finished_mode in {
            VisionMode.IDLE_GLANCE,
            VisionMode.BOREDOM_SCAN,
        }:
            self._pending_social_tracking = True
        self._log_camera_off(force=True)

    def _adjust_boredom(self, delta: float) -> None:
        self.boredom_score = self._clamp_boredom(self.boredom_score + delta)

    @staticmethod
    def _clamp_boredom(value: float) -> float:
        return max(BOREDOM_SCORE_MIN, min(BOREDOM_SCORE_MAX, value))

    def _log_camera_off(self, force: bool = False) -> None:
        if force or not self._camera_off_logged:
            logger.info("VISION_SCHEDULER: camera off")
            self._camera_off_logged = True

    async def _wait_or_stop(self, timeout: float) -> None:
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass


__all__ = ["VisionMode", "VisionScheduler"]
