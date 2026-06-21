"""Low-rate V4L2 raw camera capture and face-presence detection for CASE."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    import cv2
    import numpy as np
except Exception as exc:  # Keep the voice assistant bootable without vision deps.
    cv2 = None
    np = None
    _VISION_IMPORT_ERROR: Optional[Exception] = exc
else:
    _VISION_IMPORT_ERROR = None


logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_optional_int(name: str, default: Optional[int] = None) -> Optional[int]:
    value = os.getenv(name)
    if value is None:
        return default
    if not value.strip() or value.strip().lower() == "none":
        return None
    return int(value)


CAMERA_VIDEO_DEVICE = os.getenv("CAMERA_VIDEO_DEVICE", "/dev/video0")
CAMERA_SUBDEV_DEVICE = os.getenv("CAMERA_SUBDEV_DEVICE", "/dev/v4l-subdev0")
CAMERA_PIXEL_FORMAT = os.getenv("CAMERA_PIXEL_FORMAT", "RG10")
CAMERA_CONFIGURE_SUBDEV = _env_bool("CAMERA_CONFIGURE_SUBDEV", True)
CAMERA_SUBDEV_PAD = int(os.getenv("CAMERA_SUBDEV_PAD", "0"))
CAMERA_SUBDEV_MBUS_CODE = os.getenv(
    "CAMERA_SUBDEV_MBUS_CODE",
    "SRGGB10_1X10",
)
CAMERA_CAPTURE_WIDTH = int(os.getenv("CAMERA_CAPTURE_WIDTH", "1640"))
CAMERA_CAPTURE_HEIGHT = int(os.getenv("CAMERA_CAPTURE_HEIGHT", "1232"))
CAMERA_BYTES_PER_LINE = _env_optional_int("CAMERA_BYTES_PER_LINE")
VISION_PROCESS_WIDTH = int(os.getenv("VISION_PROCESS_WIDTH", "640"))
VISION_PROCESS_HEIGHT = int(os.getenv("VISION_PROCESS_HEIGHT", "480"))
VISION_CAPTURE_MMAP_BUFFERS = int(os.getenv("VISION_CAPTURE_MMAP_BUFFERS", "1"))
VISION_TEMP_RAW_PATH = os.getenv("VISION_TEMP_RAW_PATH", "/tmp/case_vision_frame.raw")

BLACK_LEVEL = int(os.getenv("BLACK_LEVEL", os.getenv("RAW_BLACK_LEVEL", "64")))
WHITE_LEVEL = int(os.getenv("WHITE_LEVEL", "1023"))
WB_STRENGTH = float(os.getenv("WB_STRENGTH", "0.85"))
GAMMA = float(os.getenv("GAMMA", os.getenv("RAW_GAMMA", "0.45")))
BAYER_PATTERN = os.getenv(
    "BAYER_PATTERN",
    os.getenv("BAYER_CONVERSION", "BG"),
).upper()
ENABLE_MANUAL_WB = _env_bool("ENABLE_MANUAL_WB", False)
MANUAL_WB_BLUE = float(os.getenv("MANUAL_WB_BLUE", "1.0"))
MANUAL_WB_GREEN = float(os.getenv("MANUAL_WB_GREEN", "1.0"))
MANUAL_WB_RED = float(os.getenv("MANUAL_WB_RED", "1.0"))
ENABLE_GRAY_WORLD_WB = _env_bool("ENABLE_GRAY_WORLD_WB", True)
GRAY_WORLD_WB_MAX_GAIN = float(os.getenv("GRAY_WORLD_WB_MAX_GAIN", "2.5"))

# Backward-compatible names for the earlier Phase 1 API.
RAW_BLACK_LEVEL = BLACK_LEVEL
RAW_GAMMA = GAMMA
RAW_BRIGHTNESS_GAIN = 1.0
BAYER_CONVERSION = BAYER_PATTERN
WB_BLUE_GAIN = MANUAL_WB_BLUE
WB_GREEN_GAIN = MANUAL_WB_GREEN
WB_RED_GAIN = MANUAL_WB_RED
GRAY_WORLD_WB_STRENGTH = WB_STRENGTH

VISION_ENABLED = _env_bool("VISION_ENABLED", True)
VISION_GREETING_ENABLED = _env_bool("VISION_GREETING_ENABLED", True)
VISION_FPS = float(os.getenv("VISION_FPS", "1.0"))
FACE_DETECTION_ENABLED = _env_bool("FACE_DETECTION_ENABLED", True)
FACE_MIN_SIZE = (40, 40)
FACE_SCALE_FACTOR = float(os.getenv("FACE_SCALE_FACTOR", "1.1"))
FACE_MIN_NEIGHBORS = int(os.getenv("FACE_MIN_NEIGHBORS", "5"))
USER_LOST_TIMEOUT_SEC = float(os.getenv("USER_LOST_TIMEOUT_SEC", "5.0"))
USER_DETECTED_COOLDOWN_SEC = float(
    os.getenv("USER_DETECTED_COOLDOWN_SEC", "15.0")
)
VISION_GREETING_COOLDOWN_SEC = float(
    os.getenv("VISION_GREETING_COOLDOWN_SEC", "60.0")
)

CAMERA_VERTICAL_BLANKING = _env_optional_int("CAMERA_VERTICAL_BLANKING", 10000)
CAMERA_EXPOSURE = _env_optional_int("CAMERA_EXPOSURE", 6000)
CAMERA_ANALOGUE_GAIN = _env_optional_int("CAMERA_ANALOGUE_GAIN", 80)
CAMERA_DIGITAL_GAIN = _env_optional_int("CAMERA_DIGITAL_GAIN", 512)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VISION_SNAPSHOT_DIR = PROJECT_ROOT / "output" / "vision_snapshots"


class VisionUnavailableError(RuntimeError):
    """Raised when a required local camera dependency is unavailable."""


class V4L2RawCamera:
    """Capture one 16-bit-word RG10 Bayer frame at a time with ``v4l2-ctl``."""

    def __init__(
        self,
        video_device: str = CAMERA_VIDEO_DEVICE,
        subdev_device: str = CAMERA_SUBDEV_DEVICE,
        capture_width: int = CAMERA_CAPTURE_WIDTH,
        capture_height: int = CAMERA_CAPTURE_HEIGHT,
        process_width: Optional[int] = None,
        process_height: Optional[int] = None,
        pixel_format: str = CAMERA_PIXEL_FORMAT,
        configure_subdev: bool = CAMERA_CONFIGURE_SUBDEV,
        subdev_pad: int = CAMERA_SUBDEV_PAD,
        subdev_mbus_code: str = CAMERA_SUBDEV_MBUS_CODE,
        mmap_buffers: int = VISION_CAPTURE_MMAP_BUFFERS,
        bytes_per_line: Optional[int] = CAMERA_BYTES_PER_LINE,
        raw_path: str | Path = VISION_TEMP_RAW_PATH,
        black_level: int = BLACK_LEVEL,
        white_level: int = WHITE_LEVEL,
        gamma: float = GAMMA,
        brightness_gain: float = RAW_BRIGHTNESS_GAIN,
        bayer_conversion: str = BAYER_PATTERN,
        enable_manual_wb: bool = ENABLE_MANUAL_WB,
        wb_blue_gain: float = MANUAL_WB_BLUE,
        wb_green_gain: float = MANUAL_WB_GREEN,
        wb_red_gain: float = MANUAL_WB_RED,
        enable_gray_world_wb: bool = ENABLE_GRAY_WORLD_WB,
        gray_world_wb_strength: float = WB_STRENGTH,
        gray_world_wb_max_gain: float = GRAY_WORLD_WB_MAX_GAIN,
        color_profile: str = "case_legacy",
        vertical_blanking: Optional[int] = CAMERA_VERTICAL_BLANKING,
        exposure: Optional[int] = CAMERA_EXPOSURE,
        analogue_gain: Optional[int] = CAMERA_ANALOGUE_GAIN,
        digital_gain: Optional[int] = CAMERA_DIGITAL_GAIN,
        capture_timeout_sec: float = 15.0,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        # Backward-compatible aliases: direct camera callers that still pass
        # width/height mean capture size. The test CLI uses explicit names.
        if width is not None:
            capture_width = width
            if process_width is None:
                process_width = width
        if height is not None:
            capture_height = height
            if process_height is None:
                process_height = height
        process_width = (
            VISION_PROCESS_WIDTH if process_width is None else process_width
        )
        process_height = (
            VISION_PROCESS_HEIGHT if process_height is None else process_height
        )

        if min(capture_width, capture_height, process_width, process_height) <= 0:
            raise ValueError("Capture and processing dimensions must be positive")
        if mmap_buffers not in {1, 2}:
            raise ValueError("mmap_buffers must be 1 or 2")
        if bytes_per_line is not None and bytes_per_line <= 0:
            raise ValueError("bytes_per_line must be positive when configured")
        if gamma <= 0 or brightness_gain <= 0:
            raise ValueError("Gamma and brightness gain must be positive")
        if white_level <= black_level:
            raise ValueError("White level must be greater than black level")
        bayer_conversion = bayer_conversion.upper()
        if bayer_conversion not in {"RG", "BG", "GR", "GB"}:
            raise ValueError("Bayer conversion must be RG, BG, GR, or GB")
        if min(wb_blue_gain, wb_green_gain, wb_red_gain) <= 0:
            raise ValueError("Manual white-balance gains must be positive")
        if gray_world_wb_max_gain < 1.0:
            raise ValueError("Gray-world maximum gain must be at least 1.0")

        self.video_device = video_device
        self.subdev_device = subdev_device
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.process_width = process_width
        self.process_height = process_height
        # Preserve the original helper attributes for existing direct callers.
        self.width = capture_width
        self.height = capture_height
        self.pixel_format = pixel_format
        self.configure_subdev = configure_subdev
        self.subdev_pad = subdev_pad
        self.subdev_mbus_code = subdev_mbus_code
        self.mmap_buffers = mmap_buffers
        self.bytes_per_line = bytes_per_line
        self.raw_path = Path(raw_path)
        self.black_level = black_level
        self.white_level = white_level
        self.gamma = gamma
        self.brightness_gain = brightness_gain
        self.bayer_conversion = bayer_conversion
        self.enable_manual_wb = enable_manual_wb
        self.wb_blue_gain = wb_blue_gain
        self.wb_green_gain = wb_green_gain
        self.wb_red_gain = wb_red_gain
        self.enable_gray_world_wb = enable_gray_world_wb
        self.gray_world_wb_strength = max(0.0, min(gray_world_wb_strength, 1.0))
        self.gray_world_wb_max_gain = gray_world_wb_max_gain
        self.color_profile = color_profile
        self.vertical_blanking = vertical_blanking
        self.exposure = exposure
        self.analogue_gain = analogue_gain
        self.digital_gain = digital_gain
        self.capture_timeout_sec = capture_timeout_sec
        self._capture_lock = threading.Lock()
        self._subdev_format_logged = False
        self._video_format_logged = False
        self._raw_shape_logged = False
        self._processed_size_logged = False
        self._control_logs_emitted: set[str] = set()
        self._current_controls_logged = False
        self._last_overexposure_warning_at = float("-inf")

    @property
    def expected_raw_bytes(self) -> int:
        bytes_per_line = self.bytes_per_line or self._fallback_bytes_per_line()
        return bytes_per_line * self.capture_height

    def initialize(self) -> None:
        self._require_dependencies()
        if shutil.which("v4l2-ctl") is None:
            raise VisionUnavailableError("v4l2-ctl is not installed")
        if not Path(self.video_device).exists():
            raise VisionUnavailableError(
                f"camera video device does not exist: {self.video_device}"
            )
        if self.configure_subdev and not Path(self.subdev_device).exists():
            raise VisionUnavailableError(
                f"camera subdevice does not exist: {self.subdev_device}"
            )

        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "VISION: V4L2 raw camera initialized device=%s format=%s "
            "capture=%sx%s process=%sx%s",
            self.video_device,
            self.pixel_format,
            self.capture_width,
            self.capture_height,
            self.process_width,
            self.process_height,
        )
        logger.info("VISION: color profile=%s", self.color_profile)
        logger.info("VISION: bayer=%s", self.bayer_conversion)
        logger.info(
            "VISION: black_level=%s white_level=%s",
            self.black_level,
            self.white_level,
        )
        logger.info(
            "VISION: gray_world_wb=%s strength=%.2f",
            self.enable_gray_world_wb,
            self.gray_world_wb_strength,
        )
        logger.info("VISION: gamma=%.2f", self.gamma)
        logger.info("VISION: manual_wb=%s", self.enable_manual_wb)

    def configure_subdev_format(self) -> None:
        """Match the IMX219 source-pad mode to the Unicam capture format."""
        format_spec = (
            f"pad={self.subdev_pad},width={self.capture_width},"
            f"height={self.capture_height},"
            f"code={self.subdev_mbus_code},field=none"
        )
        try:
            result = subprocess.run(
                [
                    "v4l2-ctl",
                    "-d",
                    self.subdev_device,
                    "--set-subdev-fmt",
                    format_spec,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5.0,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise VisionUnavailableError(
                f"could not configure camera subdevice: {exc}"
            ) from exc

        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise VisionUnavailableError(
                "could not configure camera subdevice "
                f"{self.subdev_device}: {detail or f'exit {result.returncode}'}"
            )
        log = logger.info if not self._subdev_format_logged else logger.debug
        log(
            "VISION: set subdev fmt width=%s height=%s code=%s",
            self.capture_width,
            self.capture_height,
            self.subdev_mbus_code,
        )
        self._subdev_format_logged = True

    def apply_configured_controls(self) -> bool:
        controls = {
            "vertical_blanking": self.vertical_blanking,
            "exposure": self.exposure,
            "analogue_gain": self.analogue_gain,
            "digital_gain": self.digital_gain,
        }
        configured = {key: value for key, value in controls.items() if value is not None}
        success = self.set_controls(**configured)
        if not self._current_controls_logged:
            self.log_current_controls(
                "exposure",
                "analogue_gain",
                "digital_gain",
            )
            self._current_controls_logged = True
        return success

    def set_controls(self, **controls: int) -> bool:
        """Set explicitly supplied IMX219 subdevice controls; do nothing by default."""
        if not controls:
            return True
        if shutil.which("v4l2-ctl") is None:
            logger.warning("VISION: cannot set camera controls; v4l2-ctl is missing")
            return False
        if not Path(self.subdev_device).exists():
            logger.warning(
                "VISION: cannot set camera controls; subdevice is missing: %s",
                self.subdev_device,
            )
            return False

        all_succeeded = True
        supported = {
            "vertical_blanking",
            "exposure",
            "analogue_gain",
            "digital_gain",
        }
        for name, value in controls.items():
            if name not in supported:
                raise ValueError(f"Unsupported camera control: {name}")
            try:
                result = subprocess.run(
                    [
                        "v4l2-ctl",
                        "-d",
                        self.subdev_device,
                        f"--set-ctrl={name}={int(value)}",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=5.0,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                logger.warning("VISION: failed to set %s: %s", name, exc)
                all_succeeded = False
                continue
            if result.returncode != 0:
                logger.warning(
                    "VISION: failed to set control %s=%s: %s; continuing capture",
                    name,
                    value,
                    result.stderr.strip() or f"exit {result.returncode}",
                )
                all_succeeded = False
                continue
            log = (
                logger.info
                if name not in self._control_logs_emitted
                else logger.debug
            )
            log("VISION: set control %s=%s", name, value)
            self._control_logs_emitted.add(name)
        return all_succeeded

    def log_current_controls(self, *control_names: str) -> None:
        """Read back effective sensor controls after format negotiation."""
        for name in control_names:
            try:
                result = subprocess.run(
                    [
                        "v4l2-ctl",
                        "-d",
                        self.subdev_device,
                        f"--get-ctrl={name}",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=5.0,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                logger.warning("VISION: failed to read current control %s: %s", name, exc)
                continue
            if result.returncode != 0:
                logger.warning(
                    "VISION: failed to read current control %s: %s",
                    name,
                    result.stderr.strip() or f"exit {result.returncode}",
                )
                continue
            output = result.stdout.strip() or result.stderr.strip()
            match = re.search(rf"\b{re.escape(name)}\s*:\s*(-?\d+)", output)
            value = match.group(1) if match else output
            logger.info("VISION: current control %s=%s", name, value)

    def capture_bgr_frame(self) -> Optional[Any]:
        """Capture and process one frame, returning ``None`` on any camera error."""
        with self._capture_lock:
            return self._capture_bgr_frame()

    def _capture_bgr_frame(self) -> Optional[Any]:
        raw = self._capture_raw_frame()
        if raw is None:
            return None
        try:
            frame = self.raw_to_bgr(raw)
            if frame is not None:
                logger.debug("VISION: captured frame")
            return frame
        except Exception:
            logger.exception("VISION: unexpected raw image processing error")
            return None

    def capture_raw_frame(self) -> Optional[Any]:
        """Capture one visible, stride-cropped uint16 Bayer frame."""
        with self._capture_lock:
            return self._capture_raw_frame()

    def _capture_raw_frame(self) -> Optional[Any]:
        try:
            self._require_dependencies()
            if self.configure_subdev:
                self.configure_subdev_format()
            self.apply_configured_controls()
            self.raw_path.unlink(missing_ok=True)
            result = subprocess.run(
                self.capture_command(),
                capture_output=True,
                text=True,
                check=False,
                timeout=self.capture_timeout_sec,
            )
            if result.returncode != 0:
                logger.warning(
                    "VISION: capture failed: %s",
                    result.stderr.strip() or f"v4l2-ctl exited {result.returncode}",
                )
                return None
            if not self.raw_path.is_file():
                logger.warning(
                    "VISION: raw capture file was not created: %s",
                    self.raw_path,
                )
                return None

            bytes_per_line, size_image = self.query_video_format()
            raw = self.read_raw_frame(bytes_per_line, size_image, result)
            return raw
        except subprocess.TimeoutExpired:
            logger.warning("VISION: capture timed out after %.1fs", self.capture_timeout_sec)
        except (OSError, ValueError, VisionUnavailableError) as exc:
            logger.warning("VISION: camera capture failed: %s", exc)
        except Exception:
            logger.exception("VISION: unexpected camera capture error")
        return None

    def query_video_format(self) -> tuple[int, int]:
        """Return negotiated ``bytesperline`` and ``sizeimage`` values."""
        if self.bytes_per_line is not None:
            bytes_per_line = self.bytes_per_line
            size_image = bytes_per_line * self.capture_height
        else:
            bytes_per_line, size_image = self._query_video_format_with_v4l2()

        log = logger.info if not self._video_format_logged else logger.debug
        log(
            "VISION: video fmt width=%s height=%s pixelformat=%s "
            "bytesperline=%s sizeimage=%s",
            self.capture_width,
            self.capture_height,
            self.pixel_format,
            bytes_per_line,
            size_image,
        )
        self._video_format_logged = True
        return bytes_per_line, size_image

    def _query_video_format_with_v4l2(self) -> tuple[int, int]:
        try:
            result = subprocess.run(
                ["v4l2-ctl", "-d", self.video_device, "--get-fmt-video"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5.0,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("VISION: could not query video format: %s", exc)
            return self._fallback_video_format()

        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        bytes_match = re.search(r"Bytes per Line\s*:\s*(\d+)", output, re.IGNORECASE)
        size_match = re.search(r"Size Image\s*:\s*(\d+)", output, re.IGNORECASE)
        if result.returncode != 0 or bytes_match is None:
            logger.warning(
                "VISION: video format query did not report Bytes per Line; "
                "using confirmed-mode fallback: %s",
                output.strip() or f"exit {result.returncode}",
            )
            return self._fallback_video_format()

        bytes_per_line = int(bytes_match.group(1))
        size_image = (
            int(size_match.group(1))
            if size_match is not None
            else bytes_per_line * self.capture_height
        )
        return bytes_per_line, size_image

    def _fallback_video_format(self) -> tuple[int, int]:
        bytes_per_line = self._fallback_bytes_per_line()
        return bytes_per_line, bytes_per_line * self.capture_height

    def _fallback_bytes_per_line(self) -> int:
        confirmed_strides = {
            (1640, 1232, "RG10"): 3296,
            (3280, 2464, "RG10"): 6560,
        }
        key = (self.capture_width, self.capture_height, self.pixel_format.upper())
        if key in confirmed_strides:
            return confirmed_strides[key]
        logger.warning(
            "VISION: no confirmed stride for %sx%s %s; assuming width * 2",
            self.capture_width,
            self.capture_height,
            self.pixel_format,
        )
        return self.capture_width * 2

    def read_raw_frame(
        self,
        bytes_per_line: int,
        size_image: int,
        capture_result: Optional[Any] = None,
    ) -> Optional[Any]:
        """Read a stride-padded uint16 frame and crop it to the visible width."""
        if bytes_per_line % 2:
            logger.warning("VISION: odd RG10 bytesperline is invalid: %s", bytes_per_line)
            return None
        stride_pixels = bytes_per_line // 2
        if stride_pixels < self.capture_width:
            logger.warning(
                "VISION: stride width %s is smaller than visible width %s",
                stride_pixels,
                self.capture_width,
            )
            return None

        expected_bytes = bytes_per_line * self.capture_height
        if size_image != expected_bytes:
            logger.warning(
                "VISION: negotiated sizeimage=%s differs from stride*height=%s",
                size_image,
                expected_bytes,
            )
        actual_bytes = self.raw_path.stat().st_size
        if actual_bytes != expected_bytes:
            command_detail = ""
            if capture_result is not None:
                command_detail = (
                    capture_result.stderr.strip() or capture_result.stdout.strip()
                )
            logger.warning(
                "VISION: raw frame size mismatch expected=%s actual=%s path=%s "
                "v4l2_output=%r",
                expected_bytes,
                actual_bytes,
                self.raw_path,
                command_detail,
            )
            return None

        raw = np.fromfile(self.raw_path, dtype=np.uint16)
        expected_samples = stride_pixels * self.capture_height
        if raw.size != expected_samples:
            logger.warning(
                "VISION: raw sample count mismatch expected=%s actual=%s",
                expected_samples,
                raw.size,
            )
            return None
        padded = raw.reshape((self.capture_height, stride_pixels))
        visible = padded[:, : self.capture_width]
        log = logger.info if not self._raw_shape_logged else logger.debug
        log(
            "VISION: raw parsed shape=%s cropped=%s",
            padded.shape,
            visible.shape,
        )
        self._raw_shape_logged = True
        return visible

    def capture_command(self) -> list[str]:
        return [
            "v4l2-ctl",
            "-d",
            self.video_device,
            (
                f"--set-fmt-video=width={self.capture_width},"
                f"height={self.capture_height},"
                f"pixelformat={self.pixel_format}"
            ),
            f"--stream-mmap={self.mmap_buffers}",
            "--stream-count=1",
            f"--stream-to={self.raw_path}",
        ]

    def raw_to_bgr(self, raw: Any) -> Optional[Any]:
        """Apply CASE's known-good fixed-level RG10 color pipeline."""
        self._require_dependencies()
        usable_range = float(self.white_level - self.black_level)
        corrected = raw.astype(np.float32)
        corrected = np.clip(
            corrected - float(self.black_level),
            0.0,
            usable_range,
        )
        raw8 = np.clip(corrected / usable_range * 255.0, 0, 255).astype(np.uint8)
        bgr8 = cv2.cvtColor(raw8, self.bayer_conversion_code())
        if self.enable_gray_world_wb:
            bgr8 = self.gray_world_white_balance(
                bgr8,
                strength=self.gray_world_wb_strength,
                max_gain=self.gray_world_wb_max_gain,
            )
        if self.enable_manual_wb:
            bgr8 = self.apply_manual_white_balance(
                bgr8,
                blue_gain=self.wb_blue_gain,
                green_gain=self.wb_green_gain,
                red_gain=self.wb_red_gain,
            )
        bgr8 = self.apply_gamma(bgr8, self.gamma)
        if self.brightness_gain != 1.0:
            bgr8 = self.apply_brightness_gain(bgr8, self.brightness_gain)
        bgr8 = self.resize(bgr8, self.process_width, self.process_height)
        if not self._processed_size_logged:
            logger.info(
                "VISION: processed frame resized to %sx%s",
                self.process_width,
                self.process_height,
            )
            self._processed_size_logged = True
        self.warn_if_overexposed(bgr8)
        return bgr8

    def bayer_conversion_code(self) -> int:
        conversions = {
            "RG": cv2.COLOR_BayerRG2BGR,
            "BG": cv2.COLOR_BayerBG2BGR,
            "GR": cv2.COLOR_BayerGR2BGR,
            "GB": cv2.COLOR_BayerGB2BGR,
        }
        return conversions[self.bayer_conversion]

    @staticmethod
    def gray_world_white_balance(
        image: Any,
        strength: float = WB_STRENGTH,
        max_gain: float = GRAY_WORLD_WB_MAX_GAIN,
    ) -> Any:
        image_float = image.astype(np.float32)
        height, width = image_float.shape[:2]
        crop = image_float[
            height // 6 : 5 * height // 6,
            width // 6 : 5 * width // 6,
        ]
        channel_means = crop.reshape(-1, 3).mean(axis=0)
        target = float(channel_means.mean())
        safe_means = np.maximum(channel_means, 1.0)
        full_gains = target / safe_means
        full_gains = np.clip(full_gains, 0.5, max_gain)
        gains = (
            (1.0 - strength) * np.ones(3, dtype=np.float32)
            + strength * full_gains
        )
        return np.clip(image_float * gains, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_manual_white_balance(
        image: Any,
        blue_gain: float,
        green_gain: float,
        red_gain: float,
    ) -> Any:
        """Apply explicit channel gains in OpenCV's BGR channel order."""
        gains = np.array([blue_gain, green_gain, red_gain], dtype=np.float32)
        return np.clip(image.astype(np.float32) * gains, 0, 255).astype(np.uint8)

    def warn_if_overexposed(self, image: Any) -> float:
        clipped_ratio = float(np.mean(np.max(image, axis=2) > 250))
        now = time.monotonic()
        if clipped_ratio > 0.05 and now - self._last_overexposure_warning_at >= 10.0:
            logger.warning(
                "VISION: warning image may be overexposed clipped_ratio=%.4f",
                clipped_ratio,
            )
            self._last_overexposure_warning_at = now
        return clipped_ratio

    @staticmethod
    def apply_gamma(image: Any, gamma: float) -> Any:
        table = np.clip(
            (np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0,
            0,
            255,
        ).astype(np.uint8)
        return cv2.LUT(image, table)

    @staticmethod
    def apply_brightness_gain(image: Any, gain: float) -> Any:
        return np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    @staticmethod
    def resize(image: Any, width: int, height: int) -> Any:
        if image.shape[1] == width and image.shape[0] == height:
            return image
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _require_dependencies() -> None:
        if cv2 is None or np is None:
            raise VisionUnavailableError(
                "OpenCV/numpy vision dependencies are unavailable: "
                f"{_VISION_IMPORT_ERROR}"
            )


class VisionEngine:
    """Async, low-FPS face-presence engine that publishes message-bus events."""

    def __init__(
        self,
        message_bus: Any,
        camera: Optional[V4L2RawCamera] = None,
        fps: float = VISION_FPS,
        face_detection_enabled: bool = FACE_DETECTION_ENABLED,
        face_min_size: tuple[int, int] = FACE_MIN_SIZE,
        face_scale_factor: float = FACE_SCALE_FACTOR,
        face_min_neighbors: int = FACE_MIN_NEIGHBORS,
        user_lost_timeout_sec: float = USER_LOST_TIMEOUT_SEC,
        user_detected_cooldown_sec: float = USER_DETECTED_COOLDOWN_SEC,
    ) -> None:
        V4L2RawCamera._require_dependencies()
        if fps <= 0:
            raise ValueError("Vision FPS must be greater than zero")
        self.message_bus = message_bus
        self.camera = camera or V4L2RawCamera()
        self.fps = min(fps, 2.0)
        if fps > 2.0:
            logger.warning("VISION: requested FPS %.2f capped at 2 FPS", fps)
        self.face_detection_enabled = face_detection_enabled
        self.face_min_size = face_min_size
        self.face_scale_factor = face_scale_factor
        self.face_min_neighbors = face_min_neighbors
        self.user_lost_timeout_sec = user_lost_timeout_sec
        self.user_detected_cooldown_sec = user_detected_cooldown_sec
        self.user_present = False
        self.last_face_seen_at: Optional[float] = None
        self.last_detected_event_at = float("-inf")
        self._detection_event_pending = False
        self._stop_event = asyncio.Event()
        self._last_debug_log_at = float("-inf")
        self._last_error_event_at = float("-inf")

        self.face_cascade = None
        if self.face_detection_enabled:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                raise VisionUnavailableError(f"Could not load Haar cascade: {cascade_path}")
            self.face_cascade = cascade

    async def run(self) -> None:
        await self._publish_status("starting")
        try:
            await asyncio.to_thread(self.camera.initialize)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("VISION: disabled due to camera error: %s", exc)
            await self._publish_error(str(exc), force=True)
            await self._publish_status("disabled", reason=str(exc))
            return

        await self._publish_status("running")
        frame_interval = 1.0 / self.fps
        while not self._stop_event.is_set():
            loop_started = time.monotonic()
            frame = await asyncio.to_thread(self.camera.capture_bgr_frame)
            if frame is None:
                reason = "camera capture failed"
                logger.warning("VISION: disabled due to camera error")
                await self._publish_error(reason, force=True)
                await self._publish_status("disabled", reason=reason)
                return

            faces = self.detect_faces(frame)
            payload = self.make_frame_payload(frame, faces)
            payload["frame_bgr"] = frame
            await self.message_bus.publish("VISION_FRAME_READY", payload)
            await self._update_presence(faces, payload)

            if time.monotonic() - self._last_debug_log_at >= 10.0:
                logger.info("VISION: captured frame faces=%s", len(faces))
                self._last_debug_log_at = time.monotonic()

            remaining = frame_interval - (time.monotonic() - loop_started)
            if remaining > 0:
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    pass

    def stop(self) -> None:
        self._stop_event.set()

    def detect_faces(self, frame_bgr: Any) -> list[dict[str, Any]]:
        if self.face_cascade is None:
            return []
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        detected = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.face_scale_factor,
            minNeighbors=self.face_min_neighbors,
            minSize=self.face_min_size,
        )
        return [
            {
                "x": int(x),
                "y": int(y),
                "w": int(width),
                "h": int(height),
                "confidence": None,
            }
            for x, y, width, height in detected
        ]

    @staticmethod
    def make_frame_payload(frame: Any, faces: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "source": "vision_engine",
            "faces": faces,
            "frame_width": int(frame.shape[1]),
            "frame_height": int(frame.shape[0]),
            "timestamp": time.time(),
        }

    async def _update_presence(
        self,
        faces: list[dict[str, Any]],
        frame_payload: dict[str, Any],
    ) -> None:
        now = time.monotonic()
        event_payload = {key: value for key, value in frame_payload.items() if key != "frame_bgr"}
        if faces:
            self.last_face_seen_at = now
            if not self.user_present:
                self.user_present = True
                self._detection_event_pending = True
            if (
                self._detection_event_pending
                and now - self.last_detected_event_at
                >= self.user_detected_cooldown_sec
            ):
                await self.message_bus.publish("VISION_USER_DETECTED", event_payload)
                self.last_detected_event_at = now
                self._detection_event_pending = False
                logger.info("VISION: face detected count=%s", len(faces))
                logger.info("VISION: published VISION_USER_DETECTED")
            return

        if (
            self.user_present
            and self.last_face_seen_at is not None
            and now - self.last_face_seen_at >= self.user_lost_timeout_sec
        ):
            self.user_present = False
            self._detection_event_pending = False
            await self.message_bus.publish("VISION_USER_LOST", event_payload)
            logger.info("VISION: user lost")

    async def capture_scene_snapshot(
        self,
        output_dir: str | Path = VISION_SNAPSHOT_DIR,
    ) -> Optional[Path]:
        """Capture one processed frame for local/on-demand scene inspection."""
        # TODO: Invoke Gemini/cloud vision only for explicit requests such as
        # "what do you see?" or "describe the room".
        frame = await asyncio.to_thread(self.camera.capture_bgr_frame)
        if frame is None:
            await self._publish_error("scene snapshot capture failed")
            return None
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / datetime.now().strftime("snapshot_%Y%m%d_%H%M%S.jpg")
        if not cv2.imwrite(str(path), frame):
            await self._publish_error(f"could not write snapshot: {path}")
            return None
        await self.message_bus.publish(
            "VISION_STATUS",
            {"source": "vision_engine", "status": "snapshot_ready", "path": str(path)},
        )
        return path

    async def _publish_status(self, status: str, **extra: Any) -> None:
        await self.message_bus.publish(
            "VISION_STATUS",
            {
                "source": "vision_engine",
                "status": status,
                "timestamp": time.time(),
                **extra,
            },
        )
        await asyncio.sleep(0)

    async def _publish_error(self, message: str, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_error_event_at < 10.0:
            return
        self._last_error_event_at = now
        await self.message_bus.publish(
            "VISION_ERROR",
            {"source": "vision_engine", "error": message, "timestamp": time.time()},
        )
        await asyncio.sleep(0)
