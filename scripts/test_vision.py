#!/usr/bin/env python3
"""Headless V4L2/OpenCV smoke test for CASE's Optic Nerve."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from middleware.message_bus import AsyncMessageBus  # noqa: E402
from src.vision.vision_engine import (  # noqa: E402
    BAYER_PATTERN,
    BLACK_LEVEL,
    CAMERA_CAPTURE_HEIGHT,
    CAMERA_CAPTURE_WIDTH,
    CAMERA_ANALOGUE_GAIN,
    CAMERA_DIGITAL_GAIN,
    CAMERA_EXPOSURE,
    CAMERA_SUBDEV_DEVICE,
    CAMERA_VIDEO_DEVICE,
    CAMERA_VERTICAL_BLANKING,
    ENABLE_GRAY_WORLD_WB,
    ENABLE_MANUAL_WB,
    GAMMA,
    MANUAL_WB_BLUE,
    MANUAL_WB_GREEN,
    MANUAL_WB_RED,
    WB_STRENGTH,
    WHITE_LEVEL,
    VISION_PROCESS_HEIGHT,
    VISION_PROCESS_WIDTH,
    VISION_SNAPSHOT_DIR,
    VISION_TEMP_RAW_PATH,
    V4L2RawCamera,
    VisionEngine,
    VisionUnavailableError,
    cv2,
)


logger = logging.getLogger("test_vision")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture RG10 frames and test CASE face-presence detection."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=VISION_PROCESS_WIDTH,
        help="Processed/output frame width (not the sensor capture width).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=VISION_PROCESS_HEIGHT,
        help="Processed/output frame height (not the sensor capture height).",
    )
    parser.add_argument(
        "--capture-width",
        type=int,
        default=CAMERA_CAPTURE_WIDTH,
        help="Sensor capture width; override only for a proven streamable mode.",
    )
    parser.add_argument(
        "--capture-height",
        type=int,
        default=CAMERA_CAPTURE_HEIGHT,
        help="Sensor capture height; override only for a proven streamable mode.",
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--video-device", default=CAMERA_VIDEO_DEVICE)
    parser.add_argument("--subdev-device", default=CAMERA_SUBDEV_DEVICE)
    parser.add_argument(
        "--no-configure-subdev",
        action="store_true",
        help="Leave the sensor pad format unchanged (advanced/debug use only).",
    )
    parser.add_argument(
        "--subdev-code",
        default="SRGGB10_1X10",
        help="Sensor media-bus code used for capture.",
    )
    parser.add_argument("--raw-path", default=VISION_TEMP_RAW_PATH)
    parser.add_argument("--mmap-buffers", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--bayer",
        choices=("BG", "GB", "RG", "GR"),
        default=BAYER_PATTERN,
        help="OpenCV Bayer conversion pattern.",
    )
    parser.add_argument("--black-level", type=int, default=BLACK_LEVEL)
    parser.add_argument("--white-level", type=int, default=WHITE_LEVEL)
    parser.add_argument("--wb-strength", type=float, default=WB_STRENGTH)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--wb-blue", type=float, default=MANUAL_WB_BLUE)
    parser.add_argument("--wb-green", type=float, default=MANUAL_WB_GREEN)
    parser.add_argument("--wb-red", type=float, default=MANUAL_WB_RED)
    parser.add_argument(
        "--disable-gray-world-wb",
        action="store_true",
        help="Disable automatic gray-world white balance for raw color testing.",
    )
    manual_wb = parser.add_mutually_exclusive_group()
    manual_wb.add_argument(
        "--manual-wb",
        dest="enable_manual_wb",
        action="store_true",
        help="Enable explicit BGR channel gains.",
    )
    manual_wb.add_argument(
        "--disable-manual-wb",
        dest="enable_manual_wb",
        action="store_false",
        help="Disable explicit BGR channel gains.",
    )
    parser.set_defaults(enable_manual_wb=ENABLE_MANUAL_WB)
    parser.add_argument(
        "--vertical-blanking",
        type=int,
        default=CAMERA_VERTICAL_BLANKING,
        help="IMX219 vertical_blanking control value.",
    )
    parser.add_argument(
        "--exposure",
        type=int,
        default=CAMERA_EXPOSURE,
        help="IMX219 exposure control value.",
    )
    parser.add_argument(
        "--analogue-gain",
        type=int,
        default=CAMERA_ANALOGUE_GAIN,
        help="IMX219 analogue_gain control value.",
    )
    parser.add_argument(
        "--digital-gain",
        type=int,
        default=CAMERA_DIGITAL_GAIN,
        help="IMX219 digital_gain control value.",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Save one processed JPG and exit.",
    )
    parser.add_argument(
        "--save-full-processed",
        action="store_true",
        help="In snapshot mode, keep the processed JPG at the raw capture size.",
    )
    parser.add_argument(
        "--test-bayer-codes",
        action="store_true",
        help="Capture and save RG, BG, GR, and GB conversion variants.",
    )
    parser.add_argument(
        "--legacy-color",
        "--case-color-profile",
        dest="legacy_color",
        action="store_true",
        help="Force the known-good CASE BayerBG legacy color pipeline.",
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save every processed frame with face boxes; no GUI is used.",
    )
    return parser.parse_args()


def build_camera(args: argparse.Namespace) -> V4L2RawCamera:
    save_full = args.snapshot and args.save_full_processed
    raw_bayer_test = args.test_bayer_codes
    if args.legacy_color:
        bayer = "BG"
        black_level = 64
        white_level = 1023
        wb_strength = 0.85
        gamma = 0.45
        gray_world_enabled = True
        manual_wb_enabled = False
        wb_blue, wb_green, wb_red = 1.0, 1.0, 1.0
        color_profile = "case_legacy"
    elif raw_bayer_test:
        bayer = args.bayer
        black_level = args.black_level
        white_level = args.white_level
        wb_strength = args.wb_strength
        gamma = args.gamma
        gray_world_enabled = False
        manual_wb_enabled = False
        wb_blue, wb_green, wb_red = 1.0, 1.0, 1.0
        color_profile = "bayer_diagnostic"
    else:
        bayer = args.bayer
        black_level = args.black_level
        white_level = args.white_level
        wb_strength = args.wb_strength
        gamma = args.gamma
        gray_world_enabled = (
            ENABLE_GRAY_WORLD_WB and not args.disable_gray_world_wb
        )
        manual_wb_enabled = args.enable_manual_wb
        wb_blue, wb_green, wb_red = args.wb_blue, args.wb_green, args.wb_red
        legacy_settings = (
            bayer == "BG"
            and black_level == 64
            and white_level == 1023
            and wb_strength == 0.85
            and gamma == 0.45
            and gray_world_enabled
            and not manual_wb_enabled
            and (wb_blue, wb_green, wb_red) == (1.0, 1.0, 1.0)
        )
        color_profile = "case_legacy" if legacy_settings else "custom"

    return V4L2RawCamera(
        video_device=args.video_device,
        subdev_device=args.subdev_device,
        capture_width=args.capture_width,
        capture_height=args.capture_height,
        process_width=args.capture_width if save_full else args.width,
        process_height=args.capture_height if save_full else args.height,
        configure_subdev=not args.no_configure_subdev,
        subdev_mbus_code=args.subdev_code,
        mmap_buffers=args.mmap_buffers,
        black_level=black_level,
        white_level=white_level,
        gamma=gamma,
        bayer_conversion=bayer,
        enable_manual_wb=manual_wb_enabled,
        wb_blue_gain=wb_blue,
        wb_green_gain=wb_green,
        wb_red_gain=wb_red,
        enable_gray_world_wb=gray_world_enabled,
        gray_world_wb_strength=wb_strength,
        color_profile=color_profile,
        vertical_blanking=args.vertical_blanking,
        exposure=args.exposure,
        analogue_gain=args.analogue_gain,
        digital_gain=args.digital_gain,
        raw_path=args.raw_path,
    )


async def run_snapshot(camera: V4L2RawCamera) -> int:
    bus = AsyncMessageBus()
    engine = VisionEngine(bus, camera=camera)
    await asyncio.to_thread(camera.initialize)
    path = await engine.capture_scene_snapshot()
    if path is None:
        print("VISION_ERROR: snapshot capture failed", flush=True)
        return 1
    print(f"VISION_SCENE_SNAPSHOT_READY: {path}", flush=True)
    return 0


async def run_bayer_code_test(camera: V4L2RawCamera) -> int:
    camera.initialize()
    VISION_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    raw = camera.capture_raw_frame()
    if raw is None:
        print("VISION_ERROR: raw capture for Bayer comparison failed", flush=True)
        return 1

    failed = False
    for code in ("RG", "BG", "GR", "GB"):
        camera.bayer_conversion = code
        frame = camera.raw_to_bgr(raw)
        if frame is None:
            print(f"VISION_ERROR: Bayer {code} capture failed", flush=True)
            failed = True
            continue
        path = VISION_SNAPSHOT_DIR / f"snapshot_bayer_{code}.jpg"
        if not cv2.imwrite(str(path), frame):
            print(f"VISION_ERROR: could not save {path}", flush=True)
            failed = True
            continue
        print(f"VISION_BAYER_SNAPSHOT_READY: {path}", flush=True)
    return 1 if failed else 0


async def run_detection(camera: V4L2RawCamera, args: argparse.Namespace) -> int:
    bus = AsyncMessageBus()
    engine = VisionEngine(bus, camera=camera, fps=args.fps)
    debug_dir = PROJECT_ROOT / "output" / "vision_debug"
    failure_seen = False

    async def print_event(name: str, payload: dict) -> None:
        faces = len(payload.get("faces", []))
        print(f"{name}: faces={faces} timestamp={payload.get('timestamp')}", flush=True)

    async def on_detected(payload: dict) -> None:
        await print_event("VISION_USER_DETECTED", payload)

    async def on_lost(payload: dict) -> None:
        await print_event("VISION_USER_LOST", payload)

    async def on_status(payload: dict) -> None:
        nonlocal failure_seen
        if payload.get("status") == "disabled":
            failure_seen = True
        print(f"VISION_STATUS: {payload}", flush=True)

    async def on_error(payload: dict) -> None:
        nonlocal failure_seen
        failure_seen = True
        print(f"VISION_ERROR: {payload.get('error')}", flush=True)

    async def save_debug_frame(payload: dict) -> None:
        if not args.save_debug:
            return
        frame = payload.get("frame_bgr")
        if frame is None:
            return
        debug_dir.mkdir(parents=True, exist_ok=True)
        annotated = frame.copy()
        for face in payload.get("faces", []):
            x, y, width, height = (face[key] for key in ("x", "y", "w", "h"))
            cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 255, 0), 2)
        filename = datetime.now().strftime("frame_%Y%m%d_%H%M%S_%f.jpg")
        path = debug_dir / filename
        if cv2.imwrite(str(path), annotated):
            print(f"VISION_FRAME_READY: saved {path}", flush=True)

    bus.subscribe("VISION_USER_DETECTED", on_detected)
    bus.subscribe("VISION_USER_LOST", on_lost)
    bus.subscribe("VISION_STATUS", on_status)
    bus.subscribe("VISION_ERROR", on_error)
    bus.subscribe("VISION_FRAME_READY", save_debug_frame)

    print("Vision test running headlessly. Press Ctrl-C to stop.", flush=True)
    await engine.run()
    return 1 if failure_seen else 0


async def async_main(args: argparse.Namespace) -> int:
    camera = build_camera(args)
    if args.test_bayer_codes:
        return await run_bayer_code_test(camera)
    if args.snapshot:
        return await run_snapshot(camera)
    return await run_detection(camera, args)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    try:
        return asyncio.run(async_main(args))
    except (VisionUnavailableError, ValueError) as exc:
        logger.error("Vision unavailable: %s", exc)
        return 1
    except KeyboardInterrupt:
        print("\nVision test stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
