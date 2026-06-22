#!/usr/bin/env bash
set -euo pipefail

VIDEO_DEVICE="/dev/video0"
SUBDEV_DEVICE="/dev/v4l-subdev0"
RAW_PATH="/tmp/case_v4l2_smoke.raw"
WIDTH=1640
HEIGHT=1232
EXPECTED_SIZE=4060672

if [[ ${1:-} == "--full" ]]; then
    WIDTH=3280
    HEIGHT=2464
    EXPECTED_SIZE=16163840
elif [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--full]" >&2
    exit 2
fi

if ! command -v v4l2-ctl >/dev/null 2>&1; then
    echo "ERROR: v4l2-ctl is not installed" >&2
    exit 1
fi

truncate -s 0 "$RAW_PATH"

echo "Setting IMX219 subdevice format to ${WIDTH}x${HEIGHT} SRGGB10_1X10"
v4l2-ctl -d "$SUBDEV_DEVICE" \
    --set-subdev-fmt "pad=0,width=${WIDTH},height=${HEIGHT},code=SRGGB10_1X10"

echo "Capturing one ${WIDTH}x${HEIGHT} RG10 frame"
v4l2-ctl -d "$VIDEO_DEVICE" \
    --set-fmt-video="width=${WIDTH},height=${HEIGHT},pixelformat=RG10" \
    --stream-mmap=1 \
    --stream-count=1 \
    --stream-to="$RAW_PATH"

RAW_SIZE=$(stat -c '%s' "$RAW_PATH")
echo "Captured $RAW_PATH (${RAW_SIZE} bytes; expected ${EXPECTED_SIZE} bytes)"

if [[ ! -s $RAW_PATH ]]; then
    echo "ERROR: capture file is empty" >&2
    exit 1
fi

if [[ $RAW_SIZE -ne $EXPECTED_SIZE ]]; then
    echo "WARNING: captured size differs from the confirmed CASE mode" >&2
fi

echo "V4L2 camera smoke test passed"
