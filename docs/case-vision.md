# CASE Vision / Optic Nerve Engineering Log

This log records the confirmed camera configuration and design decisions for CASE. Treat the values and commands below as the known-good baseline for future vision work.

## 1. Architecture Decision

CASE vision is split by responsibility:

```text
Pi Camera = perception and awareness
ESP32 Spine sensors = survival and collision avoidance
```

The Pi Camera is **not** responsible for emergency braking or last-millisecond obstacle avoidance. The ESP32 Spine and VL53L0X sensors own the safety-critical distance response:

```text
distance < 150 mm  -> HARD_STOP
150-300 mm         -> SLOW_DOWN
300-500 mm         -> CAUTION
> 500 mm           -> CLEAR
```

The Pi handles:

```text
face/user detection
scene snapshots
on-demand “what do you see?”
future low-frequency object detection
```

Do not use real-time YOLO navigation on the Raspberry Pi 4 for now.

## 2. Current Hardware / OS

```text
Board: Raspberry Pi 4
OS: Ubuntu Server 22.04
Camera: Raspberry Pi Camera V2
Sensor: IMX219
Kernel driver: unicam + imx219
Main video node: /dev/video0
Subdev control node: /dev/v4l-subdev0
```

Important notes:

```text
rpicam-jpeg is not available on this Ubuntu Server setup.
libcamera-tools `cam -l` segfaults.
Picamera2 is not the active backend.
The working backend is V4L2 raw capture + OpenCV.
```

## 3. Python Environment

Use only the normal project virtual environment:

```bash
cd ~/CASE
source venv/bin/activate
```

`venv-pi` was deleted to avoid confusion.

NumPy must remain below version 2. The confirmed working versions are:

```text
opencv-python-headless: 4.11.0
numpy: 1.26.4
```

Recommended requirements entries:

```text
numpy<2
opencv-python-headless
```

## 4. Working Camera Modes

Direct 640x480 V4L2 capture failed with:

```text
VIDIOC_STREAMON returned -1 (Invalid argument)
```

Confirmed working full-resolution mode:

```text
Subdev:
/dev/v4l-subdev0
width=3280 height=2464 code=SRGGB10_1X10

Video:
/dev/video0
width=3280 height=2464 pixelformat=RG10
bytesperline=6560
sizeimage=16163840
raw output about 16 MB
```

Confirmed working binned/lower mode:

```text
Subdev:
/dev/v4l-subdev0
width=1640 height=1232 code=SRGGB10_1X10

Video:
/dev/video0
width=1640 height=1232 pixelformat=RG10
bytesperline=3296
sizeimage=4060672
raw output about 3.9 MB
```

Default continuous CASE vision mode:

```text
capture: 1640x1232 RG10
process/output: resize to 640x480
FPS: 1
```

Use 3280x2464 only for high-quality snapshots and debugging.

## 5. Required V4L2 Capture Order

Set the sensor subdevice format before configuring or capturing from the video device.

For 1640x1232:

```bash
v4l2-ctl -d /dev/v4l-subdev0 \
  --set-subdev-fmt pad=0,width=1640,height=1232,code=SRGGB10_1X10

v4l2-ctl -d /dev/video0 \
  --set-fmt-video=width=1640,height=1232,pixelformat=RG10 \
  --stream-mmap=1 \
  --stream-count=1 \
  --stream-to=/tmp/case_vision_frame.raw
```

For full resolution:

```bash
v4l2-ctl -d /dev/v4l-subdev0 \
  --set-subdev-fmt pad=0,width=3280,height=2464,code=SRGGB10_1X10

v4l2-ctl -d /dev/video0 \
  --set-fmt-video=width=3280,height=2464,pixelformat=RG10 \
  --stream-mmap=1 \
  --stream-count=1 \
  --stream-to=/tmp/case_vision_frame.raw
```

## 6. Raw Parsing / Stride Rule

Do not blindly reshape raw data to `(height, width)` for every mode.

For 1640x1232:

```text
bytesperline = 3296
stride_pixels = 3296 / 2 = 1648
visible width = 1640
```

Correct parsing:

```python
raw = np.fromfile(raw_path, dtype=np.uint16)

stride_pixels = bytes_per_line // 2
raw = raw.reshape((height, stride_pixels))
raw = raw[:, :width]
```

For 3280x2464:

```text
bytesperline = 6560
stride_pixels = 3280
no crop needed
```

## 7. Known-Good Color Pipeline

The best-looking CASE image came from the old raw conversion script. Use this as the default color profile:

```text
Bayer: BG
OpenCV conversion: cv2.COLOR_BayerBG2BGR
Black level: 64
White level: 1023
Gray-world WB: ON
WB strength: 0.85
Gamma: 0.45
Manual WB: OFF
```

Processing order:

```text
raw uint16
-> reshape/crop using stride
-> float32
-> subtract black level
-> clip to 0..WHITE_LEVEL-BLACK_LEVEL
-> convert to uint8 raw8
-> cv2.cvtColor(raw8, cv2.COLOR_BayerBG2BGR)
-> gray_world_wb_partial
-> gamma correction
-> resize
-> save/detect
```

Reference values:

```python
BLACK_LEVEL = 64
WHITE_LEVEL = 1023
WB_STRENGTH = 0.85
GAMMA = 0.45
```

Do not default to aggressive manual white balance, and do not compensate for an incorrect Bayer conversion with extreme white-balance gains.

## 8. Brightness / Exposure Controls

Camera controls live on `/dev/v4l-subdev0`.

Good bright indoor preset:

```bash
v4l2-ctl -d /dev/v4l-subdev0 --set-ctrl=vertical_blanking=10000
v4l2-ctl -d /dev/v4l-subdev0 --set-ctrl=exposure=9000
v4l2-ctl -d /dev/v4l-subdev0 --set-ctrl=analogue_gain=120
v4l2-ctl -d /dev/v4l-subdev0 --set-ctrl=digital_gain=768
```

Softer preset to reduce overexposure:

```bash
v4l2-ctl -d /dev/v4l-subdev0 --set-ctrl=vertical_blanking=10000
v4l2-ctl -d /dev/v4l-subdev0 --set-ctrl=exposure=6000
v4l2-ctl -d /dev/v4l-subdev0 --set-ctrl=analogue_gain=80
v4l2-ctl -d /dev/v4l-subdev0 --set-ctrl=digital_gain=512
```

Apply capture settings in this order:

```text
set subdev format
-> set vertical_blanking
-> set exposure
-> set analogue_gain
-> set digital_gain
-> set video format
-> stream capture
```

## 9. CMA / Memory Notes

Full-resolution RG10 uses about 16 MB per frame. Previous failures included:

```text
dma_alloc_coherent failed
cma_alloc failed
```

These indicate CMA contiguous-memory pressure, not camera damage.

Mitigations:

```text
Use --stream-mmap=1 or --stream-mmap=2
Prefer 1640x1232 for CASE runtime
Use 3280x2464 only for snapshots
Optional boot fix: add cma=256M to /boot/firmware/cmdline.txt
```

## 10. Current Working Commands

High-quality snapshot:

```bash
cd ~/CASE
source venv/bin/activate

python3 scripts/test_vision.py --snapshot \
  --capture-width 3280 --capture-height 2464 \
  --save-full-processed \
  --legacy-color
```

Lightweight CASE snapshot:

```bash
cd ~/CASE
source venv/bin/activate

python3 scripts/test_vision.py --snapshot \
  --capture-width 1640 --capture-height 1232 \
  --width 640 --height 480 \
  --legacy-color
```

Debug output directories:

```text
output/vision_snapshots/
output/vision_debug/
```

## 11. Phase 1 Vision Goal

Next implementation target:

```text
V4L2 raw capture
-> CASE legacy color profile
-> resize to 640x480
-> OpenCV face detection
-> publish VISION_USER_DETECTED / VISION_USER_LOST
```

Message bus events:

```text
VISION_USER_DETECTED
VISION_USER_LOST
VISION_FRAME_READY
VISION_STATUS
VISION_ERROR
VISION_SCENE_SNAPSHOT_READY
```

## 12. Main App Integration Rules

Vision must run as a background asynchronous task.

```text
Do not block the voice pipeline.
Do not crash main.py if the camera fails.
Do not speak vision greetings while CASE is already speaking.
Do not interrupt an active conversation.
Only greet when idle and the cooldown has passed.
```

Suggested greeting:

```text
I see you, boss.
```

Cooldown:

```text
VISION_GREETING_COOLDOWN_SEC = 60
```

## 13. YOLO / Colab Plan

Do not train YOLO yet. First finish Phase 1:

```text
camera snapshot OK
face detection OK
vision events OK
main.py integration OK
```

YOLO training on Colab or Kaggle is Phase 2 or Phase 3, after selecting target classes and collecting images from the actual CASE camera.

Possible future classes:

```text
person
chair
cup
phone
door
hand
robot part
```

Train YOLO on Colab or Kaggle, not on the Raspberry Pi. The later deployment target is:

```text
ONNX / NCNN / TFLite exported model
low-frequency or on-demand inference on Pi
not real-time navigation
```

## 14. Current Status

Completed:

```text
V4L2 raw capture working
3280x2464 working
1640x1232 working
raw stride parsing fixed
legacy color profile restored
snapshot JPG looks good
Python venv fixed
numpy/cv2 import OK
```

Next:

```text
test 1640x1232 -> 640x480 legacy snapshot
test OpenCV face detection
publish VISION_USER_DETECTED
integrate VisionEngine into main.py
```
