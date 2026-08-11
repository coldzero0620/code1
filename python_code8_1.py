#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Version 8 CORE FINAL (single-file integration before modularization).
#
# Included in this version:
#   - CSI camera opened by one capture worker
#   - latest-frame overwrite buffer (no old-frame queue)
#   - 640x480 capture/display and 320x240 MediaPipe analysis
#   - camera automatic reopen and fatal-timeout reporting
#   - baseline sample-count/stability validation
#   - posture hysteresis and confirmed recovery
#   - BLE N/W/B/P/S, heartbeat, reconnect/resync and WARNING reminder
#   - shared runtime state prepared for later Flask integration
#   - SPACE pause/resume; pause always requests wristband motor OFF
#   - ESC opens a frozen summary and pauses posture monitoring
#   - ESC in summary returns to the live camera and resumes monitoring
#   - ENTER has no UI action; window X closes UI only
#   - posture detection, LED and BLE continue after the UI is closed
#   - wristband battery telemetry is received by BLE Notify and shown in UI
#   - BLE loss indicator: RED -> GREEN -> BLUE, one second per color
#   - GPIO23 slide-switch debounce and orderly Raspberry Pi poweroff request
#
# Intentionally deferred:
#   - wristband battery switch (implemented as a physical power cut-off)
#   - Flask/MJPEG server and Raspberry Pi access point

import asyncio
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import mediapipe as mp
import numpy as np

try:
    cv2.setNumThreads(1)
except cv2.error:
    pass

try:
    from bleak import BleakClient, BleakScanner

    BLE_AVAILABLE = True
except ImportError:
    BleakClient = BleakScanner = None
    BLE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Optional GPIO support
# ---------------------------------------------------------------------------
class DummyGPIO:
    BCM = "BCM"
    IN = "IN"
    OUT = "OUT"
    PUD_UP = "PUD_UP"
    LOW = 0
    HIGH = 1

    def setmode(self, mode):
        pass

    def setwarnings(self, enabled):
        pass

    def setup(self, pin, mode, **kwargs):
        pass

    def output(self, pin, state):
        pass

    def input(self, pin):
        return self.HIGH

    def cleanup(self):
        pass


try:
    import RPi.GPIO as GPIO

    GPIO_AVAILABLE = True
    print("[GPIO] Hardware mode")
except Exception as gpio_error:
    GPIO = DummyGPIO()
    GPIO_AVAILABLE = False
    print(f"[GPIO] Simulation mode: {gpio_error}")


# ---------------------------------------------------------------------------
# Raspberry Pi CSI camera stream
# Uses rpicam-vid because this Python 3.11 environment cannot use the
# Raspberry Pi OS Python 3.13 Picamera2 package directly.
# ---------------------------------------------------------------------------
class RpicamCapture:
    JPEG_START = b"\xff\xd8"
    JPEG_END = b"\xff\xd9"

    def __init__(self, width=640, height=480, fps=15, camera_index=0):
        rpicam_path = shutil.which("rpicam-vid")
        if rpicam_path is None:
            raise RuntimeError("rpicam-vid command was not found.")

        command = [
            rpicam_path,
            "--nopreview",
            "--timeout",
            "0",
            "--camera",
            str(camera_index),
            "--width",
            str(width),
            "--height",
            str(height),
            "--framerate",
            str(fps),
            "--codec",
            "mjpeg",
            "--quality",
            "70",
            "--output",
            "-",
        ]

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self.buffer = bytearray()
        self._release_lock = threading.Lock()

        time.sleep(0.3)
        if self.process.poll() is not None:
            raise RuntimeError(
                "rpicam-vid could not start. Close other camera programs first."
            )

    def isOpened(self):
        return (
            self.process is not None
            and self.process.poll() is None
            and self.process.stdout is not None
        )

    def read(self):
        while self.isOpened():
            start = self.buffer.find(self.JPEG_START)
            if start >= 0:
                end = self.buffer.find(self.JPEG_END, start + 2)
                if end >= 0:
                    jpeg_data = bytes(self.buffer[start : end + 2])
                    del self.buffer[: end + 2]

                    frame = cv2.imdecode(
                        np.frombuffer(jpeg_data, dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if frame is not None:
                        return True, frame

            process = self.process
            if process is None or process.stdout is None:
                break

            try:
                data = process.stdout.read(4096)
            except (OSError, ValueError):
                break
            if not data:
                break
            self.buffer.extend(data)

            if len(self.buffer) > 8_000_000:
                last_start = self.buffer.rfind(self.JPEG_START)
                if last_start >= 0:
                    del self.buffer[:last_start]
                else:
                    self.buffer.clear()

        return False, None

    def release(self):
        with self._release_lock:
            process = self.process
            self.process = None

        if process is None:
            return

        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

        if process.stdout is not None:
            process.stdout.close()


# ---------------------------------------------------------------------------
# Hardware pins and posture parameters
# ---------------------------------------------------------------------------
# Actual RGB channel mapping confirmed from the physical LED:
# GPIO17 -> BLUE, GPIO27 -> GREEN, GPIO22 -> RED
LED_R = 22
LED_G = 27
LED_B = 17

# Three-pin SPDT slide switch used as a two-wire, active-LOW shutdown request:
#   center pin -> BCM GPIO23 (physical pin 16)
#   one outer pin -> GND
#   unused outer pin -> not connected
# The internal pull-up keeps the input HIGH while the switch is open.
SHUTDOWN_SWITCH_PIN = 23
SHUTDOWN_SWITCH_DEBOUNCE_SECONDS = 0.75
SHUTDOWN_SWITCH_POLL_SECONDS = 0.05
SHUTDOWN_SWITCH_STARTUP_GRACE_SECONDS = 2.0
POWEROFF_COMMAND_TIMEOUT_SECONDS = 10.0

# Keep capture/display resolution separate from the MediaPipe input.
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
ANALYSIS_WIDTH = 320
ANALYSIS_HEIGHT = 240
CAMERA_FPS = 15

# Enter and exit thresholds are deliberately different. This prevents a
# single noisy frame around the boundary from rapidly toggling the state.
MARGIN_ENTER = 0.10
MARGIN_EXIT = 0.07
BAD_POSTURE_LIMIT = 2.0
NORMAL_CONFIRM_SECONDS = 0.40
VISIBILITY_THRESHOLD = 0.50
CALIBRATION_SECONDS = 3.0
CALIBRATION_RETRY_SECONDS = 2.0
CALIBRATION_MIN_SAMPLES = 12
CALIBRATION_MAX_MAD = 0.04

CAMERA_REOPEN_SECONDS = 1.0
CAMERA_WAIT_SECONDS = 0.75
CAMERA_FATAL_TIMEOUT_SECONDS = 20.0
ESC_KEY = 27


class RuntimeState:
    """Thread-safe state prepared for the later Flask/status API."""

    def __init__(self):
        self._lock = threading.Lock()
        self.paused = False
        self.status = "NO_POSE"
        self.score = None
        self.bad_timer = 0.0
        self.total_bad_time = 0.0
        self.total_warning_time = 0.0
        self.good_ratio = 100.0
        self.side = "-"
        self.model_source = "RULE"
        self.ble_connected = False
        self.battery_percent = None
        self.battery_voltage = None
        self.camera_ok = False
        self.updated_at = time.monotonic()

    def update(self, **values):
        with self._lock:
            for name, value in values.items():
                if not hasattr(self, name):
                    raise AttributeError(f"Unknown runtime state: {name}")
                setattr(self, name, value)
            self.updated_at = time.monotonic()

    def toggle_paused(self):
        with self._lock:
            self.paused = not self.paused
            self.updated_at = time.monotonic()
            return self.paused

    def set_paused(self, paused):
        paused = bool(paused)
        with self._lock:
            changed = self.paused != paused
            self.paused = paused
            self.updated_at = time.monotonic()
            return changed

    def is_paused(self):
        with self._lock:
            return self.paused

    def is_ble_connected(self):
        with self._lock:
            return self.ble_connected

    def snapshot(self):
        with self._lock:
            return {
                "paused": self.paused,
                "status": self.status,
                "score": self.score,
                "bad_timer": self.bad_timer,
                "total_bad_time": self.total_bad_time,
                "total_warning_time": self.total_warning_time,
                "good_ratio": self.good_ratio,
                "side": self.side,
                "model_source": self.model_source,
                "ble_connected": self.ble_connected,
                "battery_percent": self.battery_percent,
                "battery_voltage": self.battery_voltage,
                "camera_ok": self.camera_ok,
                "updated_at": self.updated_at,
            }


class LatestFrameCamera:
    """Owns the CSI camera and exposes only the newest decoded frame."""

    def __init__(self, runtime_state):
        self._runtime_state = runtime_state
        self._condition = threading.Condition()
        self._latest_frame = None
        self._frame_id = 0
        self._last_frame_time = None
        self._started_at = time.monotonic()
        self._last_error = None
        self._active_lock = threading.Lock()
        self._active_capture = None
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return

        self._thread = threading.Thread(
            target=self._capture_loop,
            name="csi-camera-capture",
            daemon=True,
        )
        self._thread.start()

    def get_latest(self, previous_id=-1, timeout=CAMERA_WAIT_SECONDS):
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._frame_id != previous_id
                    or self._stop_event.is_set()
                ),
                timeout=timeout,
            )

            if self._latest_frame is None or self._frame_id == previous_id:
                return previous_id, None, None

            return (
                self._frame_id,
                self._latest_frame.copy(),
                self._last_frame_time,
            )

    def seconds_since_frame(self):
        with self._condition:
            reference = self._last_frame_time
        if reference is None:
            reference = self._started_at
        return time.monotonic() - reference

    def stop(self):
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()

        with self._active_lock:
            capture = self._active_capture
        if capture is not None:
            capture.release()

        if self._thread is not None:
            self._thread.join(timeout=4.0)

    def _capture_loop(self):
        while not self._stop_event.is_set():
            capture = None
            try:
                print("[CAMERA] Opening CSI camera...")
                capture = RpicamCapture(
                    width=CAPTURE_WIDTH,
                    height=CAPTURE_HEIGHT,
                    fps=CAMERA_FPS,
                    camera_index=0,
                )
                with self._active_lock:
                    self._active_capture = capture

                self._last_error = None
                print("[CAMERA] Capture worker ready.")

                while not self._stop_event.is_set():
                    success, frame = capture.read()
                    if not success or frame is None:
                        raise RuntimeError("CSI camera stream stopped.")

                    if (
                        frame.shape[1] != CAPTURE_WIDTH
                        or frame.shape[0] != CAPTURE_HEIGHT
                    ):
                        frame = cv2.resize(
                            frame,
                            (CAPTURE_WIDTH, CAPTURE_HEIGHT),
                            interpolation=cv2.INTER_AREA,
                        )

                    with self._condition:
                        self._latest_frame = frame
                        self._frame_id += 1
                        self._last_frame_time = time.monotonic()
                        self._condition.notify_all()
                    self._runtime_state.update(camera_ok=True)

            except Exception as camera_error:
                self._runtime_state.update(camera_ok=False)
                message = str(camera_error)
                if (
                    not self._stop_event.is_set()
                    and message != self._last_error
                ):
                    print(f"[CAMERA] {message}")
                    print("[CAMERA] Reopening automatically...")
                    self._last_error = message
            finally:
                with self._active_lock:
                    self._active_capture = None
                if capture is not None:
                    capture.release()

            if not self._stop_event.wait(CAMERA_REOPEN_SECONDS):
                continue
            break

# No monitor/VNC is required.  When DISPLAY is unavailable the program
# automatically runs in headless mode.  --headless can also force this mode.
HEADLESS = "--headless" in sys.argv or not os.environ.get("DISPLAY")

LED_BLE_LOST_COLORS = ("RED", "GREEN", "BLUE")
LED_BLE_LOST_STEP_SECONDS = 1.0
LED_WORKER_POLL_SECONDS = 0.05
LED_ALLOWED_COLORS = frozenset(
    {"OFF", "RED", "GREEN", "BLUE", "WHITE"}
)

current_led_status = "OFF"
requested_led_status = "OFF"
_led_lock = threading.Lock()

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
for output_pin in (LED_R, LED_G, LED_B):
    GPIO.setup(output_pin, GPIO.OUT)
    GPIO.output(output_pin, GPIO.LOW)


def set_led(color):
    """Store the posture LED color requested by the monitoring loop."""
    global requested_led_status

    color = str(color).strip().upper()
    if color not in LED_ALLOWED_COLORS:
        raise ValueError(f"Unsupported LED color: {color}")

    with _led_lock:
        requested_led_status = color


def get_requested_led():
    with _led_lock:
        return requested_led_status


def drive_led_hardware(color):
    """Write GPIO only when the physical LED color actually changes."""
    global current_led_status

    with _led_lock:
        if current_led_status == color:
            return

        current_led_status = color
        GPIO.output(LED_R, GPIO.LOW)
        GPIO.output(LED_G, GPIO.LOW)
        GPIO.output(LED_B, GPIO.LOW)

        if color == "GREEN":
            GPIO.output(LED_G, GPIO.HIGH)
        elif color == "BLUE":
            GPIO.output(LED_B, GPIO.HIGH)
        elif color == "RED":
            GPIO.output(LED_R, GPIO.HIGH)
        elif color == "WHITE":
            # NO_POSE: turn on all three RGB channels.
            GPIO.output(LED_R, GPIO.HIGH)
            GPIO.output(LED_G, GPIO.HIGH)
            GPIO.output(LED_B, GPIO.HIGH)


class LedStatusWorker:
    """Shows posture color, or an RGB cycle while BLE is disconnected."""

    def __init__(self, runtime_state):
        self._runtime_state = runtime_state
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return

        self._thread = threading.Thread(
            target=self._loop,
            name="rgb-led-status",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        drive_led_hardware("OFF")

    def _loop(self):
        disconnected_since = None

        while not self._stop_event.is_set():
            now = time.monotonic()
            if self._runtime_state.is_ble_connected():
                disconnected_since = None
                target_color = get_requested_led()
            else:
                if disconnected_since is None:
                    disconnected_since = now
                step = int(
                    (now - disconnected_since)
                    / LED_BLE_LOST_STEP_SECONDS
                )
                target_color = LED_BLE_LOST_COLORS[
                    step % len(LED_BLE_LOST_COLORS)
                ]

            drive_led_hardware(target_color)
            self._stop_event.wait(LED_WORKER_POLL_SECONDS)


# ---------------------------------------------------------------------------
# BLE posture-band client
# XIAO ESP32-C3 advertises as "Posture-Band".
# Commands:
#   H = heartbeat (must never retrigger vibration)
#   N = NORMAL / motor OFF
#   W = WARNING / two short pulses (handled by XIAO firmware)
#   B = BAD / continuous vibration
#   P = NO_POSE / motor OFF
#   S = PAUSED or SHUTDOWN / motor OFF
# The wristband firmware, not Raspberry Pi GPIO, controls the vibration motor.
# P and S are preceded by N so pause/stop also remains safe with the older
# N/W/B-only firmware. Predictable motor-off after sudden Pi power loss still
# requires the XIAO firmware to implement H timeout and disconnect fail-safe.
# ---------------------------------------------------------------------------
BLE_DEVICE_NAME = "Posture-Band"
BLE_CHARACTERISTIC_UUID = "abcdefab-1234-5678-1234-abcdefabcdef"
BLE_BATTERY_CHARACTERISTIC_UUID = "abcdefac-1234-5678-1234-abcdefabcdef"
BLE_SCAN_TIMEOUT_SECONDS = 5.0
BLE_RETRY_SECONDS = 1.0
BLE_HEARTBEAT_SECONDS = 1.0
BLE_WARNING_REPEAT_SECONDS = 1.0
BLE_ALLOWED_STATES = frozenset({"N", "W", "B", "P", "S"})


class BlePostureBand:
    def __init__(self, runtime_state):
        self._runtime_state = runtime_state
        self._desired_state = "P"
        self._state_lock = threading.Lock()
        self._sent_condition = threading.Condition(self._state_lock)
        self._last_confirmed_state = None
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if not BLE_AVAILABLE:
            print("[BLE] bleak is not installed. Wristband is disabled.")
            self._runtime_state.update(ble_connected=False)
            return

        self._thread = threading.Thread(
            target=self._thread_main,
            name="posture-band-ble",
            daemon=True,
        )
        self._thread.start()

    def set_state(self, state):
        state = str(state).strip().upper()
        if state not in BLE_ALLOWED_STATES:
            raise ValueError(f"Unsupported BLE posture state: {state}")

        with self._sent_condition:
            self._desired_state = state
            self._sent_condition.notify_all()

    def stop(self):
        # S is sent as N followed by S, so both old and extended firmware turn
        # the motor off. Wait briefly for the acknowledged GATT write.
        self.set_state("S")
        if self._thread is None:
            self._runtime_state.update(ble_connected=False)
            return

        deadline = time.monotonic() + 1.0
        with self._sent_condition:
            while (
                self._last_confirmed_state != "S"
                and time.monotonic() < deadline
            ):
                self._sent_condition.wait(timeout=0.1)

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=6.0)
        self._runtime_state.update(ble_connected=False)

    def _handle_battery_notification(self, sender, data):
        try:
            payload = bytes(data).decode("utf-8").strip()
            parts = payload.split(",")
            if len(parts) != 3 or parts[0].upper() != "BAT":
                print(f"[BLE] Ignored battery payload: {payload!r}")
                return

            percent = max(0, min(100, int(parts[1])))
            voltage = float(parts[2])
            self._runtime_state.update(
                battery_percent=percent,
                battery_voltage=voltage,
            )
            print(f"[BLE] Battery: {percent}% ({voltage:.2f}V)")
        except (UnicodeDecodeError, ValueError) as error:
            print(f"[BLE] Invalid battery notification: {error}")

    def _thread_main(self):
        while not self._stop_event.is_set():
            try:
                asyncio.run(self._ble_loop())
            except Exception as error:
                print(f"[BLE] Worker error: {error}")
                self._runtime_state.update(ble_connected=False)

            if self._stop_event.wait(BLE_RETRY_SECONDS):
                break

    async def _write_command(self, client, command):
        await client.write_gatt_char(
            BLE_CHARACTERISTIC_UUID,
            command.encode("utf-8"),
            response=True,
        )

    async def _write_state_command(self, client, state):
        if state in {"P", "S"}:
            await self._write_command(client, "N")
        await self._write_command(client, state)

    async def _ble_loop(self):
        while not self._stop_event.is_set():
            try:
                print(f"[BLE] Searching for {BLE_DEVICE_NAME}...")
                device = await BleakScanner.find_device_by_name(
                    BLE_DEVICE_NAME,
                    timeout=BLE_SCAN_TIMEOUT_SECONDS,
                )
                if device is None:
                    print(
                        f"[BLE] {BLE_DEVICE_NAME} not found. "
                        "Keep the XIAO very close to the Raspberry Pi."
                    )
                    await asyncio.sleep(BLE_RETRY_SECONDS)
                    continue

                async with BleakClient(
                    device,
                    timeout=15.0,
                ) as client:
                    print(f"[BLE] Connected: {BLE_DEVICE_NAME}")
                    self._runtime_state.update(ble_connected=True)

                    await client.start_notify(
                        BLE_BATTERY_CHARACTERISTIC_UUID,
                        self._handle_battery_notification,
                    )
                    try:
                        initial_battery = await client.read_gatt_char(
                            BLE_BATTERY_CHARACTERISTIC_UUID
                        )
                        self._handle_battery_notification(
                            BLE_BATTERY_CHARACTERISTIC_UUID,
                            initial_battery,
                        )
                    except Exception as battery_read_error:
                        print(
                            f"[BLE] Initial battery read failed: "
                            f"{battery_read_error}"
                        )

                    last_sent_state = None
                    last_heartbeat_time = 0.0
                    last_warning_time = 0.0

                    while (
                        client.is_connected
                        and not self._stop_event.is_set()
                    ):
                        with self._state_lock:
                            state = self._desired_state

                        now = time.monotonic()
                        if state != last_sent_state:
                            await self._write_state_command(client, state)
                            print(f"[BLE] State sent: {state}")
                            last_sent_state = state
                            if state == "W":
                                last_warning_time = now

                            with self._sent_condition:
                                self._last_confirmed_state = state
                                self._sent_condition.notify_all()

                        elif (
                            state == "W"
                            and now - last_warning_time
                            >= BLE_WARNING_REPEAT_SECONDS
                        ):
                            await self._write_command(client, "W")
                            print("[BLE] WARNING reminder sent: W")
                            last_warning_time = now

                        if (
                            now - last_heartbeat_time
                            >= BLE_HEARTBEAT_SECONDS
                        ):
                            await self._write_command(client, "H")
                            last_heartbeat_time = now

                        await asyncio.sleep(0.1)

                self._runtime_state.update(
                    ble_connected=False,
                    battery_percent=None,
                    battery_voltage=None,
                )

            except Exception as error:
                self._runtime_state.update(
                    ble_connected=False,
                    battery_percent=None,
                    battery_voltage=None,
                )
                if not self._stop_event.is_set():
                    print(f"[BLE] Connection lost: {error}")
                    await asyncio.sleep(BLE_RETRY_SECONDS)

        self._runtime_state.update(
            ble_connected=False,
            battery_percent=None,
            battery_voltage=None,
        )


# ---------------------------------------------------------------------------
# Optional Random Forest model
#
# Supported model inputs:
#   1 feature: posture_error
#   2 features: normalized_delta, posture_error
#
# If no compatible model exists, the calibrated threshold rule is used.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = (
    BASE_DIR / "models" / "posture-rf.joblib",
    BASE_DIR / "posture-rf.joblib",
)
MODEL_PATH = next(
    (path for path in MODEL_CANDIDATES if path.exists()),
    MODEL_CANDIDATES[0],
)
rf_model = None
model_warning_shown = False

try:
    if MODEL_PATH.exists():
        import joblib

        rf_model = joblib.load(MODEL_PATH)
        print(f"[MODEL] Loaded: {MODEL_PATH.name}")
    else:
        print("[MODEL] posture-rf.joblib not found. Using threshold rule.")
except Exception as model_error:
    rf_model = None
    print(f"[MODEL] Could not load model. Using threshold rule: {model_error}")


def prediction_to_severity(prediction):
    if isinstance(prediction, (bool, np.bool_)):
        return "WARNING" if bool(prediction) else "NORMAL"

    if isinstance(prediction, (int, float, np.integer, np.floating)):
        numeric_label = int(prediction)
        return {
            0: "NORMAL",
            1: "WARNING",
            2: "BAD",
        }.get(numeric_label)

    label = str(prediction).strip().lower()
    if label in {"0", "normal", "good"}:
        return "NORMAL"
    if label in {
        "1",
        "warning",
        "forward_head",
        "forward-head",
        "poor",
    }:
        return "WARNING"
    if label in {"2", "bad", "severe"}:
        return "BAD"
    return None


def classify_posture(normalized_delta, posture_error):
    global model_warning_shown

    rule_result = (
        "WARNING"
        if posture_error >= MARGIN_ENTER
        else "NORMAL"
    )
    if rf_model is None:
        return rule_result, "RULE"

    try:
        feature_count = int(getattr(rf_model, "n_features_in_", 0))
        expected_names = {
            1: ("posture_error",),
            2: ("normalized_delta", "posture_error"),
        }.get(feature_count)

        if expected_names is None:
            if not model_warning_shown:
                print(
                    "[MODEL] Unsupported feature count. "
                    "Using threshold rule instead."
                )
                model_warning_shown = True
            return rule_result, "RULE"

        actual_names = getattr(rf_model, "feature_names_in_", None)
        if actual_names is not None:
            actual_names = tuple(str(name) for name in actual_names)
            if actual_names != expected_names:
                if not model_warning_shown:
                    print(
                        f"[MODEL] Feature order mismatch: {actual_names}. "
                        f"Expected {expected_names}. Using threshold rule."
                    )
                    model_warning_shown = True
                return rule_result, "RULE"

        if feature_count == 1:
            features = np.array([[posture_error]], dtype=np.float32)
        else:
            features = np.array(
                [[normalized_delta, posture_error]],
                dtype=np.float32,
            )

        prediction = rf_model.predict(features)[0]
        severity = prediction_to_severity(prediction)
        if severity is None:
            raise ValueError(f"Unsupported RF label: {prediction!r}")
        return severity, "RF"
    except Exception as prediction_error:
        if not model_warning_shown:
            print(
                f"[MODEL] Prediction failed. Using threshold rule: "
                f"{prediction_error}"
            )
            model_warning_shown = True
        return rule_result, "RULE"


# ---------------------------------------------------------------------------
# MediaPipe Pose helpers
# ---------------------------------------------------------------------------
mp_pose = mp.solutions.pose


def get_ear_shoulder_data(results, frame_shape, preferred_side=None):
    if results.pose_landmarks is None:
        return None, None, None, None

    landmarks = results.pose_landmarks.landmark
    candidates = [
        (
            "LEFT",
            landmarks[mp_pose.PoseLandmark.LEFT_EAR.value],
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
        ),
        (
            "RIGHT",
            landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        ),
    ]

    if preferred_side in {"LEFT", "RIGHT"}:
        side, ear_landmark, shoulder_landmark = next(
            item for item in candidates if item[0] == preferred_side
        )
    else:
        side, ear_landmark, shoulder_landmark = max(
            candidates,
            key=lambda item: min(item[1].visibility, item[2].visibility),
        )

    if (
        ear_landmark.visibility < VISIBILITY_THRESHOLD
        or shoulder_landmark.visibility < VISIBILITY_THRESHOLD
    ):
        return None, None, None, None

    height, width, _ = frame_shape
    ear_point = (
        int(ear_landmark.x * width),
        int(ear_landmark.y * height),
    )
    shoulder_point = (
        int(shoulder_landmark.x * width),
        int(shoulder_landmark.y * height),
    )

    x_delta = abs(ear_landmark.x - shoulder_landmark.x)
    y_height = max(abs(ear_landmark.y - shoulder_landmark.y), 0.001)
    normalized_delta = x_delta / y_height

    return normalized_delta, ear_point, shoulder_point, side


class ServiceStopRequested(Exception):
    """Raised after the systemd stop event is observed by the main thread."""


class CalibrationError(Exception):
    """The camera works, but baseline samples are insufficient or unstable."""


PROGRAM_STOP_EVENT = threading.Event()
POWER_OFF_REQUESTED_EVENT = threading.Event()


def handle_service_stop(signum, frame):
    # Signal handlers only set a flag. Main and calibration loops notice it
    # within their short frame-wait timeout and perform deterministic cleanup.
    PROGRAM_STOP_EVENT.set()


class ShutdownSwitchMonitor:
    """Debounce an active-LOW GPIO switch and request orderly OS shutdown."""

    def __init__(self, pin=SHUTDOWN_SWITCH_PIN):
        self._pin = pin
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if not GPIO_AVAILABLE:
            print("[POWER] Shutdown switch disabled in GPIO simulation mode.")
            return
        if self._thread is not None and self._thread.is_alive():
            return

        GPIO.setup(self._pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        self._thread = threading.Thread(
            target=self._loop,
            name="shutdown-switch",
            daemon=True,
        )
        self._thread.start()
        print(
            f"[POWER] Shutdown switch armed on BCM GPIO{self._pin} "
            "(active LOW)."
        )

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self):
        if self._stop_event.wait(SHUTDOWN_SWITCH_STARTUP_GRACE_SECONDS):
            return

        active_since = None

        while not self._stop_event.is_set():
            try:
                switch_is_active = GPIO.input(self._pin) == GPIO.LOW
            except Exception as switch_error:
                print(f"[POWER] Shutdown switch read failed: {switch_error}")
                return

            now = time.monotonic()
            if switch_is_active:
                if active_since is None:
                    active_since = now
                elif (
                    now - active_since
                    >= SHUTDOWN_SWITCH_DEBOUNCE_SECONDS
                ):
                    print(
                        "[POWER] Shutdown switch activated. "
                        "Stopping BLE, camera and GPIO before poweroff."
                    )
                    POWER_OFF_REQUESTED_EVENT.set()
                    PROGRAM_STOP_EVENT.set()
                    return
            else:
                active_since = None

            self._stop_event.wait(SHUTDOWN_SWITCH_POLL_SECONDS)


def request_os_poweroff():
    """Ask systemd to power off after application cleanup has completed."""

    systemctl_path = shutil.which("systemctl") or "/usr/bin/systemctl"

    if hasattr(os, "geteuid") and os.geteuid() == 0:
        command = [systemctl_path, "poweroff"]
    else:
        sudo_path = shutil.which("sudo")
        if sudo_path is None:
            print(
                "[POWER] sudo was not found. The program closed safely, "
                "but the OS was not powered off."
            )
            return False
        command = [sudo_path, "-n", systemctl_path, "poweroff"]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=POWEROFF_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as poweroff_error:
        print(f"[POWER] Poweroff command failed: {poweroff_error}")
        return False

    if result.returncode == 0:
        print("[POWER] Raspberry Pi poweroff requested successfully.")
        return True

    error_text = (result.stderr or result.stdout or "unknown error").strip()
    print(
        "[POWER] Safe application shutdown completed, but OS poweroff "
        f"was denied: {error_text}"
    )
    return False


def window_was_closed(window_name):
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def close_ui_window(window_name):
    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass


def process_pose_frame(pose, frame):
    analysis_frame = cv2.resize(
        frame,
        (ANALYSIS_WIDTH, ANALYSIS_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    rgb_frame = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    return pose.process(rgb_frame)


def calibrate_baseline(camera, pose, window_name, ui_visible):
    print("=== Personal baseline calibration ===")
    print("Keep a comfortable upright posture for 3 seconds.")

    samples = []
    previous_frame_id = -1
    calibration_start = None

    while True:
        if PROGRAM_STOP_EVENT.is_set():
            raise ServiceStopRequested

        frame_id, frame, _ = camera.get_latest(previous_frame_id)
        if frame is None:
            if camera.seconds_since_frame() >= CAMERA_FATAL_TIMEOUT_SECONDS:
                raise RuntimeError(
                    "Camera produced no frame during calibration."
                )
            continue
        previous_frame_id = frame_id

        results = process_pose_frame(pose, frame)

        normalized_delta, ear_point, shoulder_point, side = (
            get_ear_shoulder_data(results, frame.shape)
        )

        if normalized_delta is not None:
            if calibration_start is None:
                calibration_start = time.monotonic()
            samples.append((normalized_delta, side))

            if results.pose_landmarks is not None:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )
            cv2.line(frame, ear_point, shoulder_point, (255, 255, 0), 2)
            cv2.circle(frame, ear_point, 6, (0, 0, 255), -1)
            cv2.circle(frame, shoulder_point, 6, (0, 255, 0), -1)
            cv2.putText(
                frame,
                f"TRACKING: {side}",
                (40, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

        if calibration_start is None:
            remaining = CALIBRATION_SECONDS
            calibration_message = "Waiting for a stable side pose"
        else:
            remaining = max(
                0.0,
                CALIBRATION_SECONDS
                - (time.monotonic() - calibration_start),
            )
            calibration_message = "Keep an upright posture"

        cv2.putText(
            frame,
            f"CALIBRATING: {remaining:.1f}s",
            (40, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            calibration_message,
            (40, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        if ui_visible:
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ESC_KEY or window_was_closed(window_name):
                ui_visible = False
                close_ui_window(window_name)
                print(
                    "[UI] Calibration window closed. "
                    "Calibration continues in the background."
                )

        if (
            calibration_start is not None
            and time.monotonic() - calibration_start
            >= CALIBRATION_SECONDS
        ):
            break

    if len(samples) < CALIBRATION_MIN_SAMPLES:
        raise CalibrationError(
            f"Only {len(samples)} valid baseline samples were collected; "
            f"at least {CALIBRATION_MIN_SAMPLES} are required."
        )

    preferred_side = Counter(side for _, side in samples).most_common(1)[0][0]
    side_samples = np.array(
        [value for value, side in samples if side == preferred_side],
        dtype=np.float32,
    )
    if len(side_samples) < CALIBRATION_MIN_SAMPLES:
        raise CalibrationError(
            "Tracking side changed too often during calibration. "
            "Keep the same side toward the camera."
        )

    baseline = float(np.median(side_samples))
    median_absolute_deviation = float(
        np.median(np.abs(side_samples - baseline))
    )
    if median_absolute_deviation > CALIBRATION_MAX_MAD:
        raise CalibrationError(
            "Baseline was unstable. Keep still and recalibrate."
        )

    print(
        f"[CALIBRATION] Baseline: {baseline:.4f}, "
        f"side: {preferred_side}, samples: {len(side_samples)}, "
        f"MAD: {median_absolute_deviation:.4f}"
    )
    return baseline, preferred_side, ui_visible


def build_summary_frame(
    total_monitored_time,
    total_warning_time,
    total_bad_time,
    average_score,
    monitoring_paused,
):
    if total_monitored_time > 0:
        good_ratio = max(
            0.0,
            (
                (
                    total_monitored_time
                    - total_warning_time
                    - total_bad_time
                )
                / total_monitored_time
            )
            * 100.0,
        )
    else:
        good_ratio = 100.0

    summary = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.rectangle(summary, (80, 55), (720, 545), (35, 35, 35), -1)
    cv2.rectangle(summary, (80, 55), (720, 545), (0, 215, 255), 2)

    monitor_text = "PAUSED" if monitoring_paused else "ACTIVE"
    monitor_color = (
        (0, 215, 255) if monitoring_paused else (100, 255, 100)
    )

    lines = [
        ("POSTURE ANALYSIS REPORT", (165, 110), (0, 215, 255)),
        (
            f"Total tracked time : {total_monitored_time:.1f} sec",
            (135, 175),
            (255, 255, 255),
        ),
        (
            f"Total warning time : {total_warning_time:.1f} sec",
            (135, 225),
            (255, 190, 120),
        ),
        (
            f"Total bad time     : {total_bad_time:.1f} sec",
            (135, 275),
            (150, 150, 255),
        ),
        (
            f"Average score      : {average_score:.1f} / 100",
            (135, 325),
            (255, 215, 0),
        ),
        (
            f"Good posture rate  : {good_ratio:.1f} %",
            (135, 375),
            (100, 255, 100),
        ),
        (
            f"POSTURE MONITORING: {monitor_text}",
            (135, 435),
            monitor_color,
        ),
        (
            "ESC: Return to live camera",
            (205, 500),
            (200, 200, 200),
        ),
    ]

    for text, position, color in lines:
        cv2.putText(
            summary,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68 if position[1] < 435 else 0.58,
            color,
            2 if position[1] < 435 else 1,
        )

    return summary


STATUS_COLORS = {
    "NO_POSE": (255, 255, 255),
    "NORMAL": (0, 255, 0),
    "WARNING": (255, 0, 0),
    "BAD": (0, 0, 255),
    "PAUSED": (0, 215, 255),
}


def current_led_bgr():
    with _led_lock:
        color = current_led_status
    return {
        "RED": (0, 0, 255),
        "BLUE": (255, 0, 0),
        "WHITE": (255, 255, 255),
        "GREEN": (0, 255, 0),
    }.get(color, (0, 0, 0))


def draw_monitor_overlay(frame, snapshot):
    status = snapshot["status"]
    status_color = STATUS_COLORS.get(status, (255, 255, 255))
    score_text = (
        f"{snapshot['score']} / 100"
        if snapshot["score"] is not None
        else "--"
    )
    ble_text = "CONNECTED" if snapshot["ble_connected"] else "SEARCHING"
    if (
        snapshot["battery_percent"] is not None
        and snapshot["battery_voltage"] is not None
    ):
        battery_text = (
            f"BATTERY: {snapshot['battery_percent']}% "
            f"({snapshot['battery_voltage']:.2f}V)"
        )
    else:
        battery_text = "BATTERY: --"

    rows = (
        (f"STATUS : {status}", 40, status_color, 0.72, 2),
        (f"SCORE  : {score_text}", 75, (255, 215, 0), 0.62, 2),
        (
            f"TIMER  : {snapshot['bad_timer']:.1f}s",
            108,
            (255, 255, 255),
            0.57,
            1,
        ),
        (
            f"TOTAL BAD : {snapshot['total_bad_time']:.1f}s",
            138,
            (180, 180, 255),
            0.57,
            1,
        ),
        (
            f"GOOD RATIO: {snapshot['good_ratio']:.1f}%",
            168,
            (180, 255, 180),
            0.57,
            2,
        ),
        (
            f"SIDE: {snapshot['side']}  MODE: {snapshot['model_source']}",
            198,
            (220, 220, 220),
            0.52,
            1,
        ),
        (f"BLE: {ble_text}", 228, (220, 220, 220), 0.52, 1),
        (battery_text, 258, (220, 220, 220), 0.52, 1),
    )

    for text_value, y, color, scale, thickness in rows:
        cv2.putText(
            frame,
            text_value,
            (25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
        )

    cv2.putText(
        frame,
        "SPACE: Pause/Resume    ESC: Summary",
        (25, frame.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (235, 235, 235),
        1,
    )
    cv2.circle(
        frame,
        (frame.shape[1] - 35, 35),
        15,
        current_led_bgr(),
        -1,
    )


def main():
    signal.signal(signal.SIGTERM, handle_service_stop)
    PROGRAM_STOP_EVENT.clear()
    POWER_OFF_REQUESTED_EVENT.clear()

    window_name = "Posture Monitor"
    runtime_state = RuntimeState()
    camera = LatestFrameCamera(runtime_state)
    posture_band = BlePostureBand(runtime_state)
    led_status = LedStatusWorker(runtime_state)
    shutdown_switch = ShutdownSwitchMonitor()
    pose = None
    exit_code = 0

    total_monitored_time = 0.0
    total_warning_time = 0.0
    total_bad_time = 0.0
    score_sum = 0.0
    score_count = 0
    ui_visible = not HEADLESS
    ui_mode = "LIVE"

    try:
        shutdown_switch.start()
        led_status.start()
        posture_band.start()
        camera.start()
        print(
            f"[DISPLAY] "
            f"{'Headless/offline mode' if not ui_visible else 'UI mode'}"
        )
        print(
            f"[CAMERA] Capture {CAPTURE_WIDTH}x{CAPTURE_HEIGHT} @ "
            f"{CAMERA_FPS} FPS; analysis "
            f"{ANALYSIS_WIDTH}x{ANALYSIS_HEIGHT}"
        )

        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        set_led("BLUE")
        posture_band.set_state("P")
        while not PROGRAM_STOP_EVENT.is_set():
            try:
                baseline, preferred_side, ui_visible = calibrate_baseline(
                    camera,
                    pose,
                    window_name,
                    ui_visible,
                )
                set_led("GREEN")
                posture_band.set_state("N")
                break
            except CalibrationError as calibration_error:
                print(
                    f"[CALIBRATION] {calibration_error} "
                    f"Retrying in {CALIBRATION_RETRY_SECONDS:.0f} seconds."
                )
                runtime_state.update(
                    status="NO_POSE",
                    score=None,
                    side="-",
                )
                posture_band.set_state("P")
                set_led("BLUE")
                if PROGRAM_STOP_EVENT.wait(CALIBRATION_RETRY_SECONDS):
                    raise ServiceStopRequested

        if PROGRAM_STOP_EVENT.is_set():
            raise ServiceStopRequested

        previous_frame_id = -1
        previous_time = time.monotonic()
        bad_timer = 0.0
        normal_confirm_timer = 0.0
        non_normal_latched = False

        def resume_monitoring():
            nonlocal bad_timer
            nonlocal normal_confirm_timer
            nonlocal non_normal_latched
            nonlocal previous_time

            if not runtime_state.set_paused(False):
                return

            non_normal_latched = False
            bad_timer = 0.0
            normal_confirm_timer = 0.0
            previous_time = time.monotonic()
            posture_band.set_state("P")
            set_led("WHITE")
            runtime_state.update(
                status="NO_POSE",
                score=None,
                bad_timer=0.0,
                side="-",
            )
            print(
                "[PROGRAM] Monitoring resumed."
            )

        while not PROGRAM_STOP_EVENT.is_set():
            frame_id, frame, _ = camera.get_latest(previous_frame_id)
            if frame is None:
                # Never keep a stale WARNING/BAD motor state while the camera
                # is unavailable. The capture worker keeps reopening in the
                # background; monitoring resumes from the newest frame.
                non_normal_latched = False
                bad_timer = 0.0
                normal_confirm_timer = 0.0

                if runtime_state.is_paused():
                    posture_band.set_state("S")
                    set_led("OFF")
                    safe_status = "PAUSED"
                else:
                    posture_band.set_state("P")
                    set_led("WHITE")
                    safe_status = "NO_POSE"

                runtime_state.update(
                    status=safe_status,
                    score=None,
                    bad_timer=0.0,
                    total_bad_time=total_bad_time,
                    total_warning_time=total_warning_time,
                    side="-",
                    camera_ok=False,
                )
                if camera.seconds_since_frame() >= CAMERA_FATAL_TIMEOUT_SECONDS:
                    raise RuntimeError(
                        "Camera did not recover within "
                        f"{CAMERA_FATAL_TIMEOUT_SECONDS:.0f} seconds."
                    )
                continue
            previous_frame_id = frame_id

            current_time = time.monotonic()
            elapsed_time = min(max(current_time - previous_time, 0.0), 0.5)
            previous_time = current_time

            if runtime_state.is_paused():
                non_normal_latched = False
                bad_timer = 0.0
                normal_confirm_timer = 0.0
                posture_band.set_state("S")
                set_led("OFF")
                runtime_state.update(
                    status="PAUSED",
                    score=None,
                    bad_timer=0.0,
                    total_bad_time=total_bad_time,
                    total_warning_time=total_warning_time,
                    side="-",
                )
            else:
                results = process_pose_frame(pose, frame)
                normalized_delta, ear_point, shoulder_point, side = (
                    get_ear_shoulder_data(
                        results,
                        frame.shape,
                        preferred_side=preferred_side,
                    )
                )

                posture_score = None
                model_source = "RULE"

                if normalized_delta is not None:
                    posture_error = normalized_delta - baseline
                    severity_hint, model_source = classify_posture(
                        normalized_delta,
                        posture_error,
                    )

                    posture_score = max(
                        0,
                        min(
                            100,
                            int(100 - max(posture_error, 0.0) * 700),
                        ),
                    )
                    total_monitored_time += elapsed_time
                    score_sum += posture_score
                    score_count += 1

                    if results.pose_landmarks is not None:
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                        )
                    cv2.line(
                        frame,
                        ear_point,
                        shoulder_point,
                        (255, 255, 0),
                        2,
                    )
                    cv2.circle(frame, ear_point, 7, (0, 0, 255), -1)
                    cv2.circle(frame, shoulder_point, 7, (0, 255, 0), -1)

                    if severity_hint in {"WARNING", "BAD"}:
                        non_normal_latched = True
                        normal_confirm_timer = 0.0
                    elif non_normal_latched:
                        if posture_error <= MARGIN_EXIT:
                            normal_confirm_timer += elapsed_time
                            if normal_confirm_timer >= NORMAL_CONFIRM_SECONDS:
                                non_normal_latched = False
                                normal_confirm_timer = 0.0
                                bad_timer = 0.0
                        else:
                            normal_confirm_timer = 0.0

                    if non_normal_latched:
                        if severity_hint == "BAD":
                            bad_timer = max(bad_timer, BAD_POSTURE_LIMIT)
                        elif posture_error > MARGIN_EXIT:
                            bad_timer += elapsed_time

                        if bad_timer >= BAD_POSTURE_LIMIT:
                            status_text = "BAD"
                            total_bad_time += elapsed_time
                            posture_band.set_state("B")
                            set_led("RED")
                        else:
                            status_text = "WARNING"
                            total_warning_time += elapsed_time
                            posture_band.set_state("W")
                            set_led("BLUE")
                    else:
                        status_text = "NORMAL"
                        bad_timer = 0.0
                        posture_band.set_state("N")
                        set_led("GREEN")

                    non_normal_time = total_warning_time + total_bad_time
                    good_ratio = max(
                        0.0,
                        (
                            (total_monitored_time - non_normal_time)
                            / total_monitored_time
                        )
                        * 100.0,
                    )
                    runtime_state.update(
                        status=status_text,
                        score=posture_score,
                        bad_timer=bad_timer,
                        total_bad_time=total_bad_time,
                        total_warning_time=total_warning_time,
                        good_ratio=good_ratio,
                        side=side,
                        model_source=model_source,
                    )
                else:
                    non_normal_latched = False
                    bad_timer = 0.0
                    normal_confirm_timer = 0.0
                    posture_band.set_state("P")
                    set_led("WHITE")
                    runtime_state.update(
                        status="NO_POSE",
                        score=None,
                        bad_timer=0.0,
                        total_bad_time=total_bad_time,
                        total_warning_time=total_warning_time,
                        side="-",
                        model_source=model_source,
                    )

            if ui_visible:
                snapshot = runtime_state.snapshot()
                if ui_mode == "LIVE":
                    draw_monitor_overlay(frame, snapshot)
                    ui_frame = frame
                else:
                    average_score = (
                        score_sum / score_count
                        if score_count > 0
                        else 0.0
                    )
                    ui_frame = build_summary_frame(
                        total_monitored_time,
                        total_warning_time,
                        total_bad_time,
                        average_score,
                        monitoring_paused=snapshot["paused"],
                    )

                cv2.imshow(window_name, ui_frame)
                key = cv2.waitKey(1) & 0xFF

                if window_was_closed(window_name):
                    resume_monitoring()
                    ui_visible = False
                    close_ui_window(window_name)
                    print(
                        "[UI] Window closed. Background posture "
                        "monitoring is still running."
                    )

                elif ui_mode == "LIVE" and key == ord(" "):
                    paused = runtime_state.toggle_paused()
                    non_normal_latched = False
                    bad_timer = 0.0
                    normal_confirm_timer = 0.0
                    previous_time = time.monotonic()

                    if paused:
                        posture_band.set_state("S")
                        set_led("OFF")
                        runtime_state.update(
                            status="PAUSED",
                            score=None,
                            bad_timer=0.0,
                            side="-",
                        )
                        print("[PROGRAM] Monitoring paused.")
                    else:
                        posture_band.set_state("P")
                        set_led("WHITE")
                        runtime_state.update(
                            status="NO_POSE",
                            score=None,
                            bad_timer=0.0,
                            side="-",
                        )
                        print("[PROGRAM] Monitoring resumed.")

                elif ui_mode == "LIVE" and key == ESC_KEY:
                    # Freeze all posture statistics while the report is shown.
                    runtime_state.set_paused(True)
                    non_normal_latched = False
                    bad_timer = 0.0
                    normal_confirm_timer = 0.0
                    posture_band.set_state("S")
                    set_led("OFF")
                    runtime_state.update(
                        status="PAUSED",
                        score=None,
                        bad_timer=0.0,
                        side="-",
                    )
                    ui_mode = "SUMMARY"
                    print(
                        "[UI] Summary opened. Posture monitoring paused."
                    )

                elif ui_mode == "SUMMARY" and key == ESC_KEY:
                    # ESC only: return to the live camera and resume.
                    resume_monitoring()
                    ui_mode = "LIVE"
                    print(
                        "[UI] Returned to live camera. Monitoring resumed."
                    )

        if PROGRAM_STOP_EVENT.is_set():
            raise ServiceStopRequested
    except ServiceStopRequested:
        if POWER_OFF_REQUESTED_EVENT.is_set():
            print("[PROGRAM] Safe shutdown requested by GPIO switch.")
        else:
            print("[PROGRAM] Stop requested by systemd.")
    except RuntimeError as runtime_error:
        exit_code = 1
        print(f"[ERROR] {runtime_error}")
    except KeyboardInterrupt:
        print("[PROGRAM] Interrupted from terminal.")
    except Exception as unexpected_error:
        exit_code = 1
        print(
            f"[ERROR] {type(unexpected_error).__name__}: "
            f"{unexpected_error}"
        )
    finally:
        shutdown_switch.stop()
        led_status.stop()
        posture_band.stop()
        camera.stop()

        if pose is not None:
            pose.close()

        if GPIO_AVAILABLE:
            GPIO.cleanup()

        if not HEADLESS:
            cv2.destroyAllWindows()
        print(f"[PROGRAM] Closed safely. exit_code={exit_code}")

    if POWER_OFF_REQUESTED_EVENT.is_set():
        request_os_poweroff()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
