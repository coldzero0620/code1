#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
import numpy as np

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
    OUT = "OUT"
    LOW = 0
    HIGH = 1

    def setmode(self, mode):
        pass

    def setwarnings(self, enabled):
        pass

    def setup(self, pin, mode):
        pass

    def output(self, pin, state):
        pass

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

    def __init__(self, width=800, height=600, fps=15, camera_index=0):
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

            data = self.process.stdout.read(4096)
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
        if self.process is None:
            return

        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)

        if self.process.stdout is not None:
            self.process.stdout.close()


# ---------------------------------------------------------------------------
# Hardware pins and posture parameters
# ---------------------------------------------------------------------------
# Actual RGB channel mapping confirmed from the physical LED:
# GPIO17 -> BLUE, GPIO27 -> GREEN, GPIO22 -> RED
LED_R = 22
LED_G = 27
LED_B = 17

FRAME_WIDTH = 800
FRAME_HEIGHT = 600
CAMERA_FPS = 15

MARGIN = 0.10
BAD_POSTURE_LIMIT = 2.0
VISIBILITY_THRESHOLD = 0.50
CALIBRATION_SECONDS = 3.0
CALIBRATION_RETRY_SECONDS = 2.0

# No monitor/VNC is required.  When DISPLAY is unavailable the program
# automatically runs in headless mode.  --headless can also force this mode.
HEADLESS = "--headless" in sys.argv or not os.environ.get("DISPLAY")

current_led_status = "GREEN"

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
for output_pin in (LED_R, LED_G, LED_B):
    GPIO.setup(output_pin, GPIO.OUT)


def set_led(color):
    global current_led_status

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


# ---------------------------------------------------------------------------
# BLE posture-band client
# XIAO ESP32-C3 advertises as "PostureBand" and receives posture state text.
# The wristband firmware, not Raspberry Pi GPIO, controls the vibration motor.
# ---------------------------------------------------------------------------
BLE_DEVICE_NAME = "PostureBand"
BLE_CHARACTERISTIC_UUID = "7d2a4b21-8f77-4e24-9a63-94a4ef0d12b2"
BLE_HEARTBEAT_SECONDS = 2.0


class BlePostureBand:
    def __init__(self):
        self._desired_state = "OFF"
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if not BLE_AVAILABLE:
            print("[BLE] bleak is not installed. Wristband is disabled.")
            return

        self._thread = threading.Thread(
            target=self._thread_main,
            name="posture-band-ble",
            daemon=True,
        )
        self._thread.start()

    def set_state(self, state):
        with self._state_lock:
            self._desired_state = state

    def stop(self):
        # Give a connected band a short chance to receive OFF. The firmware
        # also switches the motor off automatically on BLE disconnection.
        self.set_state("OFF")
        time.sleep(0.25)
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _thread_main(self):
        try:
            asyncio.run(self._ble_loop())
        except Exception as error:
            print(f"[BLE] Worker stopped: {error}")

    async def _ble_loop(self):
        while not self._stop_event.is_set():
            try:
                print(f"[BLE] Searching for {BLE_DEVICE_NAME}...")
                device = await BleakScanner.find_device_by_name(
                    BLE_DEVICE_NAME,
                    timeout=5.0,
                )
                if device is None:
                    await asyncio.sleep(1.0)
                    continue

                async with BleakClient(device) as client:
                    print(f"[BLE] Connected: {BLE_DEVICE_NAME}")
                    last_sent_state = None
                    last_sent_time = 0.0

                    while (
                        client.is_connected
                        and not self._stop_event.is_set()
                    ):
                        with self._state_lock:
                            state = self._desired_state

                        now = time.monotonic()
                        if (
                            state != last_sent_state
                            or now - last_sent_time
                            >= BLE_HEARTBEAT_SECONDS
                        ):
                            await client.write_gatt_char(
                                BLE_CHARACTERISTIC_UUID,
                                state.encode("utf-8"),
                                response=True,
                            )
                            if state != last_sent_state:
                                print(f"[BLE] State sent: {state}")
                            last_sent_state = state
                            last_sent_time = now

                        await asyncio.sleep(0.1)

            except Exception as error:
                if not self._stop_event.is_set():
                    print(f"[BLE] Connection lost: {error}")
                    await asyncio.sleep(2.0)


# ---------------------------------------------------------------------------
# Optional Random Forest model
#
# Supported model inputs:
#   1 feature: posture_error
#   2 features: normalized_delta, posture_error
#
# If no compatible model exists, the calibrated threshold rule is used.
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).resolve().with_name("posture-rf.joblib")
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


def prediction_is_bad(prediction):
    if isinstance(prediction, (bool, np.bool_)):
        return bool(prediction)

    if isinstance(prediction, (int, float, np.integer, np.floating)):
        return int(prediction) == 1

    label = str(prediction).strip().lower()
    return label in {"1", "bad", "warning", "forward_head", "forward-head"}


def classify_posture(normalized_delta, posture_error):
    global model_warning_shown

    rule_result = posture_error > MARGIN
    if rf_model is None:
        return rule_result, "RULE"

    try:
        feature_count = int(getattr(rf_model, "n_features_in_", 0))

        if feature_count == 1:
            features = np.array([[posture_error]], dtype=np.float32)
        elif feature_count == 2:
            features = np.array(
                [[normalized_delta, posture_error]],
                dtype=np.float32,
            )
        else:
            if not model_warning_shown:
                print(
                    "[MODEL] Unsupported feature count. "
                    "Using threshold rule instead."
                )
                model_warning_shown = True
            return rule_result, "RULE"

        prediction = rf_model.predict(features)[0]
        return prediction_is_bad(prediction), "RF"
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


def get_ear_shoulder_data(results, frame_shape):
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


class UserRequestedExit(Exception):
    pass


class ServiceStopRequested(Exception):
    """Raised when systemd sends SIGTERM via systemctl stop."""


def handle_service_stop(signum, frame):
    # Raising an exception interrupts a blocking camera read and transfers
    # control to main()'s finally block, which safely turns hardware off.
    raise ServiceStopRequested


def window_was_closed(window_name):
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def calibrate_baseline(cap, pose, window_name):
    print("=== Personal baseline calibration ===")
    print("Keep a comfortable upright posture for 3 seconds.")

    baseline_samples = []
    calibration_start = time.monotonic()

    while time.monotonic() - calibration_start < CALIBRATION_SECONDS:
        success, frame = cap.read()
        if not success or frame is None:
            raise RuntimeError("Camera disconnected during calibration.")

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = pose.process(rgb_frame)

        normalized_delta, ear_point, shoulder_point, side = (
            get_ear_shoulder_data(results, frame.shape)
        )

        if normalized_delta is not None:
            baseline_samples.append(normalized_delta)
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

        remaining = max(
            0.0,
            CALIBRATION_SECONDS
            - (time.monotonic() - calibration_start),
        )
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
            "Keep an upright posture",
            (40, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        if not HEADLESS:
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or window_was_closed(window_name):
                raise UserRequestedExit

    if not baseline_samples:
        raise RuntimeError(
            "Calibration failed. Ear and shoulder landmarks were not detected."
        )

    baseline = float(np.median(baseline_samples))
    print(f"[CALIBRATION] Baseline: {baseline:.4f}")
    return baseline


def show_summary(
    window_name,
    total_monitored_time,
    total_bad_time,
    average_score,
):
    if total_monitored_time <= 0:
        return

    good_ratio = max(
        0.0,
        (
            (total_monitored_time - total_bad_time)
            / total_monitored_time
        )
        * 100.0,
    )

    summary = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.rectangle(summary, (100, 80), (700, 520), (35, 35, 35), -1)
    cv2.rectangle(summary, (100, 80), (700, 520), (0, 215, 255), 2)

    lines = [
        ("POSTURE ANALYSIS REPORT", (165, 140), (0, 215, 255)),
        (
            f"Total tracked time : {total_monitored_time:.1f} sec",
            (150, 220),
            (255, 255, 255),
        ),
        (
            f"Total bad time     : {total_bad_time:.1f} sec",
            (150, 280),
            (150, 150, 255),
        ),
        (
            f"Average score      : {average_score:.1f} / 100",
            (150, 340),
            (255, 215, 0),
        ),
        (
            f"Good posture rate  : {good_ratio:.1f} %",
            (150, 400),
            (100, 255, 100),
        ),
        (
            "Press any key to exit",
            (260, 480),
            (150, 150, 150),
        ),
    ]

    for text, position, color in lines:
        cv2.putText(
            summary,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7 if position[1] != 480 else 0.6,
            color,
            2 if position[1] != 480 else 1,
        )

    cv2.imshow(window_name, summary)
    cv2.waitKey(0)


def main():
    # systemctl stop sends SIGTERM. Handle it so that camera, BLE, LED and
    # GPIO resources are cleaned up instead of leaving the process blocked.
    signal.signal(signal.SIGTERM, handle_service_stop)

    window_name = "Posture Monitor"
    cap = None
    pose = None
    posture_band = BlePostureBand()

    total_monitored_time = 0.0
    total_bad_time = 0.0
    score_sum = 0.0
    score_count = 0

    try:
        posture_band.start()
        print(f"[DISPLAY] {'Headless/offline mode' if HEADLESS else 'UI mode'}")
        print("[CAMERA] CSI via rpicam-vid")
        cap = RpicamCapture(
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
            fps=CAMERA_FPS,
            camera_index=0,
        )

        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # In standalone mode, do not terminate merely because nobody was in
        # front of the camera during boot. Yellow LED means "waiting/calibrating".
        set_led("BLUE")
        while True:
            try:
                baseline = calibrate_baseline(cap, pose, window_name)
                set_led("GREEN")
                break
            except RuntimeError as calibration_error:
                if not HEADLESS:
                    raise
                print(
                    f"[CALIBRATION] {calibration_error} "
                    f"Retrying in {CALIBRATION_RETRY_SECONDS:.0f} seconds."
                )
                posture_band.set_state("OFF")
                set_led("BLUE")
                time.sleep(CALIBRATION_RETRY_SECONDS)

        bad_timer = 0.0
        previous_time = time.monotonic()

        while cap.isOpened():
            success, frame = cap.read()
            if not success or frame is None:
                raise RuntimeError("Camera stream stopped.")

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            current_time = time.monotonic()
            elapsed_time = min(current_time - previous_time, 0.5)
            previous_time = current_time

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = pose.process(rgb_frame)

            normalized_delta, ear_point, shoulder_point, side = (
                get_ear_shoulder_data(results, frame.shape)
            )

            posture_score = None
            model_source = "RULE"

            if normalized_delta is not None:
                posture_error = normalized_delta - baseline
                bad_posture, model_source = classify_posture(
                    normalized_delta,
                    posture_error,
                )

                posture_score = (
                    max(0, int(100 - posture_error * 700))
                    if posture_error > 0
                    else 100
                )

                total_monitored_time += elapsed_time
                score_sum += posture_score
                score_count += 1

                cv2.line(
                    frame,
                    ear_point,
                    shoulder_point,
                    (255, 255, 0),
                    2,
                )
                cv2.circle(frame, ear_point, 7, (0, 0, 255), -1)
                cv2.circle(frame, shoulder_point, 7, (0, 255, 0), -1)

                if bad_posture:
                    bad_timer += elapsed_time
                    total_bad_time += elapsed_time

                    if bad_timer < BAD_POSTURE_LIMIT:
                        # Poor posture detected, but warning timer has not
                        # reached the BAD threshold yet.
                        set_led("BLUE")
                        status_text = "WARNING"
                        status_color = (255, 0, 0)  # OpenCV BGR: blue

                        # The XIAO firmware produces two 0.15-second pulses.
                        posture_band.set_state("WARNING")
                    else:
                        set_led("RED")
                        status_text = "BAD"
                        status_color = (0, 0, 255)
                        posture_band.set_state("BAD")
                else:
                    bad_timer = 0.0
                    set_led("GREEN")
                    posture_band.set_state("NORMAL")
                    status_text = "NORMAL"
                    status_color = (0, 255, 0)
            else:
                bad_posture = False
                bad_timer = 0.0
                set_led("BLUE")
                posture_band.set_state("NO_POSE")
                status_text = "NO POSE"
                status_color = (255, 0, 0)
                side = "-"

            if total_monitored_time > 0:
                good_ratio = max(
                    0.0,
                    (
                        (total_monitored_time - total_bad_time)
                        / total_monitored_time
                    )
                    * 100.0,
                )
            else:
                good_ratio = 100.0

            score_text = (
                f"{posture_score} / 100"
                if posture_score is not None
                else "--"
            )

            cv2.putText(
                frame,
                f"STATUS : {status_text}",
                (35, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
            )
            cv2.putText(
                frame,
                f"SCORE  : {score_text}",
                (35, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 215, 0),
                2,
            )
            cv2.putText(
                frame,
                f"TIMER  : {bad_timer:.1f}s",
                (35, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                frame,
                f"TOTAL BAD : {total_bad_time:.1f}s",
                (35, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 180, 255),
                1,
            )
            cv2.putText(
                frame,
                f"GOOD RATIO: {good_ratio:.1f}%",
                (35, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 255, 180),
                2,
            )
            cv2.putText(
                frame,
                f"SIDE: {side}  MODE: {model_source}",
                (35, 215),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
            )

            if current_led_status == "RED":
                led_color = (0, 0, 255)
            elif current_led_status == "BLUE":
                led_color = (255, 0, 0)
            else:
                led_color = (0, 255, 0)

            cv2.circle(
                frame,
                (FRAME_WIDTH - 60, FRAME_HEIGHT - 60),
                22,
                led_color,
                -1,
            )

            if not HEADLESS:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or window_was_closed(window_name):
                    print("[PROGRAM] Monitoring stopped by user.")
                    break

        average_score = (
            score_sum / score_count if score_count > 0 else 0.0
        )
        if not HEADLESS:
            show_summary(
                window_name,
                total_monitored_time,
                total_bad_time,
                average_score,
            )

    except UserRequestedExit:
        print("[PROGRAM] Closed by user.")
    except ServiceStopRequested:
        print("[PROGRAM] Stop requested by systemd.")
    except RuntimeError as runtime_error:
        print(f"[ERROR] {runtime_error}")
    except KeyboardInterrupt:
        print("[PROGRAM] Interrupted from terminal.")
    except Exception as unexpected_error:
        print(
            f"[ERROR] {type(unexpected_error).__name__}: "
            f"{unexpected_error}"
        )
    finally:
        posture_band.stop()
        set_led("OFF")

        if pose is not None:
            pose.close()
        if cap is not None:
            cap.release()

        if GPIO_AVAILABLE:
            GPIO.cleanup()

        if not HEADLESS:
            cv2.destroyAllWindows()
        print("[PROGRAM] Closed safely.")


if __name__ == "__main__":
    main()
