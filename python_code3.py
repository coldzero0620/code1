"""
거북목(전방머리자세) 실시간 감지 시스템 - v5

v4 대비 변경:
  1. BAD 전용 지속시간 집계  - NORMAL/WARNING/BAD 시간을 각각 분리 집계하고
                               BAD 연속 지속·최장 구간·발생 횟수를 추적
  2. 각도/거리 독립 특징     - 기존 delta 는 tan(angle) 과 동일해 중복이었음.
                               머리 크기를 기준으로 한 '거리' 특징을 새로 도입
  3. 순환 라벨링 차단        - 데이터 수집 중에는 규칙 판정 결과를 화면에서 숨김
  4. 상태 히스테리시스       - 진입/해제 임계값 분리 + 최소 유지시간
  5. baseline 오염 검증      - 보정 자세가 상식 범위를 벗어나면 경고
  6. 좌하단 LED 표시부       - 자세 상태에 따른 점등색을 라벨과 함께 표시

  ※ 특징 정의가 바뀌었으므로 기존 posture_dataset.csv 는 반드시
     삭제하거나 다른 이름으로 옮긴 뒤 새로 수집해야 합니다.

라벨 기준 (수집 시):
  1 = NORMAL   바르게 앉은 상태
  2 = WARNING  목을 살짝 앞으로 뺀 상태
  3 = BAD      목을 확실히 앞으로 뺀 상태
"""

import csv
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

try:
    import joblib
except ImportError:
    joblib = None


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------
PERSON_ID = "P01"
LANDMARK_SIDE = "AUTO"          # AUTO, LEFT, RIGHT
CAMERA_MODE = "AUTO"            # AUTO, PI, WEBCAM
WEBCAM_ID = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
WINDOW_NAME = "Posture Detection"

DATASET_PATH = Path("posture_dataset.csv")
MODEL_PATH = Path("posture_rf.joblib")

# --- [2] 각도축과 거리축이 서로 독립인 특징 3개 ---
#   neck_angle_error   : 귀-어깨 선의 수직 대비 각도 편차        (각도축)
#   head_distance_error: 귀-어깨 수평거리 / 머리크기 의 편차      (거리축)
#   head_tilt_error    : 코-귀 선의 기울기 편차 (고개 숙임)       (독립축)
FEATURE_NAMES = [
    "neck_angle_error",
    "head_distance_error",
    "head_tilt_error",
]

VALID_LABELS = ("NORMAL", "WARNING", "BAD")


# ---------------------------------------------------------------------------
# Posture thresholds
# ---------------------------------------------------------------------------
# --- [4] 진입/해제 임계값 분리 (히스테리시스). 단위: 도(degree) ---
WARNING_ENTER_DEG = 7.0
WARNING_EXIT_DEG = 5.0
BAD_ENTER_DEG = 15.0
BAD_EXIT_DEG = 12.0

STATE_MIN_HOLD = 0.4            # 상태 전환에 필요한 최소 유지시간(초)
SCORE_PENALTY_PER_DEG = 3.5     # 각도 1도당 감점

BAD_ALERT_LIMIT = 2.0           # BAD 연속 지속 이 시간 초과 시 진동
VIBRATION_DURATION = 2.0
ALERT_REPEAT_INTERVAL = 10.0

EMA_ALPHA = 0.3
LOST_RESET_TIMEOUT = 2.0        # 인식 끊김이 길어지면 연속 구간 리셋

VISIBILITY_THRESHOLD = 0.5
MIN_Y_HEIGHT_PX = 10.0
MIN_HEAD_SCALE_PX = 15.0
FRONT_VIEW_RATIO = 0.25

CALIBRATION_DURATION = 6.0
CALIBRATION_MIN_SAMPLES = 15
# --- [5] 보정 시 절대 목 각도가 이 범위를 벗어나면 오염 의심 ---
BASELINE_ANGLE_MIN = 0.0
BASELINE_ANGLE_MAX = 25.0

DATA_SAVE_INTERVAL = 0.20
PREDICTION_WINDOW = 10
MAX_ELAPSED = 0.5

# --- [3] 수집 중 규칙 판정을 화면에서 숨겨 순환 라벨링 차단 ---
BLIND_WHILE_RECORDING = True


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
MOTOR_PIN = 18
LED_R, LED_G, LED_B = 17, 27, 22


class DummyGPIO:
    BCM = "BCM"
    OUT = "OUT"
    LOW = 0
    HIGH = 1

    def setwarnings(self, enabled):
        pass

    def setmode(self, mode):
        pass

    def setup(self, pins, mode):
        pass

    def output(self, pin, state):
        pass

    def cleanup(self):
        pass


try:
    import RPi.GPIO as GPIO

    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup([MOTOR_PIN, LED_R, LED_G, LED_B], GPIO.OUT)
    print("[GPIO] Real Raspberry Pi GPIO enabled")
except Exception as gpio_error:
    GPIO = DummyGPIO()
    GPIO.setmode(GPIO.BCM)
    GPIO.setup([MOTOR_PIN, LED_R, LED_G, LED_B], GPIO.OUT)
    print(f"[GPIO] Simulation mode: {gpio_error}")


current_led_status = "OFF"

# --- [6] 자세 상태 -> LED 색 매핑 (하드웨어와 화면 표시가 같은 표를 사용) ---
STATE_TO_LED = {
    "NORMAL": "GREEN",
    "WARNING": "YELLOW",
    "BAD": "RED",
    "UNKNOWN": "BLUE",
}

LED_DISPLAY = {
    "GREEN": ((0, 255, 0), "GREEN"),
    "YELLOW": ((0, 255, 255), "YELLOW"),
    "RED": ((0, 0, 255), "RED"),
    "BLUE": ((255, 120, 0), "BLUE"),
    "OFF": ((70, 70, 70), "OFF"),
}


def set_led(color):
    global current_led_status

    current_led_status = color

    GPIO.output(LED_R, GPIO.LOW)
    GPIO.output(LED_G, GPIO.LOW)
    GPIO.output(LED_B, GPIO.LOW)

    if color == "GREEN":
        GPIO.output(LED_G, GPIO.HIGH)
    elif color == "YELLOW":
        GPIO.output(LED_R, GPIO.HIGH)
        GPIO.output(LED_G, GPIO.HIGH)
    elif color == "RED":
        GPIO.output(LED_R, GPIO.HIGH)
    elif color == "BLUE":
        GPIO.output(LED_B, GPIO.HIGH)


def motor_on():
    GPIO.output(MOTOR_PIN, GPIO.HIGH)


def motor_off():
    GPIO.output(MOTOR_PIN, GPIO.LOW)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
class CameraSource:
    def __init__(self):
        self.picam2 = None
        self.cap = None
        self.source_name = ""

        requested_mode = CAMERA_MODE.upper()

        if requested_mode not in {"AUTO", "PI", "WEBCAM"}:
            raise ValueError("CAMERA_MODE must be AUTO, PI, or WEBCAM")

        if requested_mode in {"AUTO", "PI"}:
            try:
                from picamera2 import Picamera2

                self.picam2 = Picamera2()
                configuration = self.picam2.create_video_configuration(
                    main={
                        "size": (FRAME_WIDTH, FRAME_HEIGHT),
                        "format": "RGB888",
                    }
                )
                self.picam2.configure(configuration)
                self.picam2.start()
                time.sleep(1.0)

                self.source_name = "Raspberry Pi CSI Camera"
                print(f"[CAMERA] {self.source_name}")
                return
            except Exception as camera_error:
                self.picam2 = None

                if requested_mode == "PI":
                    raise RuntimeError(
                        f"Unable to start Raspberry Pi camera: {camera_error}"
                    ) from camera_error

                print(f"[CAMERA] Pi camera unavailable: {camera_error}")

        self.cap = cv2.VideoCapture(WEBCAM_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not self.cap.isOpened():
            raise RuntimeError(
                "Unable to open a camera. Check the CSI cable or WEBCAM_ID."
            )

        self.source_name = f"OpenCV Webcam {WEBCAM_ID}"
        print(f"[CAMERA] {self.source_name}")

    def read(self):
        if self.picam2 is not None:
            try:
                return True, self.picam2.capture_array("main")
            except Exception:
                return False, None

        return self.cap.read()

    def close(self):
        if self.picam2 is not None:
            try:
                self.picam2.stop()
            finally:
                self.picam2.close()

        if self.cap is not None:
            self.cap.release()


# ---------------------------------------------------------------------------
# Random Forest classifier
# ---------------------------------------------------------------------------
class PostureClassifier:
    def __init__(self, model_path):
        self.model = None
        self.classes = None
        self.probability_history = deque(maxlen=PREDICTION_WINDOW)
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            print(
                f"[MODEL] {self.model_path} not found. "
                "Using the calibrated threshold rule."
            )
            return

        if joblib is None:
            print(
                "[MODEL] joblib is not installed. "
                "Run: python -m pip install joblib scikit-learn pandas"
            )
            return

        try:
            bundle = joblib.load(self.model_path)

            if not isinstance(bundle, dict) or "model" not in bundle:
                raise ValueError("Invalid model bundle")

            saved_features = bundle.get("features")

            if saved_features != FEATURE_NAMES:
                raise ValueError(
                    "Feature order mismatch. "
                    f"model={saved_features} app={FEATURE_NAMES}"
                )

            self.model = bundle["model"]
            self.classes = np.asarray(self.model.classes_, dtype=str)
            print(f"[MODEL] Loaded {self.model_path}")
        except Exception as model_error:
            self.model = None
            self.classes = None
            print(f"[MODEL] Load failed; threshold mode enabled: {model_error}")

    @property
    def is_loaded(self):
        return self.model is not None

    def reset(self):
        self.probability_history.clear()

    def predict(self, feature_map):
        input_data = np.array(
            [[feature_map[name] for name in FEATURE_NAMES]],
            dtype=np.float32,
        )

        probabilities = self.model.predict_proba(input_data)[0]
        self.probability_history.append(probabilities)

        averaged = np.mean(np.asarray(self.probability_history), axis=0)
        best_index = int(np.argmax(averaged))

        return str(self.classes[best_index]).upper(), float(averaged[best_index])


# ---------------------------------------------------------------------------
# Landmark extraction
# ---------------------------------------------------------------------------
mp_pose = mp.solutions.pose

STATUS_MESSAGES = {
    "NO_LANDMARK": "NO PERSON (사람 미인식)",
    "FRONT_VIEW": "FRONT VIEW (옆으로 앉으세요)",
    "LOW_VISIBILITY": "EAR/SHOULDER HIDDEN (귀/어깨 가림)",
    "NOSE_HIDDEN": "NOSE HIDDEN (얼굴이 가려짐)",
    "BAD_GEOMETRY": "INVALID GEOMETRY (귀-어깨 높이차 부족)",
    "SMALL_HEAD": "TOO FAR (카메라에 더 가까이)",
}


def select_landmark_side(landmarks):
    left_score = min(
        landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].visibility,
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility,
    )
    right_score = min(
        landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].visibility,
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility,
    )

    requested_side = LANDMARK_SIDE.upper()

    if requested_side == "LEFT":
        return "LEFT", left_score
    if requested_side == "RIGHT":
        return "RIGHT", right_score
    if left_score >= right_score:
        return "LEFT", left_score

    return "RIGHT", right_score


def get_posture_data(results, frame_shape):
    """반환: (status, data). status == "OK" 일 때만 측정값이 유효."""
    if results.pose_landmarks is None:
        return "NO_LANDMARK", None

    landmarks = results.pose_landmarks.landmark
    height, width = frame_shape[:2]

    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # 정면/측면 자동 판별
    ear_span = abs(left_ear.x - right_ear.x) * width
    shoulder_span = abs(left_shoulder.x - right_shoulder.x) * width
    view_ratio = ear_span / shoulder_span if shoulder_span > 1.0 else -1.0

    if view_ratio > FRONT_VIEW_RATIO:
        return "FRONT_VIEW", {"view_ratio": view_ratio}

    side, side_score = select_landmark_side(landmarks)

    if side == "LEFT":
        ear, shoulder = left_ear, left_shoulder
    else:
        ear, shoulder = right_ear, right_shoulder

    if side_score < VISIBILITY_THRESHOLD:
        return "LOW_VISIBILITY", {"visibility": float(side_score)}

    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

    # 코는 머리 크기 기준선과 고개 숙임 계산에 모두 필요하므로 필수로 승격
    if nose.visibility < VISIBILITY_THRESHOLD:
        return "NOSE_HIDDEN", {"visibility": float(nose.visibility)}

    nose_point = (int(nose.x * width), int(nose.y * height))
    ear_point = (int(ear.x * width), int(ear.y * height))
    shoulder_point = (int(shoulder.x * width), int(shoulder.y * height))

    x_delta_pixel = abs(ear.x - shoulder.x) * width
    y_height_pixel = abs(ear.y - shoulder.y) * height

    if y_height_pixel < MIN_Y_HEIGHT_PX:
        return "BAD_GEOMETRY", {"y_height": float(y_height_pixel)}

    # --- [2] 각도축: 귀-어깨 선이 수직에서 얼마나 기울었는가 ---
    neck_angle = math.degrees(math.atan2(x_delta_pixel, y_height_pixel))

    # --- [2] 거리축: 머리 크기(코-귀)를 자로 삼은 수평 전방 이동량 ---
    #   머리 크기는 자세와 무관한 강체 치수이므로 카메라 거리만 상쇄된다.
    #   분모가 각도 계산의 분모(y_height)와 다르므로 각도와 독립이다.
    head_scale_pixel = math.hypot(
        nose_point[0] - ear_point[0],
        nose_point[1] - ear_point[1],
    )

    if head_scale_pixel < MIN_HEAD_SCALE_PX:
        return "SMALL_HEAD", {"head_scale": float(head_scale_pixel)}

    head_distance = x_delta_pixel / head_scale_pixel

    # 고개 숙임: 코가 귀보다 얼마나 아래에 있는가
    nose_ear_dx = max(abs(nose_point[0] - ear_point[0]), 1)
    head_tilt = math.degrees(
        math.atan2(nose_point[1] - ear_point[1], nose_ear_dx)
    )

    return "OK", {
        "side": side,
        "visibility": float(side_score),
        "view_ratio": view_ratio,
        "nose_point": nose_point,
        "ear_point": ear_point,
        "shoulder_point": shoulder_point,
        "head_scale": float(head_scale_pixel),
        "neck_angle": float(neck_angle),
        "head_distance": float(head_distance),
        "head_tilt": float(head_tilt),
    }


SMOOTH_KEYS = ("neck_angle", "head_distance", "head_tilt")


def apply_ema(previous, current):
    if previous is None:
        return {key: current[key] for key in SMOOTH_KEYS}

    return {
        key: EMA_ALPHA * current[key] + (1.0 - EMA_ALPHA) * previous[key]
        for key in SMOOTH_KEYS
    }


def build_feature_map(smoothed, baseline):
    return {
        "neck_angle_error": float(smoothed["neck_angle"] - baseline["angle"]),
        "head_distance_error": float(
            smoothed["head_distance"] - baseline["distance"]
        ),
        "head_tilt_error": float(smoothed["head_tilt"] - baseline["tilt"]),
    }


def draw_landmarks(frame, posture_data):
    nose_point = posture_data["nose_point"]
    ear_point = posture_data["ear_point"]
    shoulder_point = posture_data["shoulder_point"]

    cv2.line(frame, ear_point, shoulder_point, (255, 255, 0), 2)
    cv2.line(frame, nose_point, ear_point, (255, 0, 255), 2)

    # 수직 기준선 (목 각도의 기준)
    cv2.line(
        frame,
        shoulder_point,
        (shoulder_point[0], shoulder_point[1] - 90),
        (120, 120, 120),
        1,
    )

    cv2.circle(frame, nose_point, 6, (255, 0, 255), -1)
    cv2.circle(frame, ear_point, 7, (0, 0, 255), -1)
    cv2.circle(frame, shoulder_point, 7, (0, 255, 0), -1)


# ---------------------------------------------------------------------------
# [4] 히스테리시스 + 최소 유지시간 상태 머신
# ---------------------------------------------------------------------------
class PostureStateMachine:
    def __init__(self):
        self.state = "UNKNOWN"
        self.pending_state = None
        self.pending_time = 0.0

    def reset(self, state="UNKNOWN"):
        self.state = state
        self.pending_state = None
        self.pending_time = 0.0

    def _raw_from_rule(self, angle_error):
        """진입/해제 임계값을 분리해 경계에서의 떨림을 억제."""
        if self.state == "BAD":
            if angle_error < WARNING_EXIT_DEG:
                return "NORMAL"
            if angle_error < BAD_EXIT_DEG:
                return "WARNING"
            return "BAD"

        if self.state == "WARNING":
            if angle_error > BAD_ENTER_DEG:
                return "BAD"
            if angle_error < WARNING_EXIT_DEG:
                return "NORMAL"
            return "WARNING"

        if angle_error > BAD_ENTER_DEG:
            return "BAD"
        if angle_error > WARNING_ENTER_DEG:
            return "WARNING"

        return "NORMAL"

    def update(self, candidate, elapsed):
        """후보 상태가 STATE_MIN_HOLD 이상 유지될 때만 실제로 전환."""
        if candidate == self.state:
            self.pending_state = None
            self.pending_time = 0.0
            return self.state

        if candidate == self.pending_state:
            self.pending_time += elapsed
        else:
            self.pending_state = candidate
            self.pending_time = elapsed

        if self.pending_time >= STATE_MIN_HOLD or self.state == "UNKNOWN":
            self.state = candidate
            self.pending_state = None
            self.pending_time = 0.0

        return self.state

    def update_by_rule(self, angle_error, elapsed):
        return self.update(self._raw_from_rule(angle_error), elapsed)


# ---------------------------------------------------------------------------
# [1] 상태별 시간 집계
# ---------------------------------------------------------------------------
class PostureStats:
    def __init__(self):
        self.time_by_state = {"NORMAL": 0.0, "WARNING": 0.0, "BAD": 0.0}
        self.valid_time = 0.0
        self.lost_time = 0.0
        self.score_time_sum = 0.0

        self.bad_streak = 0.0
        self.longest_bad_streak = 0.0
        self.bad_episodes = 0
        self._in_bad_episode = False

    def add_valid(self, state, elapsed, score):
        self.valid_time += elapsed
        self.score_time_sum += score * elapsed

        if state in self.time_by_state:
            self.time_by_state[state] += elapsed

        if state == "BAD":
            if not self._in_bad_episode:
                self._in_bad_episode = True
                self.bad_episodes += 1
                self.bad_streak = 0.0

            self.bad_streak += elapsed
            self.longest_bad_streak = max(
                self.longest_bad_streak,
                self.bad_streak,
            )
        else:
            self._in_bad_episode = False
            self.bad_streak = 0.0

    def add_lost(self, elapsed):
        self.lost_time += elapsed

    def break_streak(self):
        """인식이 오래 끊기면 연속 구간을 끊는다."""
        self._in_bad_episode = False
        self.bad_streak = 0.0

    @property
    def average_score(self):
        if self.valid_time <= 0:
            return 0.0

        return self.score_time_sum / self.valid_time

    @property
    def good_ratio(self):
        if self.valid_time <= 0:
            return 100.0

        return self.time_by_state["NORMAL"] / self.valid_time * 100.0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibrate_baseline(camera, pose):
    print("\n=== Personal baseline calibration ===")
    print("카메라에 옆모습이 보이도록 앉아 바른 자세를 유지하세요.")

    angle_values = []
    distance_values = []
    tilt_values = []
    status_counts = {}

    start = time.monotonic()
    cancelled = False

    while time.monotonic() - start < CALIBRATION_DURATION:
        success, frame = camera.read()

        if not success:
            raise RuntimeError("Camera disconnected during calibration")

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = pose.process(rgb_frame)

        status, posture_data = get_posture_data(results, frame.shape)
        status_counts[status] = status_counts.get(status, 0) + 1

        if status == "OK":
            angle_values.append(posture_data["neck_angle"])
            distance_values.append(posture_data["head_distance"])
            tilt_values.append(posture_data["head_tilt"])
            draw_landmarks(frame, posture_data)
            message = "Calibrating - keep the reference posture"
            message_color = (0, 255, 255)
        else:
            message = STATUS_MESSAGES.get(status, status)
            message_color = (0, 165, 255)

        remaining = max(0.0, CALIBRATION_DURATION - (time.monotonic() - start))

        cv2.putText(
            frame,
            f"Calibrating: {remaining:.1f}s   samples: {len(angle_values)}",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            message,
            (25, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            message_color,
            2,
        )

        cv2.imshow(WINDOW_NAME, frame)

        if (cv2.waitKey(1) & 0xFF) in {ord("q"), 27}:
            cancelled = True
            break

    if cancelled:
        raise RuntimeError("Calibration cancelled by user")

    if len(angle_values) < CALIBRATION_MIN_SAMPLES:
        breakdown = ", ".join(
            f"{name}={count}" for name, count in sorted(status_counts.items())
        )
        raise RuntimeError(
            f"Calibration failed: valid samples {len(angle_values)} "
            f"(need {CALIBRATION_MIN_SAMPLES}). Frame breakdown -> {breakdown}"
        )

    baseline = {
        "angle": float(np.median(angle_values)),
        "distance": float(np.median(distance_values)),
        "tilt": float(np.median(tilt_values)),
        "angle_spread": float(np.std(angle_values)),
    }

    print(
        f"Baseline: angle={baseline['angle']:.2f}deg "
        f"distance={baseline['distance']:.3f} "
        f"tilt={baseline['tilt']:.2f}deg "
        f"({len(angle_values)} frames)"
    )

    # --- [5] baseline 오염 검증 ---
    warnings = []

    if not BASELINE_ANGLE_MIN <= baseline["angle"] <= BASELINE_ANGLE_MAX:
        warnings.append(
            f"목 각도 {baseline['angle']:.1f}deg 는 바른 자세 범위"
            f"({BASELINE_ANGLE_MIN:.0f}~{BASELINE_ANGLE_MAX:.0f}deg)를 벗어납니다"
        )

    if baseline["angle_spread"] > 4.0:
        warnings.append(
            f"보정 중 흔들림이 큽니다 (표준편차 {baseline['angle_spread']:.1f}deg)"
        )

    if warnings:
        print("\n[!] BASELINE WARNING")
        for line in warnings:
            print(f"    - {line}")
        print("    C 키로 재보정하는 것을 권장합니다.\n")
    else:
        print("Baseline OK\n")

    baseline["warnings"] = warnings

    return baseline


# ---------------------------------------------------------------------------
# Training-data collection
# ---------------------------------------------------------------------------
def append_training_sample(
    person_id,
    session_id,
    baseline,
    posture_data,
    smoothed,
    feature_map,
    label,
):
    file_exists = DATASET_PATH.exists()

    fieldnames = [
        "timestamp",
        "person_id",
        "session_id",
        "side",
        "baseline_angle",
        "baseline_distance",
        "baseline_tilt",
        "raw_neck_angle",
        "raw_head_distance",
        "raw_head_tilt",
        *FEATURE_NAMES,
        "label",
    ]

    row = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "person_id": person_id,
        "session_id": session_id,
        "side": posture_data["side"],
        "baseline_angle": baseline["angle"],
        "baseline_distance": baseline["distance"],
        "baseline_tilt": baseline["tilt"],
        "raw_neck_angle": smoothed["neck_angle"],
        "raw_head_distance": smoothed["head_distance"],
        "raw_head_tilt": smoothed["head_tilt"],
        **feature_map,
        "label": label,
    }

    with DATASET_PATH.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
STATE_COLOR = {
    "NORMAL": (0, 255, 0),
    "WARNING": (0, 255, 255),
    "BAD": (0, 0, 255),
    "UNKNOWN": (170, 170, 170),
}


def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"

    return f"{int(seconds // 60)}m {seconds % 60:04.1f}s"


def draw_led_panel(frame, state, blinded):
    """[6] 좌하단 LED 표시부: 자세 상태에 따른 점등색 + 라벨."""
    height = frame.shape[0]
    panel_left = 15
    panel_top = height - 78
    panel_right = 190
    panel_bottom = height - 12

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_left, panel_top),
        (panel_right, panel_bottom),
        (25, 25, 25),
        -1,
    )
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(
        frame,
        (panel_left, panel_top),
        (panel_right, panel_bottom),
        (90, 90, 90),
        1,
    )

    cv2.putText(
        frame,
        "LED",
        (panel_left + 10, panel_top + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (170, 170, 170),
        1,
    )

    led_key = "OFF" if blinded else STATE_TO_LED.get(state, "OFF")
    led_color, led_name = LED_DISPLAY[led_key]

    center = (panel_left + 32, panel_top + 45)

    # 발광 느낌을 주기 위한 외곽 halo
    cv2.circle(frame, center, 20, tuple(int(c * 0.30) for c in led_color), -1)
    cv2.circle(frame, center, 14, led_color, -1)
    cv2.circle(frame, center, 14, (240, 240, 240), 1)

    cv2.putText(
        frame,
        led_name,
        (panel_left + 58, panel_top + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        led_color,
        2,
    )
    cv2.putText(
        frame,
        "HIDDEN" if blinded else state,
        (panel_left + 58, panel_top + 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (200, 200, 200),
        1,
    )


def draw_live_ui(frame, view):
    stats = view["stats"]
    blinded = view["blinded"]
    state = view["state"]

    if blinded:
        # --- [3] 수집 중에는 판정 결과를 숨겨 순환 라벨링을 막는다 ---
        cv2.putText(
            frame,
            "COLLECTION MODE - judgement hidden",
            (25, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 200, 0),
            2,
        )
        cv2.putText(
            frame,
            "Label from your own judgement, not the app",
            (25, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
        )
    else:
        cv2.putText(
            frame,
            f"STATUS: {state}",
            (25, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            STATE_COLOR.get(state, (255, 255, 255)),
            2,
        )

        if view["detail"]:
            cv2.putText(
                frame,
                view["detail"],
                (25, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.47,
                (255, 165, 0),
                1,
            )

        score_text = (
            f"{view['score']} / 100"
            if view["score"] is not None
            else "-- / 100"
        )

        # --- [1] BAD 지속시간 중심의 표시 ---
        rows = [
            (f"SCORE: {score_text}", 84, 0.58, (255, 215, 0), 2),
            (
                f"BAD NOW  : {format_duration(stats.bad_streak)}",
                112,
                0.58,
                (0, 0, 255) if stats.bad_streak > 0 else (200, 200, 200),
                2,
            ),
            (
                f"BAD TOTAL: {format_duration(stats.time_by_state['BAD'])}"
                f"  ({stats.bad_episodes}x)",
                137,
                0.5,
                (150, 150, 255),
                1,
            ),
            (
                f"BAD LONGEST: {format_duration(stats.longest_bad_streak)}",
                159,
                0.46,
                (150, 150, 255),
                1,
            ),
            (
                f"WARNING TOTAL: {format_duration(stats.time_by_state['WARNING'])}",
                181,
                0.46,
                (150, 220, 220),
                1,
            ),
            (
                f"GOOD RATIO: {stats.good_ratio:.1f}%",
                203,
                0.5,
                (180, 255, 180),
                1,
            ),
            (
                f"LOST: {format_duration(stats.lost_time)}",
                224,
                0.42,
                (150, 150, 150),
                1,
            ),
            (f"MODE: {view['mode']}", 245, 0.44, (220, 220, 220), 1),
        ]

        for text, y_position, scale, color, thickness in rows:
            cv2.putText(
                frame,
                text,
                (25, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thickness,
            )

        if view["confidence"] is not None:
            cv2.putText(
                frame,
                f"MODEL VOTE: {view['confidence']:.2f}",
                (25, 266),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (220, 220, 220),
                1,
            )

        if view["features"] is not None:
            features = view["features"]
            cv2.putText(
                frame,
                f"ANG {features['neck_angle_error']:+.1f}deg   "
                f"DIST {features['head_distance_error']:+.2f}   "
                f"TILT {features['head_tilt_error']:+.1f}deg",
                (25, 287),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (230, 230, 230),
                1,
            )

    # 기록 상태는 항상 표시 (우측 상단)
    record_label = view["record_label"]
    recording_text = (
        f"REC: {record_label}" if record_label is not None else "REC: STOP"
    )
    cv2.putText(
        frame,
        recording_text,
        (frame.shape[1] - 175, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        STATE_COLOR.get(record_label, (170, 170, 170)),
        2,
    )

    if record_label is not None:
        cv2.putText(
            frame,
            f"rows: {view['saved_rows']}",
            (frame.shape[1] - 175, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
        )

    draw_led_panel(frame, state, blinded)

    cv2.putText(
        frame,
        "1:upright  2:slight  3:severe  0:stop  C:recalib  Q:quit",
        (15, frame.shape[0] - 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (200, 200, 200),
        1,
    )


def show_summary(stats):
    if stats.valid_time <= 0:
        print("No valid monitoring time was recorded.")
        return

    bad_time = stats.time_by_state["BAD"]
    warning_time = stats.time_by_state["WARNING"]
    normal_time = stats.time_by_state["NORMAL"]

    lines = [
        ("POSTURE ANALYSIS REPORT", (0, 215, 255)),
        (f"Valid time    : {format_duration(stats.valid_time)}", (255, 255, 255)),
        (
            f"NORMAL        : {format_duration(normal_time)}"
            f"  ({normal_time / stats.valid_time * 100:.1f}%)",
            (100, 255, 100),
        ),
        (
            f"WARNING       : {format_duration(warning_time)}"
            f"  ({warning_time / stats.valid_time * 100:.1f}%)",
            (0, 255, 255),
        ),
        (
            f"BAD           : {format_duration(bad_time)}"
            f"  ({bad_time / stats.valid_time * 100:.1f}%)",
            (120, 120, 255),
        ),
        (
            f"BAD episodes  : {stats.bad_episodes} times, "
            f"longest {format_duration(stats.longest_bad_streak)}",
            (120, 120, 255),
        ),
        (f"Average score : {stats.average_score:.1f} / 100", (255, 215, 0)),
        (f"Lost          : {format_duration(stats.lost_time)}", (170, 170, 170)),
    ]

    summary = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    cv2.rectangle(
        summary,
        (30, 30),
        (FRAME_WIDTH - 30, FRAME_HEIGHT - 30),
        (32, 32, 32),
        -1,
    )
    cv2.rectangle(
        summary,
        (30, 30),
        (FRAME_WIDTH - 30, FRAME_HEIGHT - 30),
        (0, 215, 255),
        2,
    )

    y_position = 78
    for index, (line, color) in enumerate(lines):
        cv2.putText(
            summary,
            line,
            (55, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62 if index == 0 else 0.5,
            color,
            2 if index == 0 else 1,
        )
        y_position += 44

    cv2.putText(
        summary,
        "Press any key to exit",
        (55, FRAME_HEIGHT - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (150, 150, 150),
        1,
    )

    print("\n=== POSTURE ANALYSIS REPORT ===")
    for line, _ in lines[1:]:
        print("  " + line)

    while True:
        cv2.imshow(WINDOW_NAME, summary)

        if (cv2.waitKey(30) & 0xFF) != 255:
            break

        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    camera = None
    pose = None
    stats = PostureStats()

    try:
        camera = CameraSource()
        classifier = PostureClassifier(MODEL_PATH)

        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,       # 라즈베리파이에서는 0
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        baseline = calibrate_baseline(camera, pose)

        session_id = time.strftime("%Y%m%d_%H%M%S")
        state_machine = PostureStateMachine()

        record_label = None
        saved_rows = 0
        last_save_time = 0.0

        smoothed = None
        lost_streak = 0.0
        last_alert_time = None
        motor_start_time = None
        previous_time = time.monotonic()

        print(f"[DATA] Person: {PERSON_ID}   Session: {session_id}")
        print("[KEYS] 1=NORMAL(바른자세) 2=WARNING(살짝) 3=BAD(심하게)")
        print("       0=기록중지  C=재보정  Q=종료")

        if BLIND_WHILE_RECORDING:
            print(
                "[INFO] 기록 중에는 판정 결과가 숨겨집니다. "
                "화면이 아닌 본인 판단으로 라벨을 누르세요."
            )

        while True:
            success, frame = camera.read()

            if not success:
                print("Camera disconnected. Opening the summary.")
                break

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            current_time = time.monotonic()
            elapsed = min(current_time - previous_time, MAX_ELAPSED)
            previous_time = current_time

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = pose.process(rgb_frame)

            status, posture_data = get_posture_data(results, frame.shape)

            feature_map = None
            confidence = None
            posture_score = None
            detail_text = ""

            if status == "OK":
                lost_streak = 0.0
                draw_landmarks(frame, posture_data)

                smoothed = apply_ema(smoothed, posture_data)
                feature_map = build_feature_map(smoothed, baseline)

                posture_score = max(
                    0,
                    int(
                        100
                        - max(0.0, feature_map["neck_angle_error"])
                        * SCORE_PENALTY_PER_DEG
                    ),
                )

                if classifier.is_loaded:
                    candidate, confidence = classifier.predict(feature_map)

                    if candidate not in VALID_LABELS:
                        candidate = "NORMAL"

                    # --- [4] 모델 출력에도 동일한 최소 유지시간 적용 ---
                    posture_state = state_machine.update(candidate, elapsed)
                else:
                    posture_state = state_machine.update_by_rule(
                        feature_map["neck_angle_error"],
                        elapsed,
                    )

                stats.add_valid(posture_state, elapsed, posture_score)

                if (
                    record_label is not None
                    and current_time - last_save_time >= DATA_SAVE_INTERVAL
                ):
                    append_training_sample(
                        PERSON_ID,
                        session_id,
                        baseline,
                        posture_data,
                        smoothed,
                        feature_map,
                        record_label,
                    )
                    saved_rows += 1
                    last_save_time = current_time
            else:
                posture_state = "UNKNOWN"
                stats.add_lost(elapsed)
                lost_streak += elapsed
                classifier.reset()
                smoothed = None
                detail_text = STATUS_MESSAGES.get(status, status)

                if status == "FRONT_VIEW" and posture_data:
                    detail_text += f"  ratio={posture_data['view_ratio']:.2f}"

                # 인식이 오래 끊기면 연속 BAD 구간을 끊는다
                if lost_streak >= LOST_RESET_TIMEOUT:
                    stats.break_streak()
                    state_machine.reset()
                    last_alert_time = None

            # ---------------- Hardware feedback ----------------
            set_led(STATE_TO_LED.get(posture_state, "OFF"))

            # --- [1] 진동은 BAD 연속 지속 기준으로만 발동 ---
            if posture_state == "BAD" and stats.bad_streak >= BAD_ALERT_LIMIT:
                if (
                    last_alert_time is None
                    or current_time - last_alert_time >= ALERT_REPEAT_INTERVAL
                ):
                    motor_on()
                    motor_start_time = current_time
                    last_alert_time = current_time
            elif posture_state != "BAD":
                last_alert_time = None

            if (
                motor_start_time is not None
                and current_time - motor_start_time >= VIBRATION_DURATION
            ):
                motor_off()
                motor_start_time = None

            blinded = BLIND_WHILE_RECORDING and record_label is not None

            draw_live_ui(
                frame,
                {
                    "state": posture_state,
                    "detail": detail_text,
                    "score": posture_score,
                    "stats": stats,
                    "mode": (
                        "Random Forest"
                        if classifier.is_loaded
                        else "Calibrated rule"
                    ),
                    "confidence": confidence,
                    "features": feature_map,
                    "record_label": record_label,
                    "saved_rows": saved_rows,
                    "blinded": blinded,
                },
            )

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("1"):
                record_label = "NORMAL"
                print("[RECORD] NORMAL (바른 자세)")
            elif key == ord("2"):
                record_label = "WARNING"
                print("[RECORD] WARNING (살짝 거북목)")
            elif key == ord("3"):
                record_label = "BAD"
                print("[RECORD] BAD (심한 거북목)")
            elif key == ord("0"):
                record_label = None
                print(f"[RECORD] STOP (saved {saved_rows} rows)")
            elif key in {ord("c"), ord("C")}:
                record_label = None
                motor_off()
                set_led("OFF")
                classifier.reset()
                baseline = calibrate_baseline(camera, pose)
                state_machine.reset()
                stats.break_streak()
                smoothed = None
                last_alert_time = None
                previous_time = time.monotonic()
            elif key in {ord("q"), 27}:
                print("Monitoring stopped by user.")
                break

            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    print("Monitoring window closed.")
                    break
            except cv2.error:
                pass

        show_summary(stats)

    except (RuntimeError, ValueError) as application_error:
        print(f"ERROR: {application_error}")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        motor_off()
        set_led("OFF")

        if pose is not None:
            pose.close()

        if camera is not None:
            camera.close()

        GPIO.cleanup()
        cv2.destroyAllWindows()
        print("Posture program closed safely.")


if __name__ == "__main__":
    main()
