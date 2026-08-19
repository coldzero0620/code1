#!/usr/bin/env python3
"""
features.py - 영상 프레임에서 자세 특징을 뽑는다

수집(collect_data)과 런타임(posture_runtime)이 반드시 이 파일을 거친다.
양쪽이 같은 계산식을 쓰도록 강제하는 것이 이 파일의 존재 이유다.

만들어내는 특징은 두 계열이다.

  2D 투영 계열   signed_delta, abs_delta
      귀-어깨의 화면상 수평 offset을 세로 거리로 나눈 값.
      계산이 가볍고 정측면에서는 정확하지만,
      카메라가 축을 벗어나면 급격히 무너진다.

  3D 신체 계열   fwd_ratio, cva_deg
      pose_world_landmarks로 몸에 붙은 좌표계를 세워 계산한다.
      카메라 위치와 무관하지만 z 추정 노이즈에 취약해
      MODEL_COMPLEXITY=1 이상이 필요하다.

  시점 기술자    obliquity
      0에 가까우면 정측면, 커질수록 정면.

어떤 조합을 실제로 쓸지는 train_model.py가 교차검증으로 정하고
manifest에 기록한다. 이 파일은 가능한 것을 전부 만들어둘 뿐이다.
"""

from typing import Dict, List, Optional

import cv2
import numpy as np

# 계약 상수는 contract.py가 단일 출처다.
# 아래는 기존 import 경로 호환을 위한 재수출이며, 이 파일에서 직접 쓰지 않는 것도 있다.
from .contract import (
    CAMERA_FPS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    INFERENCE_HEIGHT,
    INFERENCE_WIDTH,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MODEL_COMPLEXITY,
    OBLIQUITY_2D_LIMIT,
    POSTURE_LABELS,
    SMOOTH_LANDMARKS,
    VISIBILITY_THRESHOLD,
)

__all__ = [
    # 재수출 상수
    "CAMERA_FPS", "FRAME_HEIGHT", "FRAME_WIDTH",
    "INFERENCE_HEIGHT", "INFERENCE_WIDTH",
    "MIN_DETECTION_CONFIDENCE", "MIN_TRACKING_CONFIDENCE",
    "MODEL_COMPLEXITY", "OBLIQUITY_2D_LIMIT", "POSTURE_LABELS",
    "SMOOTH_LANDMARKS", "VISIBILITY_THRESHOLD",
    # 함수
    "create_pose", "process_pose", "extract_feature", "view_is_lateral",
    "build_feature_row", "to_vector", "draw_feature", "BaselineCalibrator",
]

try:
    import mediapipe as mp
except ImportError:  # 학습/평가 머신에는 mediapipe가 없을 수 있다
    mp = None


def create_pose():
    """MediaPipe Pose 인스턴스. 수집과 런타임이 반드시 이 함수를 쓴다."""
    if mp is None:
        raise RuntimeError("mediapipe가 설치되지 않았습니다.")
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        smooth_landmarks=SMOOTH_LANDMARKS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )


def process_pose(pose, frame: np.ndarray):
    """BGR 프레임을 추론 해상도로 줄여 MediaPipe에 통과시킨다."""
    inference_frame = cv2.resize(
        frame, (INFERENCE_WIDTH, INFERENCE_HEIGHT), interpolation=cv2.INTER_AREA
    )
    rgb_frame = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    return pose.process(rgb_frame)


def _body_frame(world, image_landmarks=None) -> Optional[Dict[str, float]]:
    """
    pose_world_landmarks(3D, 미터, 골반 원점)로 신체 자체 좌표계를 세운다.

        lateral = 오른어깨 - 왼어깨
        up      = 어깨중점 - 골반중점   (lateral에 직교화)
        forward = lateral x up

    이 세 축은 사람 몸에 붙어 있으므로 카메라가 어디 있든 동일하다.
    귀-어깨 벡터를 forward에 투영하면 카메라 각도와 무관한 전방 이동량이 나온다.

    다만 단일 카메라의 z 추정은 노이즈가 크다. MODEL_COMPLEXITY=0에서는
    이 특징을 신뢰하기 어렵다. contract.py의 주석을 참고할 것.
    """
    if world is None:
        return None

    pl = mp.solutions.pose.PoseLandmark

    # MediaPipe는 화면에 안 보이는 관절도 "추정"해서 좌표를 내놓는다.
    # 좌표만 보고 판단하면 상상해낸 골반 위치로 축을 세우게 되고,
    # 3D 특징이 조용히 엉터리가 된다. 반드시 visibility를 확인해야 한다.
    if image_landmarks is not None:
        marks = image_landmarks.landmark
        needed = (
            pl.LEFT_HIP.value, pl.RIGHT_HIP.value,
            pl.LEFT_SHOULDER.value, pl.RIGHT_SHOULDER.value,
        )
        best_hip = max(
            marks[pl.LEFT_HIP.value].visibility,
            marks[pl.RIGHT_HIP.value].visibility,
        )
        if best_hip < VISIBILITY_THRESHOLD:
            return None
        if min(marks[i].visibility for i in needed[2:]) < VISIBILITY_THRESHOLD:
            return None

    lm = world.landmark

    def vec(index):
        point = lm[index]
        return np.array([point.x, point.y, point.z], dtype=np.float64)

    try:
        left_shoulder = vec(pl.LEFT_SHOULDER.value)
        right_shoulder = vec(pl.RIGHT_SHOULDER.value)
        left_hip = vec(pl.LEFT_HIP.value)
        right_hip = vec(pl.RIGHT_HIP.value)
        left_ear = vec(pl.LEFT_EAR.value)
        right_ear = vec(pl.RIGHT_EAR.value)
    except (IndexError, AttributeError):
        return None

    shoulder_mid = (left_shoulder + right_shoulder) / 2.0
    hip_mid = (left_hip + right_hip) / 2.0
    ear_mid = (left_ear + right_ear) / 2.0

    lateral = right_shoulder - left_shoulder
    lateral_norm = np.linalg.norm(lateral)
    torso = shoulder_mid - hip_mid
    torso_len = float(np.linalg.norm(torso))

    # 어깨나 몸통이 뭉개진 프레임은 축을 세울 수 없다
    if lateral_norm < 1e-6 or torso_len < 0.05:
        return None

    lateral = lateral / lateral_norm
    up = torso - lateral * np.dot(torso, lateral)
    up_norm = np.linalg.norm(up)
    if up_norm < 1e-6:
        return None
    up = up / up_norm

    forward = np.cross(lateral, up)
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        return None
    forward = forward / forward_norm

    offset = ear_mid - shoulder_mid
    forward_component = float(np.dot(offset, forward))
    up_component = float(np.dot(offset, up))

    fwd_ratio = forward_component / torso_len
    cva_deg = float(np.degrees(np.arctan2(forward_component, up_component)))

    # ── 몸통 자체의 기울기 ──────────────────────────────────
    #
    # 위의 cva_deg는 up 축을 몸통에서 유도했으므로 정의상 "몸통 기준 목 각도"다.
    # 몸 전체가 앞으로 기울면 목과 몸통이 같이 기울어 cva_deg는 거의 변하지 않는다.
    # 즉 "목만 뺐는가" 와 "몸 전체를 숙였는가" 를 구분할 수 없다.
    #
    # 그래서 몸통을 몸이 아닌 바깥 기준으로 다시 잰다.
    # pose_world_landmarks는 골반 원점에 이미지 축과 대략 정렬된 좌표계이므로
    # -y가 화면 위쪽이다. 이것을 수직 기준으로 삼는다.
    #
    # 주의: 이 축은 중력이 아니라 카메라 기준이다. 카메라를 기울이면
    # torso_angle_deg 자체가 통째로 이동한다. 따라서 절대값은 학습 후보로
    # 쓰지 않고, baseline을 뺀 torso_error만 쓴다.
    # 그러려면 baseline 촬영과 자세 촬영 사이에 카메라를 옮기면 안 된다.
    world_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)

    # 어깨선(lateral)에 직교하는 성분만 남겨 좌우 회전의 영향을 뺀다.
    up_ref = world_up - lateral * np.dot(world_up, lateral)
    up_ref_norm = np.linalg.norm(up_ref)
    if up_ref_norm < 1e-6:
        torso_angle_deg = float("nan")
    else:
        up_ref = up_ref / up_ref_norm
        forward_ref = np.cross(lateral, up_ref)
        forward_ref_norm = np.linalg.norm(forward_ref)
        if forward_ref_norm < 1e-6:
            torso_angle_deg = float("nan")
        else:
            forward_ref = forward_ref / forward_ref_norm
            # cva_deg와 같은 부호 규약: forward 쪽으로 기울면 양수
            torso_angle_deg = float(np.degrees(np.arctan2(
                float(np.dot(torso, forward_ref)),
                float(np.dot(torso, up_ref)),
            )))

    if not (np.isfinite(fwd_ratio) and np.isfinite(cva_deg)):
        return None

    return {
        "fwd_ratio": fwd_ratio,
        "cva_deg": cva_deg,
        "torso_angle_deg": torso_angle_deg,
    }


def extract_feature(results) -> Optional[Dict]:
    """
    2D 투영 특징 + 3D 신체 좌표계 특징 + 시점 기술자를 함께 계산한다.

    반환 None = NO_POSE (모델 추론 이전에 결정되는 상태)
    """
    if results.pose_landmarks is None:
        return None

    pose_landmark = mp.solutions.pose.PoseLandmark
    landmarks = results.pose_landmarks.landmark

    candidates = [
        (
            "LEFT",
            landmarks[pose_landmark.LEFT_EAR.value],
            landmarks[pose_landmark.LEFT_SHOULDER.value],
        ),
        (
            "RIGHT",
            landmarks[pose_landmark.RIGHT_EAR.value],
            landmarks[pose_landmark.RIGHT_SHOULDER.value],
        ),
    ]

    side, ear, shoulder = max(
        candidates, key=lambda item: min(item[1].visibility, item[2].visibility)
    )

    if ear.visibility < VISIBILITY_THRESHOLD or shoulder.visibility < VISIBILITY_THRESHOLD:
        return None

    # 얼굴이 향한 방향(+1 = 화면 오른쪽). 코가 안 보이면 귀 위치로 대체한다.
    nose = landmarks[pose_landmark.NOSE.value]
    if nose.visibility >= VISIBILITY_THRESHOLD:
        facing = 1.0 if nose.x >= shoulder.x else -1.0
    else:
        facing = 1.0 if ear.x >= shoulder.x else -1.0

    raw_delta = ear.x - shoulder.x
    y_height = max(abs(ear.y - shoulder.y), 0.001)

    # 시점 기술자: 두 어깨의 이미지상 좌우 간격 / 몸통 세로 길이.
    # 0에 가까우면 정측면, 커질수록 정면에 가깝다.
    left_shoulder_2d = landmarks[pose_landmark.LEFT_SHOULDER.value]
    right_shoulder_2d = landmarks[pose_landmark.RIGHT_SHOULDER.value]
    left_hip_2d = landmarks[pose_landmark.LEFT_HIP.value]
    right_hip_2d = landmarks[pose_landmark.RIGHT_HIP.value]

    shoulder_gap = abs(left_shoulder_2d.x - right_shoulder_2d.x)
    torso_height = abs(
        (left_shoulder_2d.y + right_shoulder_2d.y) / 2.0
        - (left_hip_2d.y + right_hip_2d.y) / 2.0
    )
    obliquity = shoulder_gap / max(torso_height, 1e-3)

    feature = {
        "side": side,
        "facing": float(facing),
        "signed_delta": float(facing * raw_delta / y_height),
        "abs_delta": float(abs(raw_delta) / y_height),
        "obliquity": float(obliquity),
        "ear": (float(ear.x), float(ear.y)),
        "shoulder": (float(shoulder.x), float(shoulder.y)),
    }

    # 같은 쪽 골반. 오버레이에서 몸통 선을 그리는 데 쓴다.
    # 특징 계산에는 쓰이지 않으므로 없어도 무방하다.
    same_side_hip = landmarks[
        pose_landmark.LEFT_HIP.value if side == "LEFT"
        else pose_landmark.RIGHT_HIP.value
    ]
    feature["hip"] = (
        (float(same_side_hip.x), float(same_side_hip.y))
        if same_side_hip.visibility >= VISIBILITY_THRESHOLD else None
    )

    # 3D 특징. 실패해도 2D 특징만으로 동작해야 하므로 None을 허용한다.
    body = _body_frame(
        getattr(results, "pose_world_landmarks", None),
        results.pose_landmarks,
    )
    if body is None:
        feature.update({"fwd_ratio": float("nan"), "cva_deg": float("nan"),
                        "torso_angle_deg": float("nan"), "world_ok": 0})
    else:
        feature.update(body)
        feature["world_ok"] = 1

    return feature


def view_is_lateral(feature: Optional[Dict]) -> bool:
    """2D 특징을 신뢰할 수 있는 시점인지 판단한다."""
    if feature is None:
        return False
    return float(feature.get("obliquity", 1.0)) <= OBLIQUITY_2D_LIMIT


def build_feature_row(feature: Dict, baseline) -> Dict[str, float]:
    """
    특징 dict + baseline → 이름-값 매핑.

    baseline은 두 가지 형태를 받는다.
      float            2D 전용 (구버전 호환)
      dict             {"signed_delta":..., "fwd_ratio":..., "cva_deg":...}

    contract.ALL_FEATURES를 전부 만든다. 어떤 것을 쓸지는 manifest가 정한다.
    """
    if isinstance(baseline, dict):
        base_2d = float(baseline.get("signed_delta", 0.0))
        base_fwd = float(baseline.get("fwd_ratio", 0.0))
        base_cva = float(baseline.get("cva_deg", 0.0))
        base_torso = float(baseline.get("torso_angle_deg", 0.0))
    else:
        base_2d = float(baseline)
        base_fwd = 0.0
        base_cva = 0.0
        base_torso = 0.0

    signed_delta = float(feature["signed_delta"])
    fwd_ratio = float(feature.get("fwd_ratio", float("nan")))
    cva_deg = float(feature.get("cva_deg", float("nan")))
    torso_deg = float(feature.get("torso_angle_deg", float("nan")))

    return {
        "signed_delta": signed_delta,
        "abs_delta": float(feature["abs_delta"]),
        "posture_error": signed_delta - base_2d,
        "fwd_ratio": fwd_ratio,
        "fwd_error": fwd_ratio - base_fwd,
        "cva_deg": cva_deg,
        "cva_error": cva_deg - base_cva,
        "torso_angle_deg": torso_deg,
        "torso_error": torso_deg - base_torso,
        "obliquity": float(feature.get("obliquity", 0.0)),
    }


def to_vector(row: Dict[str, float], columns: List[str]) -> np.ndarray:
    """manifest의 feature_columns 순서대로 (1, n) 벡터를 만든다."""
    missing = [c for c in columns if c not in row]
    if missing:
        raise KeyError(f"특징 누락: {missing}")
    return np.array([[row[c] for c in columns]], dtype=np.float32)


def draw_feature(frame: np.ndarray, feature: Optional[Dict]) -> None:
    """귀-어깨 선과 어깨-골반(몸통) 선을 프레임에 표시한다."""
    if feature is None:
        return

    height, width = frame.shape[:2]

    def to_pixel(point):
        return int(point[0] * width), int(point[1] * height)

    ear = to_pixel(feature["ear"])
    shoulder = to_pixel(feature["shoulder"])

    cv2.line(frame, ear, shoulder, (255, 255, 0), 2)
    cv2.circle(frame, ear, 6, (0, 0, 255), -1)
    cv2.circle(frame, shoulder, 6, (0, 255, 0), -1)

    hip = feature.get("hip")
    if hip is not None:
        hip_px = to_pixel(hip)
        cv2.line(frame, shoulder, hip_px, (255, 180, 0), 2)
        cv2.circle(frame, hip_px, 6, (255, 180, 0), -1)


class BaselineCalibrator:
    """
    바른 자세 구간의 중앙값을 baseline으로 삼는다.
    2D와 3D 특징 각각의 기준값을 함께 잡는다.
    중앙값이라 캘리브레이션 중 한두 프레임 튀는 것은 무시된다.
    """

    KEYS = ("signed_delta", "fwd_ratio", "cva_deg", "torso_angle_deg")

    def __init__(self, seconds: float = 3.0, min_samples: int = 10):
        self.seconds = seconds
        self.min_samples = min_samples
        self._samples: Dict[str, List[float]] = {k: [] for k in self.KEYS}
        self._start: Optional[float] = None
        self.baseline: Optional[Dict[str, float]] = None
        self.spread: Optional[float] = None
        self.world_ratio: float = 0.0

    def start(self, now: float) -> None:
        self._samples = {k: [] for k in self.KEYS}
        self._start = now
        self.baseline = None
        self.spread = None
        self.world_ratio = 0.0

    @property
    def running(self) -> bool:
        return self._start is not None

    @property
    def sample_count(self) -> int:
        return len(self._samples["signed_delta"])

    def feed(self, feature: Optional[Dict], now: float) -> bool:
        """진행 중이면 True, 완료되면 False."""
        if self._start is None:
            return False

        if feature is not None:
            for key in self.KEYS:
                value = feature.get(key)
                if value is not None and np.isfinite(value):
                    self._samples[key].append(float(value))

        if now - self._start >= self.seconds:
            count_2d = len(self._samples["signed_delta"])
            if count_2d >= self.min_samples:
                baseline = {}
                for key in self.KEYS:
                    values = self._samples[key]
                    baseline[key] = float(np.median(values)) if values else 0.0
                self.baseline = baseline

                # IQR - 캘리브레이션 중 자세가 흔들렸는지 판단하는 품질 지표
                values = np.asarray(self._samples["signed_delta"], dtype=np.float64)
                self.spread = float(
                    np.percentile(values, 75) - np.percentile(values, 25)
                )
                # 3D 좌표가 잡힌 비율. 낮으면 3D 특징을 못 쓴다.
                self.world_ratio = len(self._samples["fwd_ratio"]) / count_2d
            else:
                self.baseline = None
                self.spread = None
            self._start = None
            return False

        return True

    def remaining(self, now: float) -> float:
        if self._start is None:
            return 0.0
        return max(0.0, self.seconds - (now - self._start))
