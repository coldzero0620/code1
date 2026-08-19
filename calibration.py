#!/usr/bin/env python3
"""
app/calibration.py - 개인 기준자세 측정

3초 동안 바른 자세를 유지시키고 그 구간의 중앙값을 baseline으로 잡는다.

계산 자체는 features.BaselineCalibrator가 한다.
학습 데이터 수집(tools/collect_data.py)도 같은 클래스를 쓰므로,
기준점을 잡는 방식이 학습과 실기에서 동일하다.

모듈화 전에는 런타임이 자체 캘리브레이션을 갖고 있었고
2D 값 하나(ear_shoulder_ratio 중앙값)만 잡았다. 3D 특징을 쓰는
모델에는 기준점이 아예 없었다는 뜻이다. 이제 네 축을 모두 잡는다.

여기서는 그 위에 UI와 재시도 정책만 얹는다.
"""

import time

import cv2

from ..contract import (
    CALIBRATION_MAX_SPREAD,
    CALIBRATION_MIN_SAMPLES,
    CALIBRATION_MIN_WORLD_RATIO,
    CALIBRATION_SEC,
)
from ..features import BaselineCalibrator, draw_feature, extract_feature, process_pose

__all__ = ["CalibrationError", "calibrate"]


class CalibrationError(RuntimeError):
    """카메라는 멀쩡하지만 기준자세를 못 잡았다. 재시도 대상이다."""


def _banner(frame, lines):
    y = 55
    for text, scale, color, thickness in lines:
        cv2.putText(
            frame, text, (35, y),
            cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
        )
        y += 34


def calibrate(camera, pose, state, ui, stop_event):
    """
    (baseline, ui_visible)을 돌려준다.

    baseline은 dict다. signed_delta / fwd_ratio / cva_deg / torso_angle_deg
    네 축의 중앙값이 들어 있다.

    일시정지가 걸리면 모은 표본을 버리고 처음부터 다시 잰다.
    정지 전후로 자세가 달라졌을 수 있으므로 이어 붙이면 안 된다.
    """
    print("=== 개인 기준자세 측정 ===")
    print(f"편안하고 바른 자세로 {CALIBRATION_SEC:.0f}초간 있어 주세요.")

    calibrator = BaselineCalibrator(
        seconds=CALIBRATION_SEC, min_samples=CALIBRATION_MIN_SAMPLES
    )
    previous_frame_id = -1
    was_paused = False
    started = False

    while True:
        if stop_event.is_set():
            raise KeyboardInterrupt

        frame_id, frame, _ = camera.get_latest(previous_frame_id)
        if frame is None:
            if camera.is_fatally_stalled():
                raise RuntimeError("캘리브레이션 중 카메라가 프레임을 주지 않았습니다.")
            continue
        previous_frame_id = frame_id

        # ── 일시정지 ──
        if state.is_paused():
            if not was_paused:
                calibrator = BaselineCalibrator(
                    seconds=CALIBRATION_SEC, min_samples=CALIBRATION_MIN_SAMPLES
                )
                started = False
                was_paused = True
                print("[CALIBRATION] GPIO23/UI로 일시정지. "
                      "라즈베리파이와 VNC는 계속 켜져 있습니다.")

            _banner(frame, [
                ("PAUSED - GPIO23를 ACTIVE로", 0.65, (0, 215, 255), 2),
            ])
            ui.show(frame)
            continue

        if was_paused:
            was_paused = False
            print("[CALIBRATION] 재개. 3초 측정을 처음부터 다시 합니다.")

        # ── 측정 ──
        now = time.monotonic()
        feature = extract_feature(process_pose(pose, frame))

        if feature is not None and not started:
            # 사람이 잡힌 순간부터 시간을 센다.
            calibrator.start(now)
            started = True

        if started:
            running = calibrator.feed(feature, now)
        else:
            running = True

        draw_feature(frame, feature)

        if started:
            remaining = calibrator.remaining(now)
            message = "바른 자세를 유지하세요"
        else:
            remaining = CALIBRATION_SEC
            message = "옆모습이 보이도록 앉아 주세요"

        _banner(frame, [
            (f"CALIBRATING: {remaining:.1f}s", 0.8, (0, 255, 255), 2),
            (message, 0.6, (255, 255, 255), 1),
            (f"TRACKING: {feature['side'] if feature else '-'}",
             0.6, (255, 255, 255), 1),
        ])
        ui.show(frame)

        if started and not running:
            break

    # ── 품질 검사 ──
    baseline = calibrator.baseline
    if baseline is None:
        raise CalibrationError(
            f"유효한 표본이 {calibrator.sample_count}개뿐입니다. "
            f"{CALIBRATION_MIN_SAMPLES}개 이상 필요합니다."
        )

    spread = calibrator.spread
    if spread is not None and spread > CALIBRATION_MAX_SPREAD:
        raise CalibrationError(
            f"측정 중 자세가 흔들렸습니다 (IQR {spread:.4f} > "
            f"{CALIBRATION_MAX_SPREAD:.4f}). 가만히 있어 주세요."
        )

    print(
        f"[CALIBRATION] signed_delta={baseline['signed_delta']:+.4f}  "
        f"fwd_ratio={baseline['fwd_ratio']:+.4f}  "
        f"cva={baseline['cva_deg']:+.2f}도  "
        f"torso={baseline['torso_angle_deg']:+.2f}도  "
        f"(IQR {spread:.4f})"
    )

    world_ratio = calibrator.world_ratio
    print(f"[CALIBRATION] 3D 좌표 유효 비율: {world_ratio * 100:.1f}%")
    if world_ratio < CALIBRATION_MIN_WORLD_RATIO:
        # 여기서 실패로 처리하지는 않는다. 2D 특징만 쓰는 모델이면 문제없다.
        print("[WARNING] 3D 좌표가 자주 실패합니다. 3D 특징을 쓰는 모델이라면 "
              "상반신 전체가 프레임에 들어오도록 조정하세요.")

    return baseline
