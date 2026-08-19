#!/usr/bin/env python3
"""
measure_range.py — 영상 하나로 그 사람의 자세 범위를 측정한다

촬영 현장에서 쓰는 도구다. 본 촬영 전에 아래 영상을 하나 찍고 돌린다.

    바른 자세 5초 유지
      → 천천히 거북목 끝까지 (5초에 걸쳐)
      → 끝에서 5초 유지
      → 천천히 원위치

    python3 measure_range.py ../videos/s01_range.mp4

그러면 이렇게 나온다.

    NORMAL   cva -12.3도
    WARNING  cva -29.5도   ← 이 자세를 취하게 하면 된다
    BAD      cva -46.8도

WARNING은 NORMAL과 BAD의 중간이다.
말로 하면 "BAD로 갈 때 목을 빼는 거리의 딱 절반".

시간축 그래프도 함께 출력하므로, 본 촬영 후 이 도구로 다시 확인하면
WARNING이 정말 중간에 있는지 그 자리에서 알 수 있다.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np

# 프로젝트 루트를 import 경로에 넣는다.
# tools/는 패키지가 아니라 실행 스크립트 모음이므로, 어디서 실행하든
# posture 패키지를 찾을 수 있어야 한다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posture.contract import FRAME_HEIGHT, FRAME_WIDTH, MODEL_COMPLEXITY
from posture.features import create_pose, extract_feature, process_pose

BAR_WIDTH = 44


def scan(path: Path, stride: int):
    pose = create_pose()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 240:
        fps = 30.0

    records = []
    total = 0
    detected = 0
    world = 0
    try:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            index += 1
            if index % stride:
                continue
            total += 1

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            feature = extract_feature(process_pose(pose, frame))
            if feature is None:
                continue
            detected += 1
            has_world = bool(feature.get("world_ok"))
            if has_world:
                world += 1
            records.append({
                "t": index / fps,
                "cva": feature.get("cva_deg", float("nan")),
                "fwd": feature.get("fwd_ratio", float("nan")),
                "sd": feature["signed_delta"],
                "obl": feature["obliquity"],
                "world": has_world,
            })
    finally:
        capture.release()
        pose.close()

    return records, total, detected, world


def draw_series(records, key, label, unit=""):
    values = np.array([r[key] for r in records], dtype=np.float64)
    times = np.array([r["t"] for r in records], dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        print(f"  {label}: 값이 없습니다.")
        return None, None

    values, times = values[finite], times[finite]
    low, high = values.min(), values.max()
    span = max(high - low, 1e-9)

    print(f"\n시간축 {label} (막대가 길수록 목이 앞으로)")
    step = max(1, len(values) // 40)
    for i in range(0, len(values), step):
        bar = int((values[i] - low) / span * BAR_WIDTH)
        print(f"  {times[i]:6.2f}s {values[i]:+8.2f}{unit} |{'#' * bar}")
    return values, times


def summarize(values, name, unit):
    """
    분포 양끝을 자세 범위로 본다.
    중앙값이 아니라 백분위를 쓰는 이유는, 전환 구간이 섞여 있어도
    양극단이 실제 NORMAL / BAD에 해당하기 때문이다.
    """
    good = float(np.percentile(values, 90))   # 덜 숙인 쪽
    bad = float(np.percentile(values, 10))    # 많이 숙인 쪽
    if good < bad:
        good, bad = bad, good
    middle = (good + bad) / 2.0
    gap = abs(good - bad)

    print(f"\n{name} 기준 자세 범위")
    print(f"  NORMAL   {good:+8.2f}{unit}")
    print(f"  WARNING  {middle:+8.2f}{unit}   ← 이 자세를 취하게 한다")
    print(f"  BAD      {bad:+8.2f}{unit}")
    print(f"  간격     {gap:8.2f}{unit}")
    return good, middle, bad, gap


def main():
    parser = argparse.ArgumentParser(
        description="영상 하나로 자세 범위를 측정한다"
    )
    parser.add_argument("video", help="측정할 영상 경로")
    parser.add_argument("--stride", type=int, default=2,
                        help="몇 프레임마다 볼지 (기본 2)")
    parser.add_argument("--no-graph", action="store_true",
                        help="시간축 그래프를 생략한다")
    args = parser.parse_args()

    path = Path(args.video).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"영상이 없습니다: {path}")

    print(f"영상: {path.name}   model_complexity={MODEL_COMPLEXITY}")
    records, total, detected, world = scan(path, args.stride)

    if not records:
        print("[ERROR] 포즈가 한 번도 검출되지 않았습니다.")
        print("        사람이 화면에 크게 나오는지, 조명이 충분한지 확인하세요.")
        return 1

    print(f"샘플 {total}프레임   포즈 검출 {detected / total * 100:.1f}%   "
          f"3D 유효 {world / max(detected, 1) * 100:.1f}%")

    obliquity = float(np.median([r["obl"] for r in records]))
    print(f"obliquity 중앙값 {obliquity:.3f}  ", end="")
    if obliquity <= 0.45:
        print("(정측면에 가깝다 → view=side)")
    elif obliquity <= 0.9:
        print("(비스듬하다 → view=oblique)")
    else:
        print("(정면에 가깝다 → 2D 특징을 신뢰하기 어렵다)")

    use_3d = world >= detected * 0.9
    if use_3d:
        if not args.no_graph:
            draw_series(records, "cva", "cva_deg", "도")
        values = np.array([r["cva"] for r in records])
        summarize(values[np.isfinite(values)], "cva_deg", "도")
        print("\n  3D 기준이므로 카메라 각도가 바뀌어도 같은 값이 나온다.")
    else:
        print(f"\n[WARNING] 3D 유효율 {world / max(detected, 1) * 100:.1f}%. "
              "골반이 프레임에 안 들어왔습니다.")
        print("          2D 기준으로 대신 측정합니다. 카메라 각도가 바뀌면 값이 달라집니다.")
        if not args.no_graph:
            draw_series(records, "sd", "signed_delta")
        values = np.array([r["sd"] for r in records])
        summarize(values[np.isfinite(values)], "signed_delta", "")

    print("\n말로 지시할 때")
    print("  WARNING = 바른 자세에서 거북목 끝까지 가는 거리의 딱 절반")
    print("  촬영 순서는 NORMAL → BAD → WARNING 을 권한다.")
    print("  BAD를 먼저 해봐야 절반의 감이 잡힌다.")
    print("  목만 움직이고 어깨와 골반 방향은 고정할 것.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("[PROGRAM] 사용자가 중단했습니다.")
        sys.exit(1)
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}")
        sys.exit(2)
