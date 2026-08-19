#!/usr/bin/env python3
"""
run_monitor.py - 라즈베리파이에서 자세 감지를 실행한다

    python3 run_monitor.py                # 학습된 모델 사용
    python3 run_monitor.py --model threshold   # 임계값 폴백 강제
    python3 run_monitor.py --headless     # 화면 없이 실행

모델 파일이 없거나 깨졌으면 자동으로 임계값 방식으로 내려간다.
시연 중 모델 문제로 전체가 죽는 것을 막기 위한 장치다.
"""

import argparse
import sys

from posture.app import run


def main():
    parser = argparse.ArgumentParser(description="거북목 실시간 감지")
    parser.add_argument(
        "--model", default="rf", choices=["rf", "threshold"],
        help="rf=학습된 RandomForest(기본), threshold=임계값 폴백",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="화면 없이 실행. DISPLAY가 없으면 자동으로 켜진다",
    )
    args = parser.parse_args()
    return run(model=args.model, headless=args.headless or None)


if __name__ == "__main__":
    sys.exit(main())
