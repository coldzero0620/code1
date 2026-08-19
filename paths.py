#!/usr/bin/env python3
"""
paths.py - 파일 위치의 단일 출처

모듈화 전에는 모델 경로가 세 곳에, 데이터셋 경로가 네 곳에 따로 적혀 있었다.
한 곳을 옮기면 나머지가 조용히 옛 파일을 읽는 사고가 난다.

경로는 전부 패키지 위치에서 계산한다. 따라서 어느 디렉터리에서
실행하든 같은 파일을 가리킨다.

    posture-project/
    ├── posture/          이 패키지
    ├── tools/            학습 도구
    ├── data/             posture_dataset.csv
    ├── models/           posture-rf.joblib, split_manifest.json
    └── videos/           학습용 영상
"""

from pathlib import Path

__all__ = [
    "ROOT_DIR", "DATA_DIR", "MODEL_DIR", "VIDEO_DIR",
    "DATASET_PATH", "MODEL_PATH", "SPLIT_PATH",
    "ensure_dirs",
]

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
VIDEO_DIR = ROOT_DIR / "videos"

DATASET_PATH = DATA_DIR / "posture_dataset.csv"
MODEL_PATH = MODEL_DIR / "posture-rf.joblib"
SPLIT_PATH = MODEL_DIR / "split_manifest.json"


def ensure_dirs():
    """쓰기 전에 부르면 된다. 이미 있으면 아무 일도 하지 않는다."""
    for directory in (DATA_DIR, MODEL_DIR, VIDEO_DIR):
        directory.mkdir(parents=True, exist_ok=True)
