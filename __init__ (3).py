"""
posture.judge - 특징에서 상태를 만들어내는 계층

하드웨어를 전혀 모른다. 카메라도 GPIO도 BLE도 import하지 않는다.
그래서 노트북에서 합성 입력만으로 전체를 시험할 수 있다.

    from posture.judge import build_judge, BandLink

    judge = build_judge("rf")
    status, info = judge.decide(feature, baseline, now)

    link = BandLink(send_fn=...)
    command = link.update(status)
"""

from .band import BandLink
from .classifiers import (
    MODEL_PATH,
    SPLIT_PATH,
    RandomForestPostureClassifier,
    ThresholdClassifier,
    load_manifest,
)
from .judge import PostureJudge, build_judge
from .stabilizer import StatusStabilizer

__all__ = [
    "BandLink",
    "PostureJudge",
    "RandomForestPostureClassifier",
    "StatusStabilizer",
    "ThresholdClassifier",
    "build_judge",
    "load_manifest",
    "MODEL_PATH",
    "SPLIT_PATH",
]
