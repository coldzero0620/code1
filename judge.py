#!/usr/bin/env python3
"""
judge/judge.py - NO_POSE 선처리 + 분류 + 안정화 + 지속시간 악화

    judge = build_judge("rf")
    status, info = judge.decide(feature, baseline, now)

status는 NO_POSE / NORMAL / WARNING / BAD 네 가지다.
NO_POSE는 모델 추론 이전에 결정되며 안정화를 거치지 않는다.
포즈가 사라졌는데 이전 상태를 붙들고 있으면 진동이 계속되기 때문이다.
"""

import time
from typing import Dict, Optional, Tuple

import numpy as np

from ..contract import (
    HOLD_ESCALATE_SEC,
    HOLD_RELAX_SEC,
    POSTURE_LABELS,
    WARNING_TO_BAD_SEC,
)
from ..features import build_feature_row
from .classifiers import RandomForestPostureClassifier, ThresholdClassifier
from .stabilizer import StatusStabilizer


class PostureJudge:
    """
    특징 → 상태. NO_POSE 선처리 + 판정기 + 안정화를 하나로 묶는다.

        judge = build_judge("rf")
        status, info = judge.decide(feature, baseline, now)
    """

    def __init__(
        self,
        classifier,
        escalate_sec=HOLD_ESCALATE_SEC,
        relax_sec=HOLD_RELAX_SEC,
        warning_to_bad_sec=WARNING_TO_BAD_SEC,
    ):
        self.classifier = classifier
        self.stabilizer = StatusStabilizer(escalate_sec=escalate_sec, relax_sec=relax_sec)
        self.last_raw: Optional[str] = None
        self.last_confidence: Optional[float] = None

        # 지속시간 기반 악화.
        # 분류기가 WARNING만 계속 말하더라도 그 상태가 오래 유지되면 BAD로 올린다.
        # V12.2부터 있던 동작이며, 사용자에게는
        # "계속 나쁜 자세면 결국 강하게 알린다"로 보인다.
        # 안정화(StatusStabilizer)와는 다른 축이다. 안정화는 "분류기 판정이
        # 흔들리지 않는가"를 보고, 이쪽은 "같은 경고가 얼마나 오래 갔는가"를 본다.
        self.warning_to_bad_sec = warning_to_bad_sec
        self._warning_since: Optional[float] = None

    def decide(
        self,
        feature: Optional[Dict],
        baseline: Optional[float],
        now: Optional[float] = None,
    ) -> Tuple[str, Dict]:
        if now is None:
            now = time.monotonic()

        # 1) NO_POSE는 모델 추론 이전에 결정한다
        if feature is None or baseline is None:  # baseline은 dict 또는 float
            status = self.stabilizer.update(None, now)
            self.last_raw, self.last_confidence = None, None
            return status, {
                "raw": None,
                "confidence": None,
                "posture_error": None,
                "warning_held_sec": 0.0,
                "reason": "no_pose" if feature is None else "no_baseline",
            }

        # 2) 특징 계산 → 판정 (프레임당 추론 1회)
        row = build_feature_row(feature, baseline)

        # 실제로 쓰는 특징만 검사한다.
        # 3D 특징을 안 쓰는 모델이라면 fwd_error가 NaN이어도 문제없다.
        used = getattr(self.classifier, "feature_columns", list(row.keys()))
        if not all(np.isfinite(row[c]) for c in used if c in row):
            status = self.stabilizer.update(None, now)
            self.last_raw, self.last_confidence = None, None
            return status, {
                "raw": None,
                "confidence": None,
                "posture_error": None,
                "warning_held_sec": 0.0,
                "reason": "invalid_feature",
            }

        proba = self.classifier.predict(row)
        raw = POSTURE_LABELS[int(np.argmax(proba))]

        # 3) 안정화
        status = self.stabilizer.update(proba, now)

        # 4) 지속시간 기반 악화
        status, held = self._apply_duration_escalation(status, now)

        self.last_raw = raw
        self.last_confidence = float(np.max(proba))

        # 계산된 특징을 전부 실어 보낸다.
        # 점수 표시는 manifest가 고른 축(cva_error 등)을 읽어야 하는데,
        # posture_error만 넘기면 축 단위가 달라 엉뚱한 점수가 나온다.
        info = dict(row)
        info.update({
            "raw": raw,
            "confidence": self.last_confidence,
            "side": feature.get("side"),
            "warning_held_sec": held,
            "reason": self.classifier.name,
        })
        return status, info

    def _apply_duration_escalation(self, status: str, now: float):
        """
        WARNING이 warning_to_bad_sec 이상 이어지면 BAD로 올린다.
        반환값은 (최종 상태, 현재 WARNING 지속 시간).
        """
        if self.warning_to_bad_sec is None:
            return status, 0.0

        if status != "WARNING":
            self._warning_since = None
            return status, 0.0

        if self._warning_since is None:
            self._warning_since = now
            return status, 0.0

        # 시계가 뒤로 갔다면 기준을 당겨 교착을 막는다.
        if now < self._warning_since:
            self._warning_since = now

        held = now - self._warning_since
        if held >= self.warning_to_bad_sec:
            return "BAD", held
        return status, held

    def reset(self, now: float = 0.0) -> None:
        """세션 재시작. 판정기 내부 히스테리시스까지 함께 초기화한다."""
        self.classifier.reset()
        self.stabilizer.reset("NORMAL", now)
        self._warning_since = None

    def describe(self) -> str:
        escalation = (
            f" warning→bad {self.warning_to_bad_sec:.1f}s"
            if self.warning_to_bad_sec is not None else " warning→bad off"
        )
        return self.classifier.describe() + escalation


def build_judge(model: str = "rf", **kw) -> PostureJudge:
    """
    model='rf'면 RandomForest, 로드에 실패하면 threshold로 자동 폴백한다.
    시연 중 모델 파일 문제로 전체가 죽는 것을 막는다.
    """
    hold_kw = {
        k: kw.pop(k)
        for k in ("escalate_sec", "relax_sec", "warning_to_bad_sec")
        if k in kw
    }

    if model == "rf":
        try:
            judge = PostureJudge(RandomForestPostureClassifier(), **hold_kw)
            print(f"[JUDGE] {judge.describe()}")
            return judge
        except Exception as error:
            print(f"[JUDGE] RandomForest 로드 실패 ({error}) → threshold로 폴백")

    judge = PostureJudge(ThresholdClassifier(**kw), **hold_kw)
    print(f"[JUDGE] {judge.describe()}")
    return judge
