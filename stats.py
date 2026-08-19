#!/usr/bin/env python3
"""
app/stats.py - 세션 통계와 자세 점수

V12.2에서는 이 계산이 메인 루프 안 지역변수로 흩어져 있었다.
따로 떼어내면 값 검산이 쉬워지고, 나중에 리포트 기능을 붙이기도 편하다.

────────────────────────────────────────────────────────────
점수를 축 기준으로 정규화하는 이유

V12.2는 `100 - posture_error * 700` 이었다.
posture_error는 비율(대략 0~0.4)이라 700이 맞는 계수였다.

그런데 지금은 모델이 cva_error(도, 0~40)나 torso_error(도)를 고를 수 있다.
같은 계수를 쓰면 5도만 틀어져도 점수가 0이 된다.

그래서 manifest의 threshold_hint를 기준으로 삼는다.
  bad_enter에 도달 → 0점
  0(기준자세)      → 100점
축이 무엇이든 같은 의미의 눈금이 된다.
────────────────────────────────────────────────────────────
"""

from ..contract import SCORE_MAX

__all__ = ["PostureScorer", "SessionStats"]


class PostureScorer:
    """오차 → 0~100 점. 판정에는 쓰이지 않고 표시 전용이다."""

    def __init__(self, manifest=None):
        self.axis = "posture_error"
        self.sign = 1.0
        self.zero_at = 0.32          # 이 오차에서 0점
        self.source = "기본값"

        hint = (manifest or {}).get("threshold_hint")
        if isinstance(hint, dict):
            if isinstance(hint.get("axis"), str):
                self.axis = hint["axis"]
            if isinstance(hint.get("sign"), (int, float)):
                self.sign = float(hint["sign"])
            bad_enter = hint.get("bad_enter")
            if isinstance(bad_enter, (int, float)) and abs(float(bad_enter)) > 1e-9:
                self.zero_at = abs(float(bad_enter))
                self.source = "manifest"

    def score(self, info):
        """
        info는 PostureJudge.decide()가 돌려준 dict.
        축 값을 못 구하면 None을 돌려주고, UI는 '--'로 표시한다.
        """
        value = info.get(self.axis)
        if value is None:
            # judge가 항상 담아주는 posture_error로 대체한다.
            value = info.get("posture_error")
        if value is None:
            return None

        error = max(float(value) * self.sign, 0.0)
        ratio = min(error / self.zero_at, 1.0)
        return int(round(SCORE_MAX * (1.0 - ratio)))

    def describe(self):
        return f"score(axis={self.axis}, 0점={self.zero_at:.3f}, {self.source})"


class SessionStats:
    """감시 시간과 경고/불량 누적 시간."""

    def __init__(self):
        self.monitored = 0.0
        self.warning = 0.0
        self.bad = 0.0
        self._score_sum = 0.0
        self._score_count = 0

    def add(self, status, elapsed, score=None):
        """
        elapsed는 직전 프레임과의 간격이다.
        NO_POSE와 PAUSED는 감시 시간에 넣지 않는다.
        사람이 자리에 없던 시간까지 '좋은 자세'로 세면 비율이 왜곡된다.
        """
        if status in ("NO_POSE", "PAUSED"):
            return

        self.monitored += elapsed
        if status == "WARNING":
            self.warning += elapsed
        elif status == "BAD":
            self.bad += elapsed

        if score is not None:
            self._score_sum += score
            self._score_count += 1

    @property
    def good_ratio(self):
        if self.monitored <= 0:
            return 100.0
        good = self.monitored - self.warning - self.bad
        return max(0.0, good / self.monitored * 100.0)

    @property
    def average_score(self):
        if self._score_count == 0:
            return 0.0
        return self._score_sum / self._score_count

    def reset(self):
        self.__init__()
