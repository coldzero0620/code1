#!/usr/bin/env python3
"""
judge/stabilizer.py - 프레임 단위 판정을 상태로 안정화한다

분류기는 프레임마다 독립적으로 답한다. 그대로 쓰면 경계 근처에서
NORMAL/WARNING이 초당 여러 번 뒤집히고, 진동 밴드가 계속 울린다.

여기서 두 단계로 누른다.
  1단 확률 평균   최근 N프레임 확률벡터 평균 → argmax
  2단 유지 시간   후보가 일정 시간 유지돼야 실제 전환

악화 방향은 짧게(0.6초), 완화 방향은 길게(2.0초) 잡는다.
경고를 놓치는 비용이 헛경고 비용보다 크기 때문이다.
"""

from collections import deque
from typing import Deque, Optional

import numpy as np

from ..contract import HOLD_ESCALATE_SEC, HOLD_RELAX_SEC, POSTURE_LABELS, PROBA_WINDOW, SEVERITY


class StatusStabilizer:
    """
    2단 구성.
      1단 확률 평균  최근 N프레임 확률벡터의 평균 → argmax
      2단 유지시간    후보가 일정 시간 유지돼야 실제로 전환.
                     악화 방향은 짧게, 완화 방향은 길게.

    NO_POSE는 안정화를 거치지 않는다. 포즈가 사라지면 즉시 반영한다.
    """

    def __init__(
        self,
        escalate_sec: float = HOLD_ESCALATE_SEC,
        relax_sec: float = HOLD_RELAX_SEC,
        window: int = PROBA_WINDOW,
    ):
        self.escalate_sec = escalate_sec
        self.relax_sec = relax_sec
        self.window = max(1, window)
        self.status = "NORMAL"
        self._probas: Deque[np.ndarray] = deque(maxlen=self.window)
        self._candidate = "NORMAL"
        self._since = 0.0

    def reset(self, status: str = "NORMAL", now: float = 0.0) -> None:
        self.status = status
        self._probas.clear()
        self._candidate = status
        self._since = now

    def update(self, proba: Optional[np.ndarray], now: float) -> str:
        if proba is None:  # NO_POSE
            self._probas.clear()
            self.status = "NO_POSE"
            self._candidate = "NO_POSE"
            self._since = now
            return self.status

        if self.status == "NO_POSE":  # 복귀 시 지연 없이 즉시 채택
            self._probas.append(proba)
            label = POSTURE_LABELS[int(np.argmax(proba))]
            self.reset(label, now)
            return self.status

        # 시계가 뒤로 갔다면(재시작·수동 조정) 기준 시각을 당겨 교착을 막는다.
        if now < self._since:
            self._since = now

        self._probas.append(proba)
        averaged = np.mean(np.stack(self._probas), axis=0)
        voted = POSTURE_LABELS[int(np.argmax(averaged))]

        if voted != self._candidate:
            self._candidate = voted
            self._since = now
            return self.status

        if voted == self.status:
            return self.status

        escalating = SEVERITY[voted] > SEVERITY[self.status]
        hold = self.escalate_sec if escalating else self.relax_sec
        if now - self._since >= hold:
            self.status = voted

        return self.status
