"""
posture.app - 실행 계층

judge(판정)와 hardware(주변장치)를 묶어 실제 프로그램으로 만든다.
스레드 간 상태 공유, 캘리브레이션 진행, 화면, 키 입력, 종료 처리가 여기 있다.

    from posture.app import run
    run(model="rf")
"""

from .calibration import CalibrationError, calibrate
from .monitor import PostureMonitor, run
from .stats import PostureScorer, SessionStats
from .state import RuntimeState
from .ui import MonitorUI

__all__ = [
    "CalibrationError",
    "MonitorUI",
    "PostureMonitor",
    "PostureScorer",
    "RuntimeState",
    "SessionStats",
    "calibrate",
    "run",
]
