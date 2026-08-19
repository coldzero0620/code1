#!/usr/bin/env python3
"""
hardware/switch.py - GPIO23 슬라이드 스위치

LOW  = 자세 기능 일시정지
HIGH = 자세 기능 활성 + 모든 일시정지 해제 (마스터 재개)

중요한 것은 이 스위치가 라즈베리파이 전원을 건드리지 않는다는 점이다.
OS와 VNC는 어느 위치에서도 계속 살아 있다. 스위치를 껐다고 Pi가
종료되면 다시 켜기 위해 전원을 뽑아야 하고, 그건 SD 카드에 좋지 않다.

기계식 스위치는 접점이 튄다. 그래서 같은 값이
MODE_SWITCH_DEBOUNCE_SEC 동안 유지될 때만 반영한다.
"""

import threading
import time

from ..contract import (
    MODE_SWITCH_DEBOUNCE_SEC,
    MODE_SWITCH_PIN,
    MODE_SWITCH_POLL_SEC,
    MODE_SWITCH_STARTUP_GRACE_SEC,
)
from .gpio import GPIO, GPIO_AVAILABLE, setup_input_pullup

__all__ = ["ModeSwitchMonitor"]


class ModeSwitchMonitor:
    """
    on_change(paused: bool) 콜백으로 상태 변화를 알린다.
    앱 계층이 이 콜백에서 RuntimeState를 갱신한다.
    """

    def __init__(self, on_change, pin=MODE_SWITCH_PIN):
        self._on_change = on_change
        self._pin = pin
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if not GPIO_AVAILABLE:
            print(f"[MODE] 시뮬레이션 모드라 GPIO{self._pin} 스위치를 쓰지 않습니다.")
            return
        if self._thread is not None and self._thread.is_alive():
            return

        setup_input_pullup(self._pin)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="mode-switch", daemon=True
        )
        self._thread.start()
        print(f"[MODE] GPIO{self._pin} 스위치 작동: "
              "LOW=일시정지, HIGH=강제재개. OS/VNC는 계속 켜져 있습니다.")

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def read_now(self):
        """현재 스위치가 일시정지 위치인지. 시작 시 초기 상태 확인용."""
        if not GPIO_AVAILABLE:
            return False
        try:
            return GPIO.input(self._pin) == GPIO.LOW
        except Exception:
            return False

    def _loop(self):
        # 부팅 직후에는 핀이 안정되지 않았을 수 있다.
        if self._stop_event.wait(MODE_SWITCH_STARTUP_GRACE_SEC):
            return

        candidate_level = None
        candidate_since = None
        applied_level = None

        while not self._stop_event.is_set():
            try:
                level = GPIO.input(self._pin)
            except Exception as error:
                print(f"[MODE] GPIO{self._pin} 읽기 실패: {error}")
                return

            now = time.monotonic()

            if level != candidate_level:
                candidate_level = level
                candidate_since = now
            elif (
                candidate_since is not None
                and now - candidate_since >= MODE_SWITCH_DEBOUNCE_SEC
                and level != applied_level
            ):
                applied_level = level
                paused = level == GPIO.LOW
                self._on_change(paused)

                if paused:
                    print("[MODE] GPIO23 → 일시정지. "
                          "자세 추론과 진동이 멈춥니다. OS/VNC는 유지됩니다.")
                else:
                    print("[MODE] GPIO23 → 활성. "
                          "모든 일시정지가 해제되고 자세 감지가 재개됩니다.")

            self._stop_event.wait(MODE_SWITCH_POLL_SEC)
