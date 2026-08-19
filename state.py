#!/usr/bin/env python3
"""
app/state.py - 여러 스레드가 함께 보는 상태

캡처 스레드, LED 워커, BLE 워커, 스위치 감시, 메인 루프가 동시에 돌아간다.
이 객체가 그 사이의 유일한 공유 지점이다.

나중에 Flask 상태 API를 붙일 때도 snapshot() 하나만 노출하면 된다.

────────────────────────────────────────────────────────────
일시정지가 두 겹인 이유

  ui_paused      SPACE 키, 요약 화면
  switch_paused  GPIO23 슬라이드 스위치

물리 스위치가 마스터다. HIGH로 올리면 UI 쪽 정지까지 함께 풀린다.
그렇게 하지 않으면 "스위치는 켰는데 왜 안 움직이지" 상황이 생긴다.
실제 정지 여부는 둘의 OR이다.
────────────────────────────────────────────────────────────
"""

import threading
import time

__all__ = ["RuntimeState"]


class RuntimeState:
    def __init__(self):
        self._lock = threading.Lock()

        self.ui_paused = False
        self.switch_paused = False
        self.paused = False

        self.status = "NO_POSE"
        self.score = None
        self.warning_held = 0.0
        self.side = "-"
        self.model_source = "-"
        self.confidence = None

        self.total_monitored_time = 0.0
        self.total_warning_time = 0.0
        self.total_bad_time = 0.0
        self.good_ratio = 100.0

        self.ble_connected = False
        self.battery_percent = None
        self.battery_voltage = None
        self.battery_charging = None
        self.camera_ok = False

        self.updated_at = time.monotonic()

    def update(self, **values):
        with self._lock:
            for name, value in values.items():
                if not hasattr(self, name):
                    raise AttributeError(f"알 수 없는 상태 항목: {name}")
                setattr(self, name, value)
            self.updated_at = time.monotonic()

    # ── 일시정지 ───────────────────────────────────────────
    def _refresh_paused_locked(self):
        effective = self.ui_paused or self.switch_paused
        changed = self.paused != effective
        self.paused = effective
        self.updated_at = time.monotonic()
        return changed

    def toggle_paused(self):
        """SPACE 키. 스위치 정지가 걸려 있으면 그쪽이 우선한다."""
        with self._lock:
            self.ui_paused = not self.ui_paused
            self._refresh_paused_locked()
            return self.paused

    def set_paused(self, paused):
        with self._lock:
            self.ui_paused = bool(paused)
            return self._refresh_paused_locked()

    def set_switch_paused(self, paused):
        """
        물리 스위치. HIGH(=paused False)는 마스터 재개 명령이므로
        UI 쪽 정지까지 함께 푼다.
        """
        with self._lock:
            paused = bool(paused)
            self.switch_paused = paused
            if not paused:
                self.ui_paused = False
            return self._refresh_paused_locked()

    # ── 조회 ───────────────────────────────────────────────
    def is_paused(self):
        with self._lock:
            return self.paused

    def is_switch_paused(self):
        with self._lock:
            return self.switch_paused

    def is_ble_connected(self):
        with self._lock:
            return self.ble_connected

    def snapshot(self):
        with self._lock:
            return {
                "paused": self.paused,
                "ui_paused": self.ui_paused,
                "switch_paused": self.switch_paused,
                "status": self.status,
                "score": self.score,
                "warning_held": self.warning_held,
                "side": self.side,
                "model_source": self.model_source,
                "confidence": self.confidence,
                "total_monitored_time": self.total_monitored_time,
                "total_warning_time": self.total_warning_time,
                "total_bad_time": self.total_bad_time,
                "good_ratio": self.good_ratio,
                "ble_connected": self.ble_connected,
                "battery_percent": self.battery_percent,
                "battery_voltage": self.battery_voltage,
                "battery_charging": self.battery_charging,
                "camera_ok": self.camera_ok,
                "updated_at": self.updated_at,
            }
