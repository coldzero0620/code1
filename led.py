#!/usr/bin/env python3
"""
hardware/led.py - RGB LED 상태 표시

두 부분이다.

    LedController   색 요청을 받아 GPIO를 쓴다. 색이 바뀔 때만 쓴다.
    LedStatusWorker 별도 스레드. BLE가 끊겼으면 흰색 점멸로 덮어쓴다.

색이 의미하는 것
    GREEN   NORMAL
    BLUE    WARNING, 또는 캘리브레이션 중
    RED     BAD
    WHITE   NO_POSE
    OFF     일시정지
    WHITE 점멸(0.5초)  밴드 연결 끊김. 자세 색보다 우선한다.

밴드 연결이 끊기면 진동이 안 오므로, 사용자가 그 사실을 알아야 한다.
그래서 자세 색을 덮어쓴다.
"""

import threading
import time

from ..contract import LED_PIN_BLUE, LED_PIN_GREEN, LED_PIN_RED
from .gpio import GPIO, setup_output_pins

__all__ = ["LedController", "LedStatusWorker", "LED_COLORS"]

LED_COLORS = frozenset({"OFF", "RED", "GREEN", "BLUE", "WHITE"})

BLE_LOST_BLINK_SEC = 0.5
WORKER_POLL_SEC = 0.05

# 오버레이에서 LED 상태를 그릴 때 쓰는 BGR 값
LED_BGR = {
    "RED": (0, 0, 255),
    "GREEN": (0, 255, 0),
    "BLUE": (255, 0, 0),
    "WHITE": (255, 255, 255),
    "OFF": (0, 0, 0),
}


class LedController:
    """요청 색과 실제 색을 분리해 관리한다."""

    def __init__(self):
        setup_output_pins(LED_PIN_RED, LED_PIN_GREEN, LED_PIN_BLUE)
        self._lock = threading.Lock()
        self._requested = "OFF"
        self._current = "OFF"

    def request(self, color):
        """자세 판정 루프가 원하는 색. 실제 출력은 워커가 정한다."""
        color = str(color).strip().upper()
        if color not in LED_COLORS:
            raise ValueError(f"지원하지 않는 LED 색: {color}")
        with self._lock:
            self._requested = color

    @property
    def requested(self):
        with self._lock:
            return self._requested

    @property
    def current(self):
        with self._lock:
            return self._current

    def current_bgr(self):
        return LED_BGR.get(self.current, (0, 0, 0))

    def drive(self, color):
        """실제 GPIO 출력. 색이 바뀔 때만 쓴다."""
        with self._lock:
            if self._current == color:
                return
            self._current = color

            GPIO.output(LED_PIN_RED, GPIO.LOW)
            GPIO.output(LED_PIN_GREEN, GPIO.LOW)
            GPIO.output(LED_PIN_BLUE, GPIO.LOW)

            if color == "GREEN":
                GPIO.output(LED_PIN_GREEN, GPIO.HIGH)
            elif color == "BLUE":
                GPIO.output(LED_PIN_BLUE, GPIO.HIGH)
            elif color == "RED":
                GPIO.output(LED_PIN_RED, GPIO.HIGH)
            elif color == "WHITE":
                GPIO.output(LED_PIN_RED, GPIO.HIGH)
                GPIO.output(LED_PIN_GREEN, GPIO.HIGH)
                GPIO.output(LED_PIN_BLUE, GPIO.HIGH)


class LedStatusWorker:
    """
    자세 색을 표시하되, 밴드가 끊겼으면 흰색 점멸로 덮어쓴다.

    is_connected는 콜백이다. 하드웨어 계층이 앱 상태 객체를 직접
    참조하지 않도록 하기 위한 것이다.
    """

    def __init__(self, controller, is_connected):
        self._controller = controller
        self._is_connected = is_connected
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="rgb-led-status", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._controller.drive("OFF")

    def _loop(self):
        disconnected_since = None

        while not self._stop_event.is_set():
            now = time.monotonic()
            if self._is_connected():
                disconnected_since = None
                target = self._controller.requested
            else:
                if disconnected_since is None:
                    disconnected_since = now
                blink_step = int((now - disconnected_since) / BLE_LOST_BLINK_SEC)
                target = "WHITE" if blink_step % 2 == 0 else "OFF"

            self._controller.drive(target)
            self._stop_event.wait(WORKER_POLL_SEC)
