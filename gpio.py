#!/usr/bin/env python3
"""
hardware/gpio.py - GPIO 접근을 한 곳으로 모은다

라즈베리파이가 아닌 곳에서도 import가 성공해야 한다.
그래야 노트북에서 앱 전체를 켜보고 로직을 확인할 수 있다.

RPi.GPIO가 없으면 아무것도 하지 않는 더미로 대체하고,
GPIO_AVAILABLE을 False로 둔다. 호출부는 이 플래그만 보면 된다.
"""

import threading

__all__ = ["GPIO", "GPIO_AVAILABLE", "setup_output_pins", "cleanup"]


class _DummyGPIO:
    """RPi.GPIO와 같은 이름을 가진 무동작 대체물."""

    BCM = "BCM"
    IN = "IN"
    OUT = "OUT"
    PUD_UP = "PUD_UP"
    LOW = 0
    HIGH = 1

    def setmode(self, mode):
        pass

    def setwarnings(self, enabled):
        pass

    def setup(self, pin, mode, **kwargs):
        pass

    def output(self, pin, state):
        pass

    def input(self, pin):
        # 스위치는 active-LOW다. HIGH를 돌려주면 "정지 아님"이 되어
        # 시뮬레이션에서 자세 감지가 정상 동작한다.
        return self.HIGH

    def cleanup(self):
        pass


try:
    import RPi.GPIO as _GPIO

    GPIO = _GPIO
    GPIO_AVAILABLE = True
except Exception as _error:  # ImportError 외에 권한 오류도 잡는다
    GPIO = _DummyGPIO()
    GPIO_AVAILABLE = False
    _IMPORT_ERROR = _error


_setup_lock = threading.Lock()
_initialised = False


def _ensure_mode():
    global _initialised
    with _setup_lock:
        if _initialised:
            return
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        _initialised = True
        if GPIO_AVAILABLE:
            print("[GPIO] 하드웨어 모드")
        else:
            print(f"[GPIO] 시뮬레이션 모드: {_IMPORT_ERROR}")


def setup_output_pins(*pins):
    """출력 핀을 LOW로 초기화한다. 여러 번 불러도 안전하다."""
    _ensure_mode()
    for pin in pins:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)


def setup_input_pullup(pin):
    """내부 풀업을 켠 입력 핀."""
    _ensure_mode()
    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)


def cleanup():
    if GPIO_AVAILABLE:
        GPIO.cleanup()
