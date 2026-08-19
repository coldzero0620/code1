"""
posture.hardware - 라즈베리파이 주변장치

카메라, RGB LED, BLE 진동 밴드, 슬라이드 스위치.

이 계층은 판정 로직을 모른다. judge 패키지에서 가져오는 것은
밴드 명령 규약(BandLink) 하나뿐이며, 그것도 "무엇을 보낼지"를
직접 정하지 않기 위해서다.

상태 변화는 전부 콜백으로 위쪽에 알린다. RuntimeState를 직접
참조하지 않으므로, 하드웨어만 따로 시험할 수 있다.

라즈베리파이가 아닌 곳에서도 import는 성공한다.
RPi.GPIO가 없으면 더미로 대체되고, bleak이 없으면 밴드 기능만 꺼진다.
카메라는 rpicam-vid가 없으면 CameraUnavailable을 올린다.
"""

from .band import BLE_AVAILABLE, PostureBand, parse_battery_payload
from .camera import CameraUnavailable, LatestFrameCamera, RpicamCapture
from .gpio import GPIO, GPIO_AVAILABLE, cleanup
from .led import LED_BGR, LedController, LedStatusWorker
from .switch import ModeSwitchMonitor

__all__ = [
    "BLE_AVAILABLE",
    "CameraUnavailable",
    "GPIO",
    "GPIO_AVAILABLE",
    "LED_BGR",
    "LatestFrameCamera",
    "LedController",
    "LedStatusWorker",
    "ModeSwitchMonitor",
    "PostureBand",
    "RpicamCapture",
    "cleanup",
    "parse_battery_payload",
]
