#!/usr/bin/env python3
"""
hardware/band.py - XIAO ESP32-C3 진동 밴드 BLE 연결

여기는 "어떻게 보내는가"만 담당한다.
"무엇을 언제 보내는가"는 judge/band.py의 BandLink가 정한다.

모듈화 전에는 런타임이 N/W/B/P/S 문자를 코드 안에 직접 박아 썼고,
파이프라인은 contract.BAND_COMMANDS를 썼다. 두 벌이 따로 놀았다.
이제 문자 결정은 BandLink 한 곳에서만 이뤄진다.

────────────────────────────────────────────────────────────
펌웨어 규약

  펌웨어는 수신 문자열의 첫 글자만 읽는다.
  따라서 상태 문자열("NO_POSE")을 그대로 보내면 'N'(=NORMAL)로 오해석된다.
  반드시 BAND_COMMANDS 표를 거쳐야 한다.

  P와 S는 앞에 N을 먼저 보낸다. N/W/B만 아는 구버전 펌웨어에서도
  일시정지·종료 때 모터가 확실히 꺼지도록 하기 위함이다.

  Pi가 갑자기 꺼졌을 때의 모터 정지는 펌웨어 쪽 H 타임아웃과
  연결 해제 페일세이프가 담당한다. 여기서 보장할 수 없다.
────────────────────────────────────────────────────────────
"""

import asyncio
import threading
import time

from ..contract import (
    BAND_PREFIX_WITH_NORMAL,
    BLE_BATTERY_CHAR_UUID,
    BLE_COMMAND_CHAR_UUID,
    BLE_CONNECT_TIMEOUT_SEC,
    BLE_DEVICE_NAME,
    BLE_POLL_SEC,
    BLE_RETRY_SEC,
    BLE_SCAN_TIMEOUT_SEC,
    BLE_SERVICE_UUID,
)
from ..judge.band import BandLink

__all__ = ["PostureBand", "BLE_AVAILABLE", "parse_battery_payload"]

try:
    from bleak import BleakClient, BleakScanner

    BLE_AVAILABLE = True
except ImportError:
    BleakClient = BleakScanner = None
    BLE_AVAILABLE = False


def parse_battery_payload(payload: str):
    """
    "BAT,87,3.95,1" → (87, 3.95, True)

    충전 필드는 없을 수도 있다. 그 경우 None을 돌려준다.
    형식이 어긋나면 ValueError를 올린다.
    이 함수는 순수 함수라 BLE 없이 시험할 수 있다.
    """
    parts = payload.strip().split(",")
    if len(parts) not in (3, 4) or parts[0].upper() != "BAT":
        raise ValueError(f"알 수 없는 배터리 형식: {payload!r}")

    percent = max(0, min(100, int(parts[1])))
    voltage = float(parts[2])

    charging = None
    if len(parts) == 4:
        token = parts[3].strip().upper()
        if token in {"1", "TRUE", "YES", "ON", "CHARGING"}:
            charging = True
        elif token in {"0", "FALSE", "NO", "OFF", "NOT_CHARGING", "IDLE", "FULL"}:
            charging = False
        else:
            raise ValueError(f"알 수 없는 충전 상태: {parts[3]!r}")

    return percent, voltage, charging


class PostureBand:
    """
    BLE 연결을 유지하며 BandLink가 정한 명령을 전송한다.

    on_connection(bool)     연결 상태가 바뀔 때
    on_battery(percent, voltage, charging)  배터리 알림 수신 시

    두 콜백 모두 앱 계층이 넘긴다. 이 파일은 RuntimeState를 모른다.
    """

    def __init__(self, on_connection=None, on_battery=None):
        self._on_connection = on_connection
        self._on_battery = on_battery

        self._status = "PAUSED"
        self._status_lock = threading.Lock()
        self._sent_condition = threading.Condition(self._status_lock)
        self._last_confirmed = None

        self._known_address = None
        self._stop_event = threading.Event()
        self._thread = None

    # ── 앱이 부르는 부분 ────────────────────────────────────
    def start(self):
        if not BLE_AVAILABLE:
            print("[BLE] bleak이 설치돼 있지 않습니다. 밴드 기능이 꺼집니다.")
            self._report_disconnected()
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._thread_main, name="posture-band-ble", daemon=True
        )
        self._thread.start()

    def set_status(self, status):
        """판정 상태를 그대로 넘긴다. 명령 문자 변환은 BandLink가 한다."""
        status = str(status).strip().upper()
        with self._sent_condition:
            self._status = status
            self._sent_condition.notify_all()

    def stop(self):
        """
        종료 시 모터를 확실히 끈다.
        PAUSED 명령이 실제로 전송된 것을 짧게 기다린 뒤 스레드를 내린다.
        """
        self.set_status("PAUSED")

        if self._thread is None:
            self._report_disconnected()
            return

        deadline = time.monotonic() + 1.0
        with self._sent_condition:
            while self._last_confirmed != "PAUSED" and time.monotonic() < deadline:
                self._sent_condition.wait(timeout=0.1)

        self._stop_event.set()
        self._thread.join(timeout=6.0)
        self._report_disconnected()

    # ── 내부 ───────────────────────────────────────────────
    def _report_disconnected(self):
        if self._on_connection is not None:
            self._on_connection(False)
        if self._on_battery is not None:
            self._on_battery(None, None, None)

    def _handle_battery(self, sender, data):
        try:
            percent, voltage, charging = parse_battery_payload(
                bytes(data).decode("utf-8")
            )
        except (UnicodeDecodeError, ValueError) as error:
            print(f"[BLE] 배터리 알림 해석 실패: {error}")
            return

        if self._on_battery is not None:
            self._on_battery(percent, voltage, charging)

        charging_text = "YES" if charging else "NO" if charging is False else "--"
        print(f"[BLE] 배터리 {percent}% ({voltage:.2f}V), 충전={charging_text}")

    def _thread_main(self):
        while not self._stop_event.is_set():
            try:
                asyncio.run(self._ble_loop())
            except Exception as error:
                print(f"[BLE] 워커 오류: {error}")
                self._report_disconnected()

            if self._stop_event.wait(BLE_RETRY_SEC):
                break

    def _matches_target(self, device, advertisement_data):
        # 한 번 붙은 뒤에는 주소로 즉시 판별한다.
        # 이름 매칭은 scan response를 기다려야 해서 재연결이 느리다.
        if (
            self._known_address is not None
            and device.address.casefold() == self._known_address.casefold()
        ):
            return True

        advertised = {
            uuid.casefold() for uuid in (advertisement_data.service_uuids or [])
        }
        if BLE_SERVICE_UUID.casefold() in advertised:
            return True

        local_name = advertisement_data.local_name or device.name
        return local_name == BLE_DEVICE_NAME

    async def _ble_loop(self):
        while not self._stop_event.is_set():
            try:
                print(f"[BLE] {BLE_DEVICE_NAME} 검색 중...")
                device = await BleakScanner.find_device_by_filter(
                    self._matches_target, timeout=BLE_SCAN_TIMEOUT_SEC
                )
                if device is None:
                    print(f"[BLE] {BLE_DEVICE_NAME}를 못 찾았습니다. "
                          "밴드를 라즈베리파이 가까이 두세요.")
                    await asyncio.sleep(BLE_RETRY_SEC)
                    continue

                self._known_address = device.address
                await self._serve_connection(device)

            except Exception as error:
                self._report_disconnected()
                if not self._stop_event.is_set():
                    print(f"[BLE] 연결이 끊겼습니다: {error}")
                    await asyncio.sleep(BLE_RETRY_SEC)

        self._report_disconnected()

    async def _serve_connection(self, device):
        async with BleakClient(device, timeout=BLE_CONNECT_TIMEOUT_SEC) as client:
            print(f"[BLE] 연결됨: {BLE_DEVICE_NAME}")
            if self._on_connection is not None:
                self._on_connection(True)

            await self._subscribe_battery(client)

            # 명령 결정은 BandLink에 맡긴다. 여기는 전송만 한다.
            pending = []
            link = BandLink(send_fn=pending.append)
            link.on_connected(time.monotonic())

            while client.is_connected and not self._stop_event.is_set():
                with self._status_lock:
                    status = self._status

                now = time.monotonic()
                pending.clear()
                link.update(status, now)

                for command in pending:
                    await self._write(client, command)

                if pending:
                    with self._sent_condition:
                        self._last_confirmed = status
                        self._sent_condition.notify_all()

                await asyncio.sleep(BLE_POLL_SEC)

        self._report_disconnected()

    async def _subscribe_battery(self, client):
        try:
            await client.start_notify(BLE_BATTERY_CHAR_UUID, self._handle_battery)
        except Exception as error:
            print(f"[BLE] 배터리 알림 구독 실패: {error}")
            return
        try:
            initial = await client.read_gatt_char(BLE_BATTERY_CHAR_UUID)
            self._handle_battery(BLE_BATTERY_CHAR_UUID, initial)
        except Exception as error:
            print(f"[BLE] 배터리 초기값 읽기 실패: {error}")

    async def _write(self, client, command):
        # P/S 앞에 N을 먼저 보낸다. 구버전 펌웨어 대비.
        if command in BAND_PREFIX_WITH_NORMAL:
            await client.write_gatt_char(
                BLE_COMMAND_CHAR_UUID, b"N", response=True
            )
        await client.write_gatt_char(
            BLE_COMMAND_CHAR_UUID, command.encode("utf-8"), response=True
        )
