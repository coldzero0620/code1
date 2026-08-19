#!/usr/bin/env python3
"""
judge/band.py - 판정 상태 → 진동 밴드 명령

BLE 전송 자체는 여기 없다. hardware/band.py가 담당한다.
이 파일은 "어떤 문자를 언제 보낼 것인가"라는 규약만 다루므로
하드웨어 없이 시험할 수 있다.
"""

from typing import Callable, Optional

from ..contract import (
    BAND_COMMANDS,
    BAND_HEARTBEAT_CHAR,
    BAND_KEEPALIVE_SEC,
    BAND_NON_IDEMPOTENT,
    BAND_WARNING_REPEAT_SEC,
)


class BandLink:
    """
    판정 상태를 밴드 명령으로 옮긴다. BLE 전송 자체는 관여하지 않는다.

        link = BandLink(send_fn=lambda c: client.write_gatt_char(UUID, c.encode()))
        link.on_connected()          # 연결될 때마다
        link.update(status)          # 판정 루프 안에서 매 프레임

    펌웨어 규약 3가지를 지킨다.

    1. contract.BAND_COMMANDS만 사용한다.
       펌웨어는 첫 글자만 읽으므로 상태 문자열을 그대로 보내면
       NO_POSE가 NORMAL로 오해석된다.

    2. 유지 중에는 현재 명령을 재전송한다.
       N/B/P/S는 펌웨어에서 멱등이다. 이렇게 하면 통신 이상이나
       재연결로 밴드가 초기화돼도 다음 주기(2초)에 자동 복구된다.
       펌웨어를 고치지 않고 BAD 경고 소실을 막는 방법이다.

    3. WARNING만 예외로 heartbeat를 보낸다.
       'W'는 경고 패턴이 끝난 뒤 다시 받으면 재발동하므로,
       매번 보내면 끊김 없는 진동이 된다.
    """

    def __init__(
        self,
        send_fn: Callable[[str], None],
        keepalive_sec: float = BAND_KEEPALIVE_SEC,
        warning_repeat_sec: Optional[float] = BAND_WARNING_REPEAT_SEC,
    ):
        self.send_fn = send_fn
        self.keepalive_sec = keepalive_sec
        self.warning_repeat_sec = warning_repeat_sec

        self._sent_status: Optional[str] = None
        self._last_send = 0.0
        self._last_warning_alert = 0.0
        self._paused = False
        self.send_failures = 0

    # ── 연결 관리 ──────────────────────────────────────
    def on_connected(self, now: Optional[float] = None) -> None:
        """
        재연결 직후 호출한다.

        펌웨어는 onConnect에서 stopVibration()을 불러 상태를 OFF로 되돌린다.
        _sent_status를 비워 다음 update()에서 현재 상태를 강제 재전송한다.
        """
        self._sent_status = None
        self._last_send = 0.0

    def on_disconnected(self) -> None:
        self._sent_status = None

    # ── 일시정지 ───────────────────────────────────────
    def pause(self) -> None:
        """캘리브레이션 중이거나 사용자가 멈췄을 때. 진동을 끈다."""
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    @property
    def paused(self) -> bool:
        return self._paused

    # ── 매 프레임 호출 ─────────────────────────────────
    def update(self, status: str, now: Optional[float] = None) -> Optional[str]:
        """보낸 문자를 반환한다. 보낼 것이 없으면 None."""
        if now is None:
            import time

            now = time.monotonic()

        # 시계가 뒤로 갔다면(재부팅·수동 조정) 기준을 당겨 교착을 막는다.
        if now < self._last_send:
            self._last_send = now
            self._last_warning_alert = now

        effective = "PAUSED" if self._paused else status

        command = BAND_COMMANDS.get(effective)
        if command is None:
            # 알 수 없는 상태를 밴드에 넘기지 않는다.
            print(f"[BAND] 알 수 없는 상태 '{status}' → 무시하고 연결만 유지")
            return self._keepalive(BAND_HEARTBEAT_CHAR, now)

        # 1) 상태 전환 → 명령 전송
        if effective != self._sent_status:
            if self._send(command, now):
                self._sent_status = effective
                if effective == "WARNING":
                    self._last_warning_alert = now
                return command
            return None

        # 2) WARNING 재알림
        if (
            effective == "WARNING"
            and self.warning_repeat_sec is not None
            and now - self._last_warning_alert >= self.warning_repeat_sec
        ):
            if self._send(command, now):
                self._last_warning_alert = now
                return command
            return None

        # 3) 유지 중 → 멱등 명령은 재전송, 그 외는 heartbeat
        keep = (
            BAND_HEARTBEAT_CHAR if effective in BAND_NON_IDEMPOTENT else command
        )
        return self._keepalive(keep, now)

    # ── 내부 ───────────────────────────────────────────
    def _keepalive(self, char: str, now: float) -> Optional[str]:
        if now - self._last_send < self.keepalive_sec:
            return None
        return char if self._send(char, now) else None

    def _send(self, command: str, now: float) -> bool:
        try:
            self.send_fn(command)
        except Exception as error:
            self.send_failures += 1
            # 여기서 죽으면 안 된다. BLE 재연결은 상위 계층의 몫이다.
            if self.send_failures in (1, 10, 100):
                print(f"[BAND] 전송 실패 {self.send_failures}회: {error}")
            return False

        self._last_send = now
        return True
