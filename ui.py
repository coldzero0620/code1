#!/usr/bin/env python3
"""
app/ui.py - 화면 표시

MonitorUI가 창 하나를 관리한다. 헤드리스에서는 모든 호출이 조용히 무시되므로
메인 루프가 `if ui_visible:` 조건을 여기저기 두지 않아도 된다.

중요한 성질: UI가 꺼져도 자세 감지·LED·진동은 계속 돈다.
창을 닫는 것은 화면만 끄는 것이지 프로그램을 멈추는 것이 아니다.
"""

import os
import sys

import cv2
import numpy as np

from ..hardware.led import LED_BGR

__all__ = ["MonitorUI", "STATUS_COLORS", "is_headless_env"]

ESC_KEY = 27
SPACE_KEY = ord(" ")

STATUS_COLORS = {
    "NO_POSE": (255, 255, 255),
    "NORMAL": (0, 255, 0),
    "WARNING": (255, 0, 0),
    "BAD": (0, 0, 255),
    "PAUSED": (0, 215, 255),
}


def is_headless_env(argv=None):
    """--headless가 있거나 DISPLAY가 없으면 헤드리스."""
    argv = sys.argv if argv is None else argv
    return "--headless" in argv or not os.environ.get("DISPLAY")


class MonitorUI:
    """
    창 하나와 키 입력을 담당한다.

    show()는 마지막으로 눌린 키를 돌려준다. 창이 닫히면 자동으로
    visible을 False로 내리고, 이후 호출은 전부 무시된다.
    """

    def __init__(self, window_name="Posture Monitor", visible=True):
        self.window_name = window_name
        self.visible = bool(visible)
        self.mode = "LIVE"          # LIVE 또는 SUMMARY
        self._just_closed = False

    # ── 창 ────────────────────────────────────────────────
    def _was_closed(self):
        try:
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def close(self):
        if not self.visible:
            return
        self.visible = False
        self._just_closed = True
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass

    def take_closed_event(self):
        """창이 방금 닫혔는지 한 번만 알려준다."""
        closed, self._just_closed = self._just_closed, False
        return closed

    def show(self, frame):
        """프레임을 표시하고 눌린 키를 돌려준다. 헤드리스면 None."""
        if not self.visible:
            return None

        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if self._was_closed():
            self.close()
            print("[UI] 창이 닫혔습니다. 자세 감지는 백그라운드에서 계속됩니다.")
            return None

        return key

    def destroy_all(self):
        if self.visible:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    # ── 그리기 ────────────────────────────────────────────
    def draw_overlay(self, frame, snapshot, led_color="OFF"):
        status = snapshot["status"]
        status_color = STATUS_COLORS.get(status, (255, 255, 255))

        score_text = (
            f"{snapshot['score']} / 100" if snapshot["score"] is not None else "--"
        )
        ble_text = "CONNECTED" if snapshot["ble_connected"] else "SEARCHING"

        if (
            snapshot["battery_percent"] is not None
            and snapshot["battery_voltage"] is not None
        ):
            battery_text = (
                f"BATTERY: {snapshot['battery_percent']}% "
                f"({snapshot['battery_voltage']:.2f}V)"
            )
        else:
            battery_text = "BATTERY: --"

        charging = snapshot["battery_charging"]
        charging_text = (
            "CHARGING: YES" if charging is True
            else "CHARGING: NO" if charging is False
            else "CHARGING: --"
        )

        rows = (
            (f"STATUS : {status}", status_color, 0.72, 2),
            (f"SCORE  : {score_text}", (255, 215, 0), 0.62, 2),
            (f"HELD   : {snapshot['warning_held']:.1f}s", (255, 255, 255), 0.57, 1),
            (f"TOTAL BAD : {snapshot['total_bad_time']:.1f}s", (180, 180, 255), 0.57, 1),
            (f"GOOD RATIO: {snapshot['good_ratio']:.1f}%", (180, 255, 180), 0.57, 2),
            (f"BLE: {ble_text}", (220, 220, 220), 0.52, 1),
            (battery_text, (220, 220, 220), 0.52, 1),
            (charging_text, (220, 220, 220), 0.52, 1),
        )

        row_gap = 30
        bottom_y = frame.shape[0] - 18
        first_y = bottom_y - row_gap * (len(rows) - 1)
        for index, (text, color, scale, thickness) in enumerate(rows):
            cv2.putText(
                frame, text, (25, first_y + index * row_gap),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
            )

        cv2.putText(
            frame, "SPACE: Pause/Resume    ESC: Summary", (25, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1,
        )
        cv2.circle(
            frame, (frame.shape[1] - 35, 35), 15,
            LED_BGR.get(led_color, (0, 0, 0)), -1,
        )

    def build_summary(self, stats, paused):
        summary = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.rectangle(summary, (80, 55), (720, 545), (35, 35, 35), -1)
        cv2.rectangle(summary, (80, 55), (720, 545), (0, 215, 255), 2)

        monitor_text = "PAUSED" if paused else "ACTIVE"
        monitor_color = (0, 215, 255) if paused else (100, 255, 100)

        lines = [
            ("POSTURE ANALYSIS REPORT", (165, 110), (0, 215, 255), 0.68, 2),
            (f"Total tracked time : {stats.monitored:.1f} sec",
             (135, 175), (255, 255, 255), 0.68, 2),
            (f"Total warning time : {stats.warning:.1f} sec",
             (135, 225), (255, 190, 120), 0.68, 2),
            (f"Total bad time     : {stats.bad:.1f} sec",
             (135, 275), (150, 150, 255), 0.68, 2),
            (f"Average score      : {stats.average_score:.1f} / 100",
             (135, 325), (255, 215, 0), 0.68, 2),
            (f"Good posture rate  : {stats.good_ratio:.1f} %",
             (135, 375), (100, 255, 100), 0.68, 2),
            (f"POSTURE MONITORING: {monitor_text}",
             (135, 435), monitor_color, 0.58, 1),
        ]

        for text, position, color, scale, thickness in lines:
            cv2.putText(
                summary, text, position,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
            )

        cv2.putText(
            summary, "ESC: Return to live camera", (25, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1,
        )
        return summary
