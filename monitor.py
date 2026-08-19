#!/usr/bin/env python3
"""
app/monitor.py - 메인 감시 루프

계층을 조립하고 프레임마다 다음을 반복한다.

    카메라 → 특징 추출 → 판정 → LED / 밴드 / 화면

V12.2에서는 이 루프 안에 특징 계산, 분류, 히스테리시스, 타이머가
전부 들어 있어 700줄이었다. 그 부분은 전부 judge 패키지로 옮겼다.
여기 남은 것은 "언제 무엇을 부를 것인가"뿐이다.

────────────────────────────────────────────────────────────
상태 → 출력 대응

    NORMAL   초록 LED,  밴드 N
    WARNING  파랑 LED,  밴드 W
    BAD      빨강 LED,  밴드 B
    NO_POSE  흰색 LED,  밴드 P
    PAUSED   LED 끔,    밴드 S

밴드 연결이 끊기면 LED 워커가 흰색 점멸로 덮어쓴다.
────────────────────────────────────────────────────────────
"""

import signal
import threading
import time

from ..contract import CALIBRATION_RETRY_SEC, MODEL_COMPLEXITY
from ..features import create_pose, draw_feature, extract_feature, process_pose
from ..hardware import (
    LatestFrameCamera,
    LedController,
    LedStatusWorker,
    ModeSwitchMonitor,
    PostureBand,
    cleanup as gpio_cleanup,
)
from ..judge import build_judge, load_manifest
from .calibration import CalibrationError, calibrate
from .stats import PostureScorer, SessionStats
from .state import RuntimeState
from .ui import ESC_KEY, SPACE_KEY, MonitorUI, is_headless_env

__all__ = ["PostureMonitor", "run"]

# 상태별 LED 색
LED_FOR_STATUS = {
    "NORMAL": "GREEN",
    "WARNING": "BLUE",
    "BAD": "RED",
    "NO_POSE": "WHITE",
    "PAUSED": "OFF",
}

# 프레임 간격 상한. 정지 후 재개처럼 큰 공백이 통계에 통째로 들어가는 것을 막는다.
MAX_FRAME_GAP_SEC = 0.5


class PostureMonitor:
    def __init__(self, model="rf", headless=None, window_name="Posture Monitor"):
        self.stop_event = threading.Event()
        self.state = RuntimeState()
        self.stats = SessionStats()

        headless = is_headless_env() if headless is None else headless
        self.ui = MonitorUI(window_name=window_name, visible=not headless)

        self.led = LedController()
        self.led_worker = LedStatusWorker(self.led, self.state.is_ble_connected)
        self.band = PostureBand(
            on_connection=lambda ok: self.state.update(ble_connected=ok),
            on_battery=self._on_battery,
        )
        self.switch = ModeSwitchMonitor(self.state.set_switch_paused)
        self.camera = LatestFrameCamera(
            on_state=lambda ok: self.state.update(camera_ok=ok)
        )

        self.model = model
        self.judge = None
        self.scorer = None
        self.pose = None

    def _on_battery(self, percent, voltage, charging):
        self.state.update(
            battery_percent=percent,
            battery_voltage=voltage,
            battery_charging=charging,
        )

    # ── 출력 ──────────────────────────────────────────────
    def _apply_status(self, status):
        """판정 상태를 LED와 밴드에 반영한다."""
        self.led.request(LED_FOR_STATUS.get(status, "WHITE"))
        self.band.set_status(status)

    # ── 생명주기 ──────────────────────────────────────────
    def _handle_signal(self, signum, frame):
        # systemd stop이나 터미널 종료. GPIO23은 여기 관여하지 않는다.
        # 스위치가 프로그램을 종료시키면 Pi 전원을 뽑아야 재시작할 수 있다.
        self.stop_event.set()

    def start_workers(self):
        signal.signal(signal.SIGTERM, self._handle_signal)

        self.switch.start()
        self.led_worker.start()
        self.band.start()
        self.camera.start()

        # 스위치 초기 위치를 즉시 반영한다.
        # 그러지 않으면 껐다 켤 때까지 정지 상태가 반영되지 않는다.
        self.state.set_switch_paused(self.switch.read_now())

        manifest = load_manifest()
        self.judge = build_judge(self.model)
        self.scorer = PostureScorer(manifest)
        print(f"[JUDGE] {self.scorer.describe()}")

        self.pose = create_pose()
        print(f"[POSE] MediaPipe model_complexity={MODEL_COMPLEXITY}")

        self.led.request("BLUE")
        self._apply_status("PAUSED")

    def shutdown(self):
        self.switch.stop()
        self.led_worker.stop()
        self.band.stop()
        self.camera.stop()

        if self.pose is not None:
            self.pose.close()

        gpio_cleanup()
        self.ui.destroy_all()

    # ── 캘리브레이션 ──────────────────────────────────────
    def run_calibration(self):
        """성공할 때까지 재시도한다. 정지 요청이 오면 중단한다."""
        while not self.stop_event.is_set():
            try:
                baseline = calibrate(
                    self.camera, self.pose, self.state, self.ui, self.stop_event
                )
                self.led.request("GREEN")
                self._apply_status("NORMAL")
                return baseline
            except CalibrationError as error:
                print(f"[CALIBRATION] {error} "
                      f"{CALIBRATION_RETRY_SEC:.0f}초 후 다시 시도합니다.")
                self.state.update(status="NO_POSE", score=None, side="-")
                self._apply_status("NO_POSE")
                self.led.request("BLUE")
                if self.stop_event.wait(CALIBRATION_RETRY_SEC):
                    break
        raise KeyboardInterrupt

    # ── 메인 루프 ─────────────────────────────────────────
    def run_session(self, baseline):
        previous_frame_id = -1
        previous_time = time.monotonic()
        previous_switch_paused = self.state.is_switch_paused()

        def resume(force=False):
            nonlocal previous_time
            changed = self.state.set_paused(False)
            if not changed and not force:
                return
            self.judge.reset(time.monotonic())
            previous_time = time.monotonic()
            self._apply_status("NO_POSE")
            self.state.update(status="NO_POSE", score=None, warning_held=0.0, side="-")
            print("[PROGRAM] 감시를 재개합니다.")

        while not self.stop_event.is_set():
            frame_id, frame, _ = self.camera.get_latest(previous_frame_id)

            # ── 프레임이 없을 때 ──
            if frame is None:
                # 카메라가 끊긴 채로 WARNING/BAD 진동을 붙들고 있으면 안 된다.
                self.judge.reset(time.monotonic())
                status = "PAUSED" if self.state.is_paused() else "NO_POSE"
                self._apply_status(status)
                self.state.update(
                    status=status, score=None, warning_held=0.0,
                    side="-", camera_ok=False,
                )
                if self.camera.is_fatally_stalled():
                    raise RuntimeError("카메라가 정해진 시간 안에 복구되지 않았습니다.")
                continue

            previous_frame_id = frame_id
            now = time.monotonic()
            elapsed = min(max(now - previous_time, 0.0), MAX_FRAME_GAP_SEC)
            previous_time = now

            # ── 스위치 마스터 재개 ──
            switch_paused_now = self.state.is_switch_paused()
            if previous_switch_paused and not switch_paused_now:
                resume(force=True)
                self.ui.mode = "LIVE"
                print("[MODE] 마스터 재개: UI 정지와 요약 화면을 함께 해제했습니다.")
            previous_switch_paused = switch_paused_now

            # ── 판정 ──
            if self.state.is_paused():
                self.judge.reset(now)
                self._apply_status("PAUSED")
                self.state.update(
                    status="PAUSED", score=None, warning_held=0.0, side="-"
                )
            else:
                feature = extract_feature(process_pose(self.pose, frame))
                status, info = self.judge.decide(feature, baseline, now)

                score = self.scorer.score(info) if feature is not None else None
                self.stats.add(status, elapsed, score)

                draw_feature(frame, feature)
                self._apply_status(status)
                self.state.update(
                    status=status,
                    score=score,
                    warning_held=info.get("warning_held_sec", 0.0),
                    side=info.get("side") or "-",
                    model_source=info.get("reason", "-"),
                    confidence=info.get("confidence"),
                    total_monitored_time=self.stats.monitored,
                    total_warning_time=self.stats.warning,
                    total_bad_time=self.stats.bad,
                    good_ratio=self.stats.good_ratio,
                )

            # ── 화면 ──
            self._render(frame, resume)

    def _render(self, frame, resume):
        if not self.ui.visible:
            return

        snapshot = self.state.snapshot()
        if self.ui.mode == "LIVE":
            self.ui.draw_overlay(frame, snapshot, self.led.current)
            key = self.ui.show(frame)
        else:
            key = self.ui.show(
                self.ui.build_summary(self.stats, snapshot["paused"])
            )

        if self.ui.take_closed_event():
            # 창을 닫는 것은 화면만 끄는 것이다. 감시는 계속된다.
            resume()
            return

        if key is None:
            return

        if self.ui.mode == "LIVE" and key == SPACE_KEY:
            paused = self.state.toggle_paused()
            self.judge.reset(time.monotonic())
            if paused:
                self._apply_status("PAUSED")
                self.state.update(status="PAUSED", score=None, warning_held=0.0)
                print("[PROGRAM] 감시를 일시정지했습니다.")
            else:
                resume(force=True)

        elif self.ui.mode == "LIVE" and key == ESC_KEY:
            # 요약을 보는 동안 통계가 계속 쌓이면 숫자가 흔들린다. 함께 멈춘다.
            self.state.set_paused(True)
            self.judge.reset(time.monotonic())
            self._apply_status("PAUSED")
            self.state.update(status="PAUSED", score=None, warning_held=0.0)
            self.ui.mode = "SUMMARY"
            print("[UI] 요약 화면. 감시를 일시정지했습니다.")

        elif self.ui.mode == "SUMMARY" and key == ESC_KEY:
            resume(force=True)
            self.ui.mode = "LIVE"
            print("[UI] 라이브 화면으로 돌아갑니다.")


def run(model="rf", headless=None):
    monitor = PostureMonitor(model=model, headless=headless)
    exit_code = 0
    try:
        monitor.start_workers()
        while not monitor.stop_event.is_set():
            baseline = monitor.run_calibration()
            monitor.run_session(baseline)
    except KeyboardInterrupt:
        print("[PROGRAM] 중단 요청을 받았습니다.")
    except RuntimeError as error:
        exit_code = 1
        print(f"[ERROR] {error}")
    except Exception as error:
        exit_code = 1
        print(f"[ERROR] {type(error).__name__}: {error}")
    finally:
        monitor.shutdown()
        print(f"[PROGRAM] 안전하게 종료했습니다. exit_code={exit_code}")
    return exit_code
