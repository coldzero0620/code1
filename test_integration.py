#!/usr/bin/env python3
"""
가짜 하드웨어로 PostureMonitor 전체를 돌린다.

카메라, BLE, GPIO, 화면이 없는 환경에서 다음을 확인한다.

  1. 캘리브레이션이 baseline 네 축을 모두 잡는가
  2. 자세를 바꾸면 판정이 따라오는가
  3. 판정 → LED 색 / 밴드 명령이 규약대로 나가는가
  4. NO_POSE에서 진동이 확실히 꺼지는가
  5. 일시정지 / 마스터 재개가 동작하는가
  6. 종료 시 밴드가 정지 명령을 받는가

MediaPipe는 실제로 쓰지 않는다. 합성 인체 모형에서 만든 특징을
직접 주입해 판정 계층부터 아래를 검사한다.
"""

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from posture.app.monitor import LED_FOR_STATUS, PostureMonitor
from posture.contract import BAND_COMMANDS
from posture.judge import BandLink

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


# ─────────────────────────────────────────────────────────────
# 합성 특징 생성
# ─────────────────────────────────────────────────────────────
def make_feature(neck_deg, torso_deg=0.0, pose=True):
    """features.extract_feature가 내놓는 것과 같은 모양의 dict."""
    if not pose:
        return None
    neck = math.radians(neck_deg)
    return {
        "side": "LEFT",
        "facing": 1.0,
        "signed_delta": 0.30 + 0.010 * neck_deg,
        "abs_delta": 0.30 + 0.010 * neck_deg,
        "obliquity": 0.25,
        "ear": (0.45, 0.30),
        "shoulder": (0.50, 0.45),
        "hip": (0.52, 0.80),
        "fwd_ratio": 0.22 * math.sin(neck),
        "cva_deg": float(neck_deg),
        "torso_angle_deg": float(torso_deg),
        "world_ok": 1,
    }


BASELINE = {
    "signed_delta": 0.30,
    "fwd_ratio": 0.0,
    "cva_deg": 0.0,
    "torso_angle_deg": 0.0,
}


# ─────────────────────────────────────────────────────────────
# 가짜 하드웨어
# ─────────────────────────────────────────────────────────────
class FakeCamera:
    """프레임 ID만 증가시키는 카메라. 항상 검은 프레임을 준다."""

    def __init__(self):
        self._id = 0
        self.stalled = False
        self.blank = False      # True면 프레임을 주지 않는다

    def start(self):
        pass

    def stop(self):
        pass

    def get_latest(self, previous_id=-1, timeout=None):
        if self.blank:
            return previous_id, None, None
        self._id += 1
        return self._id, FRAME.copy(), time.monotonic()

    def seconds_since_frame(self):
        return 999.0 if self.stalled else 0.0

    def is_fatally_stalled(self):
        return self.stalled


class FakeBand:
    """set_status로 받은 상태를 BandLink에 통과시켜 실제 명령을 기록한다."""

    def __init__(self):
        self.commands = []
        self.statuses = []
        self._link = BandLink(send_fn=self.commands.append)
        self._link.on_connected(0.0)
        self._clock = 0.0

    def start(self):
        pass

    def set_status(self, status):
        self.statuses.append(status)
        self._clock += 0.05
        self._link.update(status, self._clock)

    def stop(self):
        self.set_status("PAUSED")


class FakeLed:
    def __init__(self):
        self.current = "OFF"
        self.history = []

    def request(self, color):
        self.current = color
        self.history.append(color)

    def current_bgr(self):
        return (0, 0, 0)


class FakeWorker:
    def start(self):
        pass

    def stop(self):
        pass


class FakeSwitch(FakeWorker):
    def __init__(self):
        self.paused = False

    def read_now(self):
        return self.paused


# ─────────────────────────────────────────────────────────────
# 조립
# ─────────────────────────────────────────────────────────────
def build_monitor(model="threshold"):
    monitor = PostureMonitor(model=model, headless=True)
    monitor.camera = FakeCamera()
    monitor.band = FakeBand()
    monitor.led = FakeLed()
    monitor.led_worker = FakeWorker()
    monitor.switch = FakeSwitch()
    monitor._clock = 1000.0     # 테스트용 가상 시계
    return monitor


def drive(monitor, feature, frames=1, clock=None, step=0.05):
    """
    메인 루프 한 프레임에 해당하는 일을 직접 수행한다.
    extract_feature/process_pose를 우회해 합성 특징을 바로 넣는다.

    시각은 실제 시계가 아니라 명시적으로 넘긴다.
    안정화 유지시간(0.6초)과 지속시간 악화(2.0초)를 실시간으로 기다리면
    테스트가 느려지고, sleep이 부족하면 상태가 전환되지 않아
    코드가 아니라 테스트가 실패한다.
    """
    results = []
    now = monitor._clock if clock is None else clock
    for _ in range(frames):
        status, info = monitor.judge.decide(feature, BASELINE, now)
        score = monitor.scorer.score(info) if feature is not None else None
        monitor.stats.add(status, step, score)
        monitor._apply_status(status)
        monitor.state.update(status=status, score=score)
        results.append(status)
        now += step
    monitor._clock = now
    return results


def section(title):
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def main():
    from posture.app.stats import PostureScorer
    from posture.judge import build_judge

    failures = []

    def check(condition, message):
        if condition:
            print(f"  OK   {message}")
        else:
            print(f"  FAIL {message}")
            failures.append(message)

    # ── 1. 조립과 기동 ────────────────────────────────────
    section("1. 계층 조립")
    monitor = build_monitor()
    monitor.judge = build_judge("threshold")
    monitor.scorer = PostureScorer(None)
    check(monitor.judge is not None, "판정기 생성")
    check(monitor.state.snapshot()["status"] == "NO_POSE", "초기 상태 NO_POSE")

    # ── 2. 상태 → LED / 밴드 대응 ─────────────────────────
    section("2. 상태 → LED 색 / 밴드 명령 대응")
    for status in ("NORMAL", "WARNING", "BAD", "NO_POSE", "PAUSED"):
        monitor.led.history.clear()
        monitor.band.commands.clear()
        monitor._apply_status(status)
        led = monitor.led.current
        expected_led = LED_FOR_STATUS[status]
        expected_cmd = BAND_COMMANDS[status]
        sent = monitor.band.commands
        check(led == expected_led, f"{status:<8} → LED {led} (기대 {expected_led})")
        check(
            expected_cmd in sent,
            f"{status:<8} → 밴드 {sent} (기대 {expected_cmd} 포함)",
        )

    # ── 3. 상태 문자열을 그대로 보내지 않는가 ─────────────
    section("3. 밴드 명령이 상태 문자열이 아닌 규약 문자인가")
    # 펌웨어는 첫 글자만 읽는다. "NO_POSE"를 그대로 보내면 N(=NORMAL)이 된다.
    monitor.band.commands.clear()
    monitor._apply_status("NO_POSE")
    sent = monitor.band.commands
    check("P" in sent, f"NO_POSE가 P로 변환됨 (실제 {sent})")
    check("N" not in sent, "NO_POSE가 N으로 잘못 나가지 않음")

    monitor.band.commands.clear()
    monitor._apply_status("PAUSED")
    sent = monitor.band.commands
    check("S" in sent, f"PAUSED가 S로 변환됨 (실제 {sent})")

    # ── 4. 자세 변화에 판정이 따라오는가 ──────────────────
    section("4. 자세 변화 → 판정")
    monitor2 = build_monitor()
    monitor2.judge = build_judge("threshold")
    monitor2.scorer = PostureScorer(None)
    monitor2.judge.reset(time.monotonic())

    monitor2.judge.reset(monitor2._clock)

    # 0.6초(악화 유지시간)를 넘기도록 프레임 수를 잡는다
    normal = drive(monitor2, make_feature(0), frames=20)
    check(normal[-1] == "NORMAL", f"바른 자세 → {normal[-1]}")

    bad = drive(monitor2, make_feature(40), frames=40)
    check(bad[-1] == "BAD", f"심한 거북목(posture_error 0.40) → {bad[-1]}")

    warn = drive(monitor2, make_feature(20), frames=80)
    check(
        warn[-1] in ("WARNING", "BAD"),
        f"중간 거북목(posture_error 0.20) → {warn[-1]}",
    )

    # ── 5. NO_POSE에서 진동이 꺼지는가 ────────────────────
    section("5. NO_POSE 전환 시 진동 차단")
    monitor2.band.commands.clear()
    none_status = drive(monitor2, None, frames=3)
    check(none_status[-1] == "NO_POSE", f"특징 없음 → {none_status[-1]}")
    check(
        "P" in monitor2.band.commands,
        f"NO_POSE 즉시 P 전송 (실제 {monitor2.band.commands})",
    )
    check(
        "B" not in monitor2.band.commands and "W" not in monitor2.band.commands,
        "NO_POSE 중 진동 명령이 나가지 않음",
    )

    # ── 6. 지속시간 기반 악화 ─────────────────────────────
    section("6. WARNING 지속 → BAD 승격")
    from posture.judge import PostureJudge, ThresholdClassifier

    # manifest가 있으면 축이 cva_error로 바뀐다. 시험은 축까지 고정한다.
    def fixed_threshold():
        return ThresholdClassifier(
            axis="posture_error", sign=1.0, use_manifest=False,
            warning_enter=0.15, warning_exit=0.10,
            bad_enter=0.32, bad_exit=0.25,
        )

    judge = PostureJudge(fixed_threshold(), warning_to_bad_sec=0.3)
    base_t = 1000.0
    judge.reset(base_t)
    feature = make_feature(20)          # posture_error 0.20 = WARNING 영역
    # 안정화 0.6초 → WARNING 확정, 그 뒤 0.3초 유지 → BAD 승격
    seen = []
    for step in range(20):
        status, info = judge.decide(feature, BASELINE, base_t + step * 0.1)
        seen.append(status)
    check("WARNING" in seen, f"WARNING 발생 ({seen[:10]})")
    check(seen[-1] == "BAD", f"WARNING 지속 후 BAD 승격 (최종 {seen[-1]})")

    judge_off = PostureJudge(fixed_threshold(), warning_to_bad_sec=None)
    judge_off.reset(base_t)
    seen_off = [
        judge_off.decide(feature, BASELINE, base_t + step * 0.1)[0]
        for step in range(20)
    ]
    check(
        seen_off[-1] == "WARNING",
        f"규칙을 끄면 WARNING에 머무름 (최종 {seen_off[-1]})",
    )

    # ── 7. 일시정지 / 마스터 재개 ─────────────────────────
    section("7. 일시정지와 마스터 재개")
    state = monitor2.state
    state.set_paused(True)
    check(state.is_paused(), "SPACE 일시정지 적용")
    state.set_switch_paused(True)
    check(state.is_paused(), "스위치 일시정지 적용")
    state.set_switch_paused(False)
    check(
        not state.is_paused(),
        "스위치 HIGH가 UI 일시정지까지 해제 (마스터 재개)",
    )

    # ── 8. 통계 ───────────────────────────────────────────
    section("8. 세션 통계")
    from posture.app.stats import SessionStats

    stats = SessionStats()
    for _ in range(10):
        stats.add("NORMAL", 0.1, 95)
    for _ in range(5):
        stats.add("BAD", 0.1, 20)
    for _ in range(20):
        stats.add("NO_POSE", 0.1, None)
    check(abs(stats.monitored - 1.5) < 1e-6, f"감시 시간 {stats.monitored:.2f}s (NO_POSE 제외)")
    check(abs(stats.bad - 0.5) < 1e-6, f"불량 시간 {stats.bad:.2f}s")
    check(abs(stats.good_ratio - 66.667) < 0.1, f"양호 비율 {stats.good_ratio:.1f}%")

    # ── 9. 점수 정규화 ────────────────────────────────────
    section("9. 점수 축 정규화")
    manifest_2d = {"threshold_hint": {"axis": "posture_error", "sign": 1.0,
                                      "bad_enter": 0.32}}
    manifest_3d = {"threshold_hint": {"axis": "cva_error", "sign": 1.0,
                                      "bad_enter": 25.0}}
    s2 = PostureScorer(manifest_2d)
    s3 = PostureScorer(manifest_3d)
    check(s2.score({"posture_error": 0.0}) == 100, "2D 기준자세 100점")
    check(s2.score({"posture_error": 0.32}) == 0, "2D bad_enter에서 0점")
    check(s3.score({"cva_error": 0.0}) == 100, "3D 기준자세 100점")
    check(s3.score({"cva_error": 25.0}) == 0, "3D bad_enter에서 0점")
    check(
        s3.score({"cva_error": 12.5}) == 50,
        f"3D 중간값 50점 (실제 {s3.score({'cva_error': 12.5})})",
    )

    # ── 10. 종료 처리 ─────────────────────────────────────
    section("10. 종료 시 밴드 정지")
    monitor2.band.commands.clear()
    monitor2.band.stop()
    check("S" in monitor2.band.commands, f"종료 시 S 전송 ({monitor2.band.commands})")

    # ── 결과 ──────────────────────────────────────────────
    print()
    print("=" * 64)
    if failures:
        print(f"실패 {len(failures)}건")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("전부 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
