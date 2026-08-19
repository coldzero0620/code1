#!/usr/bin/env python3
"""
collect_data.py - 라즈베리파이 카메라로 자세 데이터를 수집한다

한 번 실행에 한 라벨을 촬영해 posture_dataset.csv에 이어 쓴다.

    python3 collect_data.py --subject s01 --session a --label BAD --view side

진행 순서
    캘리브레이션  바른 자세를 유지해 이 사람의 기준점(baseline)을 잡는다
    준비          지정한 자세로 바꾸는 시간
    settle        수집은 시작하되 저장은 안 하는 구간 (자세 안정화)
    수집          --seconds 만큼 저장

--seconds는 실제 저장 시간이다. settle은 여기에 포함되지 않는다.

안전장치
    기존 CSV 헤더가 현재 스키마와 다르면 시작 전에 중단한다.
    캘리브레이션이 흔들렸거나 3D 좌표가 자주 실패하면 경고한다.
"""

import argparse
import csv
import os
import sys
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np

# 프로젝트 루트를 import 경로에 넣는다.
# tools/는 패키지가 아니라 실행 스크립트 모음이므로, 어디서 실행하든
# posture 패키지를 찾을 수 있어야 한다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posture.contract import (
    CAMERA_FPS,
    CAMERA_SENSOR_MODE,
    MJPEG_QUALITY,
    VIEWS,
    CSV_FIELDS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MODEL_COMPLEXITY,
    RUNTIME_STATUSES,
    SCHEMA_VERSION,
)
from posture.features import (
    BaselineCalibrator,
    view_is_lateral,
    create_pose,
    draw_feature,
    extract_feature,
    process_pose,
)

DEFAULT_CALIBRATION_SECONDS = 3.0
DEFAULT_PREPARE_SECONDS = 3.0
DEFAULT_SETTLE_SECONDS = 2.0

from posture.paths import DATASET_PATH, ensure_dirs
LABELS = list(RUNTIME_STATUSES)

# 캘리브레이션 중 signed_delta IQR이 이 값을 넘으면 자세가 흔들린 것으로 본다.
BASELINE_SPREAD_LIMIT = 0.08


class RpicamCapture:
    JPEG_START = b"\xff\xd8"
    JPEG_END = b"\xff\xd9"

    READ_TIMEOUT_SEC = 5.0

    def __init__(self, width=FRAME_WIDTH, height=FRAME_HEIGHT,
                 fps=CAMERA_FPS, camera_index=0, sensor_mode=CAMERA_SENSOR_MODE):
        rpicam_path = shutil.which("rpicam-vid")
        if rpicam_path is None:
            raise RuntimeError("rpicam-vid command was not found.")

        command = [
            rpicam_path, "--nopreview", "--timeout", "0",
            "--camera", str(camera_index),
        ]

        # 센서 모드. contract.CAMERA_SENSOR_MODE가 단일 출처이며
        # 런타임(V12.2 메인)과 반드시 같아야 한다. 이유는 contract.py 주석 참고.
        if sensor_mode:
            command += ["--mode", sensor_mode]

        command += [
            "--width", str(width), "--height", str(height),
            "--framerate", str(fps),
            "--codec", "mjpeg", "--quality", str(MJPEG_QUALITY),
            # 인코딩이 끝난 프레임을 즉시 stdout으로 내보낸다.
            "--flush",
            "--output", "-",
        ]

        self.sensor_mode = sensor_mode

        # stderr를 PIPE로 두고 읽지 않으면, rpicam-vid가 경고를 많이 쓸 때
        # 파이프 버퍼가 차서 프로세스가 멈춘다. 긴 수집 세션에서 카메라가
        # 조용히 정지하는 원인이 된다. 임시 파일로 받아 필요할 때만 읽는다.
        self._stderr_file = tempfile.TemporaryFile()

        self.process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=self._stderr_file, bufsize=0
        )
        self.buffer = bytearray()

        # 논블로킹으로 읽어야 "쌓인 프레임을 전부 비우고 최신 것만 쓰기"가 된다.
        os.set_blocking(self.process.stdout.fileno(), False)

        time.sleep(0.3)
        if self.process.poll() is not None:
            detail = self._read_stderr()
            hint = ""
            if sensor_mode and "mode" in detail.lower():
                hint = (
                    f"\n  센서 모드 '{sensor_mode}' 를 카메라가 지원하지 않는 것 같습니다.\n"
                    "  IMX219(카메라 모듈 2)가 아니면 contract.CAMERA_SENSOR_MODE를\n"
                    "  빈 문자열로 두세요. 단, 런타임 쪽도 같이 바꿔야 합니다."
                )
            raise RuntimeError(
                "rpicam-vid could not start. Close other camera programs first.\n"
                f"  {detail}{hint}"
            )

    def _read_stderr(self, limit=800):
        try:
            self._stderr_file.seek(0)
            return self._stderr_file.read(limit).decode("utf-8", "replace").strip()
        except (OSError, ValueError):
            return ""

    def is_opened(self):
        return (
            self.process is not None
            and self.process.poll() is None
            and self.process.stdout is not None
        )

    def _pump(self):
        """
        지금 파이프에 들어와 있는 바이트를 전부 버퍼로 옮긴다.
        반환 False = stdout EOF (더 이상 들어올 프레임이 없음)
        """
        stream = self.process.stdout if self.process is not None else None
        if stream is None or stream.closed:
            return False
        while True:
            try:
                data = stream.read(65536)
            except (BlockingIOError, InterruptedError):
                return True
            except ValueError:    # 닫힌 스트림
                return False
            if data is None:      # 논블로킹인데 아직 데이터가 없음
                return True
            if not data:          # EOF
                return False
            self.buffer.extend(data)

    def _take_latest_frame(self):
        """
        버퍼에 여러 프레임이 쌓여 있으면 마지막 완성 프레임만 쓰고 나머지는 버린다.

        수집 루프가 MediaPipe 처리 속도에 묶여 카메라보다 느려지면
        오래된 프레임이 파이프에 계속 쌓인다. 그대로 순서대로 읽으면
        화면과 저장 내용이 실제 시각보다 몇 초씩 뒤처지고,
        --seconds 로 지정한 구간이 사람이 실제로 취한 자세와 어긋난다.
        런타임(V12.2)도 최신 프레임 덮어쓰기 방식이므로 동작을 맞춘다.
        """
        end = self.buffer.rfind(self.JPEG_END)
        if end < 0:
            # 시작 마커조차 없는 쓰레기가 계속 쌓이면 버린다
            if len(self.buffer) > 8_000_000:
                self.buffer.clear()
            return None

        start = self.buffer.rfind(self.JPEG_START, 0, end)
        if start < 0:
            del self.buffer[: end + 2]
            return None

        jpeg_data = bytes(self.buffer[start : end + 2])
        del self.buffer[: end + 2]
        return cv2.imdecode(
            np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR
        )

    def read(self):
        """
        가장 최근 완성 프레임을 반환한다.

        종료 판정은 프로세스 상태가 아니라 stdout EOF로 한다.
        rpicam-vid가 이미 끝났더라도 파이프에 남은 프레임은 마저 써야 한다.
        """
        deadline = time.monotonic() + self.READ_TIMEOUT_SEC
        while True:
            alive = self._pump()
            frame = self._take_latest_frame()
            if frame is not None:
                return True, frame
            if not alive:                       # EOF, 남은 프레임도 없음
                return False, None
            if time.monotonic() >= deadline:    # 카메라가 응답하지 않음
                return False, None
            time.sleep(0.002)

    def release(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        if self.process.stdout is not None:
            self.process.stdout.close()
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None


def show(window_name, frame, headless):
    """headless=True면 imshow를 건너뛴다. 반환 True면 사용자가 중단을 요청한 것."""
    if headless:
        return False
    cv2.imshow(window_name, frame)
    return (cv2.waitKey(1) & 0xFF) == ord("q")


def banner(frame, lines):
    y = 40
    for text, scale, color, thickness in lines:
        cv2.putText(frame, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
        y += 30


def calibrate_baseline(cap, pose, seconds, window_name, headless=False):
    calibrator = BaselineCalibrator(seconds=seconds, min_samples=10)
    calibrator.start(time.monotonic())

    while True:
        success, frame = cap.read()
        if not success or frame is None:
            raise RuntimeError("Camera stream stopped during calibration.")

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        feature = extract_feature(process_pose(pose, frame))
        now = time.monotonic()
        running = calibrator.feed(feature, now)

        draw_feature(frame, feature)
        banner(frame, [
            (f"CALIBRATING: {calibrator.remaining(now):.1f}s", 0.8, (0, 255, 255), 2),
            ("Keep a comfortable upright posture", 0.6, (255, 255, 255), 1),
            (f"samples: {calibrator.sample_count}", 0.6, (255, 255, 255), 1),
        ])
        if show(window_name, frame, headless):
            raise KeyboardInterrupt

        if not running:
            break

    if calibrator.baseline is None:
        raise RuntimeError("Calibration failed. Ear and shoulder were not detected enough.")

    if calibrator.spread is not None and calibrator.spread > BASELINE_SPREAD_LIMIT:
        print(
            f"[WARNING] Baseline spread(IQR)={calibrator.spread:.4f} > "
            f"{BASELINE_SPREAD_LIMIT}. 캘리브레이션 중 자세가 흔들렸습니다. "
            "이 세션은 버리고 다시 찍는 것을 권합니다."
        )

    return calibrator.baseline, calibrator.spread, calibrator.world_ratio


def prepare_for_collection(cap, pose, label, seconds, window_name, headless=False):
    start_time = time.monotonic()
    instruction = (
        "Move out of detectable side-pose view"
        if label == "NO_POSE"
        else f"Move into the {label} posture now"
    )

    while time.monotonic() - start_time < seconds:
        success, frame = cap.read()
        if not success or frame is None:
            raise RuntimeError("Camera stream stopped before collection.")

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        feature = extract_feature(process_pose(pose, frame))
        draw_feature(frame, feature)

        remaining = max(0.0, seconds - (time.monotonic() - start_time))
        banner(frame, [
            (f"PREPARE {label}: {remaining:.1f}s", 0.8, (0, 255, 255), 2),
            (instruction, 0.6, (255, 255, 255), 1),
        ])
        if show(window_name, frame, headless):
            raise KeyboardInterrupt


def ensure_dataset_header(path):
    """
    기존 파일이 있으면 헤더가 현재 CSV_FIELDS와 정확히 같은지 검사한다.
    다르면 즉시 실패한다. 열이 어긋난 채 append되면 학습이 조용히 망가진다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as file:
            header = next(csv.reader(file), [])
        if header != CSV_FIELDS:
            raise RuntimeError(
                f"기존 CSV 헤더가 현재 스키마와 다릅니다: {path}\n"
                f"  파일: {header}\n"
                f"  현재: {CSV_FIELDS}\n"
                "  파일을 다른 이름으로 옮기고 새로 수집하세요."
            )
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=CSV_FIELDS).writeheader()


LABEL_COLORS = {
    "NO_POSE": (160, 160, 160),
    "NORMAL": (0, 255, 0),
    "WARNING": (255, 0, 0),
    "BAD": (0, 0, 255),
}


def make_row(args, frame_id, feature, baseline):
    common = {
        "timestamp": f"{time.time():.6f}",
        "subject": args.subject,
        "session": args.session,
        "label": args.label,
        "frame_id": frame_id,
        "view": args.view,
        "model_complexity": MODEL_COMPLEXITY,
        "schema_version": SCHEMA_VERSION,
    }
    if feature is None:
        common.update({k: "" for k in CSV_FIELDS if k not in common})
        common["pose_detected"] = 0
        return common

    base = baseline or {}
    base_2d = float(base.get("signed_delta", 0.0))
    base_fwd = float(base.get("fwd_ratio", 0.0))
    base_cva = float(base.get("cva_deg", 0.0))
    base_torso = float(base.get("torso_angle_deg", 0.0))

    signed = float(feature["signed_delta"])
    fwd = feature.get("fwd_ratio")
    cva = feature.get("cva_deg")
    torso = feature.get("torso_angle_deg")
    has_world = fwd is not None and np.isfinite(fwd) and np.isfinite(cva)
    has_torso = has_world and torso is not None and np.isfinite(torso)

    common.update({
        "pose_detected": 1,
        "side": feature["side"],
        "facing": f"{feature['facing']:+.0f}",
        "signed_delta": f"{signed:.8f}",
        "abs_delta": f"{feature['abs_delta']:.8f}",
        "obliquity": f"{feature.get('obliquity', 0.0):.8f}",
        "baseline": f"{base_2d:.8f}",
        "posture_error": f"{signed - base_2d:.8f}",
        "world_ok": 1 if has_world else 0,
    })

    if has_world:
        common.update({
            "fwd_ratio": f"{fwd:.8f}",
            "cva_deg": f"{cva:.6f}",
            "baseline_fwd": f"{base_fwd:.8f}",
            "baseline_cva": f"{base_cva:.6f}",
            "fwd_error": f"{fwd - base_fwd:.8f}",
            "cva_error": f"{cva - base_cva:.6f}",
        })
    else:
        common.update({k: "" for k in
                       ("fwd_ratio", "cva_deg", "baseline_fwd",
                        "baseline_cva", "fwd_error", "cva_error")})

    if has_torso:
        common.update({
            "torso_angle_deg": f"{torso:.6f}",
            "baseline_torso": f"{base_torso:.6f}",
            "torso_error": f"{torso - base_torso:.6f}",
        })
    else:
        common.update({k: "" for k in
                       ("torso_angle_deg", "baseline_torso", "torso_error")})

    return common


def collect_session(args):
    ensure_dirs()   # data/ 가 없으면 첫 수집에서 실패한다
    dataset_path = Path(args.output).expanduser().resolve()
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_dataset_header(dataset_path)

    cap = None
    pose = None
    window_name = "Posture Dataset Collector"
    saved_rows = 0
    valid_frames = 0
    total_frames = 0
    lateral_frames = 0
    world_frames = 0

    try:
        cap = RpicamCapture(
            width=FRAME_WIDTH, height=FRAME_HEIGHT,
            fps=CAMERA_FPS, camera_index=args.camera,
            sensor_mode=args.sensor_mode,
        )
        print(
            f"[CAMERA] {FRAME_WIDTH}x{FRAME_HEIGHT} @ {CAMERA_FPS}fps  "
            f"MJPEG q{MJPEG_QUALITY}  "
            f"sensor mode: {args.sensor_mode or '자동 선택'}"
        )
        if not args.sensor_mode:
            print("[WARNING] 센서 모드가 지정되지 않았습니다. "
                  "화각이 런타임과 달라질 수 있습니다.")
        pose = create_pose()

        baseline = None
        if args.label != "NO_POSE":
            baseline, spread, world_ratio = calibrate_baseline(
                cap, pose, args.calibration, window_name, args.headless
            )
            print(f"[CALIBRATION] signed_delta={baseline['signed_delta']:+.6f}  "
                  f"fwd_ratio={baseline['fwd_ratio']:+.6f}  "
                  f"cva={baseline['cva_deg']:+.2f}도  IQR={spread:.6f}")
            print(f"[CALIBRATION] 3D world landmark 유효 비율: {world_ratio * 100:.1f}%")
            if world_ratio < 0.9:
                print("[WARNING] 3D 좌표가 자주 실패합니다. "
                      "각도 불변 특징을 쓰려면 상반신 전체가 보이도록 프레이밍하세요.")

        prepare_for_collection(
            cap, pose, args.label, args.prepare, window_name, args.headless
        )

        # settle 구간은 --seconds에 포함되지 않는다. 요청한 만큼 온전히 저장된다.
        total_seconds = args.settle + args.seconds
        start_time = time.monotonic()
        frame_id = 0
        progress_step = max(1, int(CAMERA_FPS) * 5)

        with dataset_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)

            while True:
                elapsed = time.monotonic() - start_time
                if elapsed >= total_seconds:
                    break

                success, frame = cap.read()
                if not success or frame is None:
                    raise RuntimeError("Camera stream stopped during collection.")

                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                feature = extract_feature(process_pose(pose, frame))

                total_frames += 1
                frame_id += 1
                settling = elapsed < args.settle

                if feature is not None:
                    valid_frames += 1
                    lateral_frames += 1 if view_is_lateral(feature) else 0
                    world_frames += 1 if feature.get("world_ok") else 0
                    draw_feature(frame, feature)

                wanted = (feature is None) if args.label == "NO_POSE" else (feature is not None)
                if wanted and not settling:
                    writer.writerow(make_row(args, frame_id, feature, baseline))
                    saved_rows += 1

                remaining = max(0.0, total_seconds - elapsed)
                detected = "POSE" if feature is not None else "NO POSE"
                lines = [
                    (f"LABEL: {args.label}", 0.8, LABEL_COLORS[args.label], 2),
                    (f"DETECTED: {detected}", 0.6, (255, 255, 255), 1),
                    (f"TIME: {remaining:.1f}s  SAVED: {saved_rows}", 0.6, (255, 255, 255), 1),
                    (f"SUBJECT: {args.subject}  SESSION: {args.session}", 0.55, (220, 220, 220), 1),
                ]
                if settling:
                    lines.append(("SETTLING (not saved)", 0.6, (0, 200, 255), 2))
                banner(frame, lines)

                if show(window_name, frame, args.headless):
                    break

                if args.headless and frame_id % progress_step == 0:
                    print(f"[PROGRESS] {remaining:.0f}s left  saved={saved_rows}  {detected}")

        collect_elapsed = time.monotonic() - start_time
        pose_ratio = (valid_frames / total_frames * 100.0) if total_frames else 0.0
        print(f"[DATASET] Saved rows: {saved_rows}")
        print(f"[DATASET] Pose-detected ratio: {pose_ratio:.1f}%")
        print(f"[DATASET] File: {dataset_path}")
        print(f"[DATASET] model_complexity={MODEL_COMPLEXITY}  schema_version={SCHEMA_VERSION}")
        if valid_frames:
            print(f"[DATASET] view={args.view}  정측면 판정 비율: "
                  f"{lateral_frames / valid_frames * 100:.1f}%  "
                  f"3D 유효: {world_frames / valid_frames * 100:.1f}%")

        if saved_rows == 0:
            reason = (
                "유효한 포즈가 계속 잡혀서" if args.label == "NO_POSE"
                else "귀/어깨가 한 번도 잡히지 않아서"
            )
            print(f"[WARNING] {reason} 저장된 행이 없습니다.")

        # ── 실측 처리 속도 ──────────────────────────────────
        # 최신 프레임 방식이라 루프 속도는 카메라가 아니라 MediaPipe가 결정한다.
        # 따라서 "몇 행이 저장됐는가"를 CAMERA_FPS로 재면 안 된다.
        # 대신 실제 처리 속도를 재서, 학습 프레임 간격이 런타임과
        # 얼마나 다른지를 직접 보여준다.
        processed_fps = total_frames / collect_elapsed if collect_elapsed > 0 else 0.0
        print(
            f"[DATASET] 실측 처리 속도: {processed_fps:.1f} fps "
            f"(카메라 요청 {CAMERA_FPS} fps, {collect_elapsed:.1f}초 동안 {total_frames}프레임)"
        )

        if processed_fps < CAMERA_FPS * 0.75:
            print(
                f"[WARNING] MediaPipe가 카메라 속도를 못 따라갑니다 "
                f"({processed_fps:.1f} < {CAMERA_FPS} fps)."
            )
            print(
                "          지연은 안 쌓이지만, 학습 데이터의 프레임 간격이 "
                "런타임과 달라집니다."
            )
            print(
                f"          런타임도 같은 속도로 떨어진다면 문제없습니다. "
                f"런타임이 더 빠르다면 contract.CAMERA_FPS를 "
                f"{int(processed_fps)} 근처로 낮춰 양쪽을 맞추세요."
            )

        # 저장 행 수는 카메라 fps가 아니라 실제로 처리한 프레임 수와 비교한다.
        expected_rows = processed_fps * args.seconds
        if saved_rows and expected_rows and saved_rows < expected_rows * 0.5:
            print(
                f"[WARNING] 처리한 프레임에 비해 저장된 행이 적습니다 "
                f"({saved_rows}행 / 기대 ~{int(expected_rows)}행). "
                "조명과 프레이밍을 확인하세요."
            )

        if args.label != "NO_POSE" and 0 < pose_ratio < 80.0:
            print(f"[WARNING] Pose-detected ratio가 낮습니다 ({pose_ratio:.1f}%).")

    finally:
        if pose is not None:
            pose.close()
        if cap is not None:
            cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--label", required=True, choices=LABELS)
    parser.add_argument("--view", default="side", choices=VIEWS,
                        help="촬영 시점. 각도 일반화 검증에 쓰인다. "
                             "side=정측면 oblique=비스듬히 low=아래 high=위")
    parser.add_argument("--seconds", type=float, default=20.0,
                        help="실제로 저장할 시간(초). settle은 여기에 포함되지 않는다.")
    parser.add_argument("--calibration", type=float, default=DEFAULT_CALIBRATION_SECONDS)
    parser.add_argument("--prepare", type=float, default=DEFAULT_PREPARE_SECONDS)
    parser.add_argument("--settle", type=float, default=DEFAULT_SETTLE_SECONDS,
                        help="수집 시작 후 버릴 시간(초). 자세 잡는 구간을 제외한다.")
    parser.add_argument("--headless", action="store_true",
                        help="디스플레이 없이 실행 (SSH 전용 접속 시)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--sensor-mode", default=CAMERA_SENSOR_MODE,
        help="rpicam-vid --mode 값. 기본은 IMX219 전체 화각. "
             "다른 카메라라 이 모드가 없으면 빈 문자열('')로 두세요. "
             "그 경우 런타임 쪽 설정도 같이 바꿔야 합니다.",
    )
    parser.add_argument("--output", default=str(DATASET_PATH))
    return parser.parse_args()


if __name__ == "__main__":
    try:
        collect_session(parse_args())
    except KeyboardInterrupt:
        print("[PROGRAM] Stopped by user.")
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}")
