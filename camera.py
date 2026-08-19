#!/usr/bin/env python3
"""
hardware/camera.py - CSI 카메라 입력

두 개의 클래스가 있다.

    RpicamCapture     rpicam-vid의 MJPEG stdout을 프레임으로 바꾼다
    LatestFrameCamera 별도 스레드로 계속 읽고, 최신 프레임만 보관한다

모듈화 전에는 런타임과 수집 스크립트가 각자 RpicamCapture를 갖고 있었다.
화각 옵션과 품질이 달라 학습 데이터와 실기 입력이 어긋났다.
이제 이 파일 하나만 존재하며, 설정은 전부 contract.py에서 온다.

────────────────────────────────────────────────────────────
왜 "최신 프레임만" 쓰는가

MediaPipe 처리가 카메라보다 느리면 프레임이 파이프에 쌓인다.
순서대로 꺼내 쓰면 화면과 판정이 실제 시각보다 몇 초씩 뒤처진다.
자세 알림에서 몇 초 지연은 사실상 오작동이다.

그래서 읽을 때마다 파이프를 비우고 가장 최근 완성 프레임만 남긴다.
프레임은 버려지지만 지연은 쌓이지 않는다.
────────────────────────────────────────────────────────────
"""

import os
import shutil
import subprocess
import tempfile
import threading
import time

import cv2
import numpy as np

from ..contract import (
    CAMERA_FATAL_TIMEOUT_SEC,
    CAMERA_FPS,
    CAMERA_REOPEN_SEC,
    CAMERA_SENSOR_MODE,
    CAMERA_WAIT_SEC,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MJPEG_QUALITY,
)

__all__ = ["RpicamCapture", "LatestFrameCamera", "CameraUnavailable"]


class CameraUnavailable(RuntimeError):
    """카메라를 열 수 없거나 스트림이 끊겼다."""


class RpicamCapture:
    """rpicam-vid MJPEG 스트림 → BGR 프레임."""

    JPEG_START = b"\xff\xd8"
    JPEG_END = b"\xff\xd9"
    READ_TIMEOUT_SEC = 5.0
    MAX_BUFFER_BYTES = 8_000_000

    def __init__(
        self,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        fps=CAMERA_FPS,
        camera_index=0,
        sensor_mode=CAMERA_SENSOR_MODE,
    ):
        rpicam_path = shutil.which("rpicam-vid")
        if rpicam_path is None:
            raise CameraUnavailable("rpicam-vid 명령을 찾을 수 없습니다.")

        command = [
            rpicam_path, "--nopreview", "--timeout", "0",
            "--camera", str(camera_index),
        ]

        # 센서 모드를 지정하지 않으면 드라이버가 센터 크롭 모드를 고를 수 있다.
        # 그러면 화각이 좁아져 골반이 프레임 밖으로 나가고 3D 특징이 전부 NaN이 된다.
        # 자세한 이유는 contract.CAMERA_SENSOR_MODE 주석 참고.
        if sensor_mode:
            command += ["--mode", sensor_mode]

        command += [
            "--width", str(width), "--height", str(height),
            "--framerate", str(fps),
            "--codec", "mjpeg", "--quality", str(MJPEG_QUALITY),
            "--flush",              # 인코딩 끝난 프레임을 즉시 내보낸다
            "--output", "-",
        ]

        self.width = width
        self.height = height
        self.sensor_mode = sensor_mode
        self.buffer = bytearray()
        self._release_lock = threading.Lock()

        # stderr를 PIPE로 두고 읽지 않으면 파이프 버퍼가 차서 rpicam-vid가 멈춘다.
        # 긴 세션에서 카메라가 조용히 정지하는 원인이다. 임시 파일로 받는다.
        self._stderr_file = tempfile.TemporaryFile()

        self.process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=self._stderr_file, bufsize=0
        )

        # 논블로킹이어야 "쌓인 것 전부 비우고 최신만 쓰기"가 가능하다.
        os.set_blocking(self.process.stdout.fileno(), False)

        time.sleep(0.3)
        if self.process.poll() is not None:
            detail = self._read_stderr()
            hint = ""
            if sensor_mode and "mode" in detail.lower():
                hint = (
                    f"\n  센서 모드 '{sensor_mode}' 를 카메라가 지원하지 않는 것 같습니다."
                    "\n  IMX219(카메라 모듈 2)가 아니면 contract.CAMERA_SENSOR_MODE를"
                    "\n  빈 문자열로 두세요. 수집과 런타임 양쪽 다 바꿔야 합니다."
                )
            raise CameraUnavailable(
                "rpicam-vid를 시작하지 못했습니다. 다른 카메라 프로그램을 먼저 종료하세요.\n"
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
        지금 파이프에 있는 바이트를 전부 버퍼로 옮긴다.
        반환 False = stdout EOF (더 들어올 프레임 없음)
        """
        process = self.process
        stream = process.stdout if process is not None else None
        if stream is None or stream.closed:
            return False
        while True:
            try:
                data = stream.read(65536)
            except (BlockingIOError, InterruptedError):
                return True
            except (OSError, ValueError):
                return False
            if data is None:      # 논블로킹인데 아직 데이터 없음
                return True
            if not data:          # EOF
                return False
            self.buffer.extend(data)

    def _take_latest_frame(self):
        """버퍼에 쌓인 것 중 마지막 완성 프레임만 쓰고 나머지는 버린다."""
        end = self.buffer.rfind(self.JPEG_END)
        if end < 0:
            if len(self.buffer) > self.MAX_BUFFER_BYTES:
                self.buffer.clear()
            return None

        start = self.buffer.rfind(self.JPEG_START, 0, end)
        if start < 0:
            del self.buffer[: end + 2]
            return None

        jpeg_data = bytes(self.buffer[start : end + 2])
        del self.buffer[: end + 2]
        return cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)

    def read(self):
        """
        (성공여부, 프레임). 종료 판정은 프로세스 상태가 아니라 stdout EOF로 한다.
        rpicam-vid가 이미 끝났어도 파이프에 남은 프레임은 마저 써야 한다.
        """
        deadline = time.monotonic() + self.READ_TIMEOUT_SEC
        while True:
            alive = self._pump()
            frame = self._take_latest_frame()
            if frame is not None:
                return True, frame
            if not alive:
                return False, None
            if time.monotonic() >= deadline:
                return False, None
            time.sleep(0.002)

    def release(self):
        with self._release_lock:
            process = self.process
            self.process = None

        if process is None:
            return

        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

        if process.stdout is not None:
            process.stdout.close()
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None


class LatestFrameCamera:
    """
    카메라를 소유하고 최신 프레임만 노출한다.

    캡처 스레드가 계속 돌면서 프레임을 갈아 끼운다.
    소비자는 get_latest()로 "직전에 본 것과 다른 프레임"을 기다린다.
    카메라가 끊기면 자동으로 다시 연다.

    on_state는 카메라 가용 여부가 바뀔 때 호출된다.
    RuntimeState를 직접 import하지 않기 위한 콜백이다.
    하드웨어 계층이 앱 계층을 모르게 하는 것이 이 설계의 목적이다.
    """

    def __init__(self, on_state=None, camera_index=0):
        self._on_state = on_state
        self._camera_index = camera_index
        self._condition = threading.Condition()
        self._latest_frame = None
        self._frame_id = 0
        self._last_frame_time = None
        self._started_at = time.monotonic()
        self._last_error = None
        self._active_lock = threading.Lock()
        self._active_capture = None
        self._stop_event = threading.Event()
        self._thread = None

    def _notify_state(self, ok):
        if self._on_state is not None:
            self._on_state(ok)

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, name="csi-camera-capture", daemon=True
        )
        self._thread.start()

    def get_latest(self, previous_id=-1, timeout=CAMERA_WAIT_SEC):
        """(frame_id, frame, captured_at). 새 프레임이 없으면 frame이 None."""
        with self._condition:
            self._condition.wait_for(
                lambda: self._frame_id != previous_id or self._stop_event.is_set(),
                timeout=timeout,
            )
            if self._latest_frame is None or self._frame_id == previous_id:
                return previous_id, None, None
            return self._frame_id, self._latest_frame.copy(), self._last_frame_time

    def seconds_since_frame(self):
        with self._condition:
            reference = self._last_frame_time
        if reference is None:
            reference = self._started_at
        return time.monotonic() - reference

    def is_fatally_stalled(self):
        return self.seconds_since_frame() >= CAMERA_FATAL_TIMEOUT_SEC

    def stop(self):
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()

        with self._active_lock:
            capture = self._active_capture
        if capture is not None:
            capture.release()

        if self._thread is not None:
            self._thread.join(timeout=4.0)

    def _capture_loop(self):
        while not self._stop_event.is_set():
            capture = None
            try:
                print("[CAMERA] CSI 카메라 여는 중...")
                capture = RpicamCapture(camera_index=self._camera_index)
                with self._active_lock:
                    self._active_capture = capture

                self._last_error = None
                print(
                    f"[CAMERA] 준비됨. {FRAME_WIDTH}x{FRAME_HEIGHT} @ {CAMERA_FPS}fps, "
                    f"MJPEG q{MJPEG_QUALITY}, 센서 모드 "
                    f"{CAMERA_SENSOR_MODE or '자동'}"
                )

                while not self._stop_event.is_set():
                    success, frame = capture.read()
                    if not success or frame is None:
                        raise CameraUnavailable("CSI 카메라 스트림이 끊겼습니다.")

                    if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
                        frame = cv2.resize(
                            frame, (FRAME_WIDTH, FRAME_HEIGHT),
                            interpolation=cv2.INTER_AREA,
                        )

                    with self._condition:
                        self._latest_frame = frame
                        self._frame_id += 1
                        self._last_frame_time = time.monotonic()
                        self._condition.notify_all()
                    self._notify_state(True)

            except Exception as camera_error:
                self._notify_state(False)
                message = str(camera_error)
                # 같은 오류를 반복 출력하지 않는다. 재연결 중에는 매초 발생한다.
                if not self._stop_event.is_set() and message != self._last_error:
                    print(f"[CAMERA] {message}")
                    print("[CAMERA] 자동으로 다시 엽니다...")
                    self._last_error = message
            finally:
                with self._active_lock:
                    self._active_capture = None
                if capture is not None:
                    capture.release()

            if self._stop_event.wait(CAMERA_REOPEN_SEC):
                break
