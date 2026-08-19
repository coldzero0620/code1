#!/usr/bin/env python3
"""
ingest_video.py - 영상 파일에서 학습 데이터를 만든다

videos/ 폴더에 영상을 넣고 실행하면 posture_dataset.csv가 만들어진다.

    python3 ingest_video.py

파일명 규칙 (단일 라벨 영상)

    {subject}_{view}_{label}_{session}.mp4
    s01_side_BAD_a.mp4          s01, 정측면, BAD, 세션 a

기준점 영상

    {subject}_BASELINE.mp4              3D 기준점만 제공 (2D 특징은 못 씀)
    {subject}_{view}_BASELINE.mp4       해당 시점의 2D 기준점 + 3D 기준점

여러 자세가 섞인 영상은 segments.csv에 구간을 적는다.

    file,subject,view,session,start,end,label
    s01_side_MIXED_a.mp4,s01,side,a,0:00,0:30,BASELINE
    s01_side_MIXED_a.mp4,s01,side,a,0:40,1:20,NORMAL
    s01_side_MIXED_a.mp4,s01,side,a,1:30,2:10,BAD

segments.csv에 등록된 파일은 파일명 규칙 스캔에서 제외되므로 충돌하지 않는다.
구간 사이(0:30~0:40)는 적지 않으면 자동으로 버려진다.

같은 영상을 다시 처리하면 기존 행을 지우고 새로 쓴다. 중복이 쌓이지 않는다.

────────────────────────────────────────────────────────────
baseline에 관한 중요한 제약

  3D 특징(fwd_error, cva_error)은 사람당 기준점 하나면 충분하다.
  신체 좌표계라 카메라 각도와 무관하기 때문이다.

  2D 특징(posture_error)은 시점마다 기준점이 다르다.
  정측면에서 잰 값을 비스듬한 영상에 적용하면 틀린다.

  따라서 {subject}_BASELINE.mp4 하나만 찍으면 2D 특징은
  기준점이 없어 빈칸이 되고, 학습에서 해당 조합만 자동으로 제외된다.
  행 자체가 사라지지는 않으므로 3D 특징으로는 그대로 학습된다.
  2D도 쓰려면 시점마다 기준점 영상이 따로 필요하다.

  다른 시점의 기준점을 대신 쓰지 않는다. 값이 조용히 틀리는 것보다
  빈칸으로 남기고 학습에서 빼는 쪽이 안전하기 때문이다.
────────────────────────────────────────────────────────────
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np

# 프로젝트 루트를 import 경로에 넣는다.
# tools/는 패키지가 아니라 실행 스크립트 모음이므로, 어디서 실행하든
# posture 패키지를 찾을 수 있어야 한다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posture.contract import (
    BASELINE_LABEL,
    CAMERA_FPS,
    CSV_FIELDS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MODEL_COMPLEXITY,
    POSTURE_LABELS,
    SCHEMA_VERSION,
    SEGMENT_TRIM_SEC,
    VIDEO_EXTENSIONS,
    VIDEO_PROCESS_FPS,
    VIDEO_SAMPLE_FPS,
    VIEWS,
)
from posture.features import create_pose, extract_feature, process_pose
from posture.paths import DATASET_PATH as _DATASET_PATH, VIDEO_DIR, ensure_dirs

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO_DIR = VIDEO_DIR
DEFAULT_SEGMENTS = "segments.csv"
DATASET_PATH = _DATASET_PATH

VALID_SEGMENT_LABELS = set(POSTURE_LABELS) | {BASELINE_LABEL, "NO_POSE"}


# ─────────────────────────────────────────────────────────────
# 시간 파싱
# ─────────────────────────────────────────────────────────────
def parse_time(text) -> float:
    """'1:30' / '90' / '1:30.5' → 초"""
    text = str(text).strip()
    if not text:
        raise ValueError("빈 시간 값")
    parts = text.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except ValueError:
        pass
    raise ValueError(f"시간 형식을 알 수 없습니다: '{text}'")


# ─────────────────────────────────────────────────────────────
# 작업 목록 만들기
# ─────────────────────────────────────────────────────────────
class Segment:
    """영상 한 구간 = CSV 여러 행이 될 단위"""

    def __init__(self, path, subject, view, session, label, start=None, end=None):
        self.path = path
        self.subject = subject
        self.view = view
        self.session = session
        self.label = label
        self.start = start
        self.end = end

    @property
    def source(self):
        return self.path.name

    def __repr__(self):
        span = ""
        if self.start is not None:
            span = f" [{self.start:.0f}~{self.end:.0f}s]"
        return f"{self.path.name}:{self.label}{span}"


def parse_filename(path: Path):
    """
    {subject}_BASELINE            → (subject, None, None, BASELINE)
    {subject}_{view}_BASELINE     → (subject, view, None, BASELINE)
    {subject}_{view}_{label}_{session}
    인식 실패 시 None
    """
    parts = path.stem.split("_")

    if len(parts) == 2 and parts[1].upper() == BASELINE_LABEL:
        return parts[0], None, "baseline", BASELINE_LABEL

    if len(parts) == 3 and parts[2].upper() == BASELINE_LABEL:
        if parts[1] not in VIEWS:
            return None
        return parts[0], parts[1], "baseline", BASELINE_LABEL

    if len(parts) == 4:
        subject, view, label, session = parts
        if view not in VIEWS:
            return None
        if label.upper() not in VALID_SEGMENT_LABELS:
            return None
        return subject, view, session, label.upper()

    return None


def load_segments(segments_path: Path, video_dir: Path):
    """segments.csv → Segment 목록, 그리고 거기 등장한 파일명 집합"""
    if not segments_path.exists():
        return [], set()

    required = {"file", "subject", "view", "session", "start", "end", "label"}
    segments, listed, problems = [], set(), []

    with segments_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{segments_path.name}에 열이 없습니다: {sorted(missing)}\n"
                f"  필요한 열: {sorted(required)}"
            )

        for line_no, row in enumerate(reader, start=2):
            name = (row.get("file") or "").strip()
            if not name:
                continue
            path = video_dir / name
            listed.add(name)

            if not path.exists():
                problems.append(f"  {line_no}행: 파일 없음 '{name}'")
                continue

            label = (row.get("label") or "").strip().upper()
            if label not in VALID_SEGMENT_LABELS:
                problems.append(f"  {line_no}행: 알 수 없는 라벨 '{label}'")
                continue

            view = (row.get("view") or "").strip()
            if view not in VIEWS:
                problems.append(f"  {line_no}행: 알 수 없는 시점 '{view}'")
                continue

            try:
                start = parse_time(row["start"])
                end = parse_time(row["end"])
            except ValueError as error:
                problems.append(f"  {line_no}행: {error}")
                continue

            if end <= start:
                problems.append(f"  {line_no}행: end가 start보다 빠릅니다")
                continue

            segments.append(
                Segment(
                    path,
                    (row.get("subject") or "").strip(),
                    view,
                    (row.get("session") or "").strip(),
                    label,
                    start,
                    end,
                )
            )

    if problems:
        raise ValueError(
            f"{segments_path.name}에 문제가 있습니다:\n" + "\n".join(problems)
        )

    return segments, listed


def collect_work(video_dir: Path, segments_path: Path):
    if not video_dir.exists():
        raise FileNotFoundError(
            f"영상 폴더가 없습니다: {video_dir}\n"
            f"  폴더를 만들고 영상을 넣으세요."
        )

    segments, listed = load_segments(segments_path, video_dir)

    videos = sorted(
        path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix in VIDEO_EXTENSIONS
    )
    if not videos:
        raise FileNotFoundError(f"영상이 없습니다: {video_dir}")

    unknown = []
    for path in videos:
        if path.name in listed:
            continue  # segments.csv가 담당
        parsed = parse_filename(path)
        if parsed is None:
            unknown.append(path.name)
            continue
        subject, view, session, label = parsed
        segments.append(Segment(path, subject, view, session, label))

    if unknown:
        print("[WARNING] 파일명 규칙에 맞지 않아 건너뜁니다:")
        for name in unknown:
            print(f"  {name}")
        print("  규칙: {subject}_{view}_{label}_{session}.mp4")
        print(f"  시점: {VIEWS}")
        print("  여러 자세가 섞였다면 segments.csv에 구간을 적으세요.")

    return segments


# ─────────────────────────────────────────────────────────────
# 영상 처리
# ─────────────────────────────────────────────────────────────
def scan_video(pose, path: Path, sample_fps: float, process_fps: float,
               progress: bool = True):
    """
    영상을 처음부터 끝까지 한 번만 연속으로 훑는다.

    구간마다 seek 하면 안 된다. MediaPipe Pose는 static_image_mode=False에서
    직전 프레임의 추적 상태를 이어 쓰므로, 영상 중간으로 건너뛰면
    추적기가 새로 탐색을 시작해 전혀 다른 값이 나온다.

    실측: 같은 구간을 seek해서 읽으면 cva -16.5도,
          처음부터 연속으로 읽으면 -44.6도. 28도 차이다.

    영상당 한 번만 읽으므로 구간이 여러 개여도 처리 시간이 늘지 않는다.

    ────────────────────────────────────────────────────────────
    처리 간격과 저장 간격은 별개다.

      process_fps  MediaPipe에 통과시키는 간격. 런타임과 같아야 한다.
                   추적 상태가 이 간격으로 이어지기 때문이다.
      sample_fps   CSV에 저장하는 간격. 중복 제거가 목적이다.

    예전에는 이 둘이 하나였다. sample_fps=5로 두면 MediaPipe도 5fps로만
    돌아서, 호출 사이 200ms 동안의 움직임을 평활하게 된다.
    런타임은 50ms 간격이므로 같은 자세에서도 다른 값이 나왔다.
    ────────────────────────────────────────────────────────────
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {path.name}")

    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if not source_fps or source_fps <= 0 or source_fps > 240:
            source_fps = 30.0
        frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # process_fps가 0 이하이거나 원본보다 빠르면 전 프레임을 처리한다.
        process_all = process_fps <= 0 or process_fps >= source_fps
        process_interval = 0.0 if process_all else 1.0 / process_fps
        sample_interval = 1.0 / sample_fps if sample_fps > 0 else 0.0

        scanned = []
        index = 0
        processed = 0
        next_process = 0.0
        next_sample = 0.0
        started = time.monotonic()
        eps = 1e-9

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            position = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if position <= 0.0 and index > 0:
                position = index / source_fps   # POS_MSEC를 못 주는 코덱 대비
            index += 1

            # ── 처리할 프레임인가 ──
            if not process_all:
                if position + eps < next_process:
                    continue
                # 고정 스케줄로 전진시킨다. 실제 위치를 기준으로 다시 잡으면
                # 원본 fps 격자에 걸려 목표보다 느려진다.
                while next_process <= position + eps:
                    next_process += process_interval

            if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            feature = extract_feature(process_pose(pose, frame))
            processed += 1

            # ── 저장할 프레임인가 ──
            if sample_interval <= 0.0 or position + eps >= next_sample:
                scanned.append((position, feature))
                while next_sample <= position + eps:
                    next_sample += sample_interval

            if progress and frame_total and processed % 200 == 0:
                done = index / frame_total
                spent = time.monotonic() - started
                left = spent / max(done, 1e-6) - spent
                print(f"  {path.name}  {done * 100:5.1f}%  "
                      f"처리 {processed}프레임  남은 시간 ~{left:.0f}초",
                      end="\r", flush=True)

        if progress and frame_total:
            print(" " * 78, end="\r")

        spent = time.monotonic() - started
        rate = f"{index / spent:.0f}" if spent > 0 else "-"
        print(f"[SCAN] {path.name:<34} 원본 {index}프레임 "
              f"→ MediaPipe {processed} → 저장 {len(scanned)}  "
              f"({spent:.0f}초, {rate}프레임/초)")

        return scanned
    finally:
        capture.release()


def slice_segment(scanned, segment: Segment):
    """
    연속 스캔 결과에서 구간에 해당하는 부분만 잘라낸다.
    구간 앞뒤 SEGMENT_TRIM_SEC 초는 자세 전환 프레임이므로 버린다.
    """
    if segment.start is None:
        return list(scanned)

    start = segment.start + SEGMENT_TRIM_SEC
    end = max(start, segment.end - SEGMENT_TRIM_SEC)
    return [(t, f) for t, f in scanned if start <= t <= end]


class VideoScanner:
    """
    영상별 스캔 결과를 캐시한다.

    Pose 인스턴스는 영상마다 새로 만든다. 한 인스턴스로 여러 영상을
    처리하면 앞 영상의 추적 상태가 뒤 영상에 새어 들어가,
    결과가 파일 처리 순서에 따라 달라진다.
    """

    def __init__(self, sample_fps: float, process_fps: float):
        self.sample_fps = sample_fps
        self.process_fps = process_fps
        self._cache = {}

    def get(self, segment: Segment):
        path = segment.path
        if path not in self._cache:
            pose = create_pose()
            try:
                self._cache[path] = scan_video(
                    pose, path, self.sample_fps, self.process_fps
                )
            finally:
                pose.close()
        part = slice_segment(self._cache[path], segment)
        return part, len(part)


def median_of(values):
    clean = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(clean)) if clean else None


# ─────────────────────────────────────────────────────────────
# CSV 쓰기
# ─────────────────────────────────────────────────────────────
def load_existing(path: Path):
    """기존 CSV를 읽어 source별로 묶는다. 스키마가 다르면 중단한다."""
    if not path.exists() or path.stat().st_size == 0:
        return {}

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_FIELDS:
            raise RuntimeError(
                f"기존 CSV 스키마가 다릅니다: {path}\n"
                f"  파일: {reader.fieldnames}\n"
                f"  현재: {CSV_FIELDS}\n"
                f"  파일을 다른 이름으로 옮기고 다시 실행하세요."
            )
        grouped = defaultdict(list)
        for row in reader:
            grouped[row.get("source", "")].append(row)
    return dict(grouped)


def make_row(segment, frame_id, feature, baseline_2d, baseline_3d):
    row = {key: "" for key in CSV_FIELDS}
    row.update({
        "timestamp": f"{frame_id}",
        "subject": segment.subject,
        "session": segment.session,
        "label": segment.label,
        "frame_id": frame_id,
        "view": segment.view,
        "source": segment.source,
        "model_complexity": MODEL_COMPLEXITY,
        "schema_version": SCHEMA_VERSION,
    })

    if feature is None:
        row["pose_detected"] = 0
        return row

    signed = float(feature["signed_delta"])
    fwd = feature.get("fwd_ratio")
    cva = feature.get("cva_deg")
    torso = feature.get("torso_angle_deg")
    has_world = (
        fwd is not None and cva is not None
        and np.isfinite(fwd) and np.isfinite(cva)
    )
    has_torso = has_world and torso is not None and np.isfinite(torso)

    row.update({
        "pose_detected": 1,
        "side": feature["side"],
        "facing": f"{feature['facing']:+.0f}",
        "signed_delta": f"{signed:.8f}",
        "abs_delta": f"{feature['abs_delta']:.8f}",
        "obliquity": f"{feature.get('obliquity', 0.0):.8f}",
        "world_ok": 1 if has_world else 0,
    })

    if baseline_2d is not None:
        row["baseline"] = f"{baseline_2d:.8f}"
        row["posture_error"] = f"{signed - baseline_2d:.8f}"

    if has_world:
        row["fwd_ratio"] = f"{fwd:.8f}"
        row["cva_deg"] = f"{cva:.6f}"
        if baseline_3d is not None:
            base_fwd, base_cva, base_torso = baseline_3d
            row["baseline_fwd"] = f"{base_fwd:.8f}"
            row["baseline_cva"] = f"{base_cva:.6f}"
            row["fwd_error"] = f"{fwd - base_fwd:.8f}"
            row["cva_error"] = f"{cva - base_cva:.6f}"

            # 몸통 각도는 3D가 잡혀도 어깨선이 수직에 가까우면 못 낼 수 있다.
            if has_torso and base_torso is not None:
                row["torso_angle_deg"] = f"{torso:.6f}"
                row["baseline_torso"] = f"{base_torso:.6f}"
                row["torso_error"] = f"{torso - base_torso:.6f}"

    return row


# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    video_dir = Path(args.videos).expanduser().resolve()
    segments_path = video_dir / args.segments
    ensure_dirs()   # data/ 와 videos/ 가 없으면 첫 실행에서 실패한다
    dataset_path = Path(args.output).expanduser().resolve()

    work = collect_work(video_dir, segments_path)
    if not work:
        raise ValueError("처리할 구간이 없습니다.")

    baseline_work = [s for s in work if s.label == BASELINE_LABEL]
    posture_work = [s for s in work if s.label != BASELINE_LABEL]

    print(f"[INGEST] 영상 폴더: {video_dir}")
    print(f"[INGEST] 기준점 구간 {len(baseline_work)}개, 자세 구간 {len(posture_work)}개")
    process_label = (
        "원본 전 프레임" if args.process_fps <= 0 else f"{args.process_fps:.1f} fps"
    )
    print(f"[INGEST] MediaPipe 처리 {process_label}, "
          f"CSV 저장 {args.fps:.1f} fps, model_complexity={MODEL_COMPLEXITY}")
    if args.process_fps > 0 and abs(args.process_fps - CAMERA_FPS) > 0.5:
        print(f"[WARNING] 처리 속도({args.process_fps:.1f})가 "
              f"contract.CAMERA_FPS({CAMERA_FPS})와 다릅니다. "
              "런타임과 프레임 간격이 어긋납니다.")

    scanner = VideoScanner(args.fps, args.process_fps)

    try:
        # ── 1단계: 기준점 ────────────────────────────────
        raw_2d = defaultdict(list)      # (subject, view) → signed_delta
        raw_3d = defaultdict(list)      # subject → (fwd, cva)
        raw_torso = defaultdict(list)   # subject → torso_angle_deg

        for segment in baseline_work:
            features, total = scanner.get(segment)
            valid = [f for _, f in features if f is not None]
            if not valid:
                print(f"[WARNING] {segment}: 포즈가 한 번도 안 잡혔습니다. 건너뜁니다.")
                continue

            for feature in valid:
                raw_2d[(segment.subject, segment.view)].append(feature["signed_delta"])
                fwd, cva = feature.get("fwd_ratio"), feature.get("cva_deg")
                if fwd is not None and np.isfinite(fwd) and np.isfinite(cva):
                    raw_3d[segment.subject].append((fwd, cva))
                torso = feature.get("torso_angle_deg")
                if torso is not None and np.isfinite(torso):
                    raw_torso[segment.subject].append(float(torso))

            world = sum(1 for f in valid if f.get("world_ok")) / len(valid)
            print(f"[BASELINE] {segment.path.name:<34} "
                  f"프레임 {total:>5}  포즈 {len(valid) / max(total, 1) * 100:5.1f}%  "
                  f"3D {world * 100:5.1f}%")

        baseline_2d = {key: median_of(values) for key, values in raw_2d.items()}
        baseline_3d = {}
        for subject, pairs in raw_3d.items():
            # 몸통 각도는 3D가 잡혀도 못 낼 수 있으므로 None을 허용한다.
            torso_values = raw_torso.get(subject)
            baseline_3d[subject] = (
                median_of([p[0] for p in pairs]),
                median_of([p[1] for p in pairs]),
                median_of(torso_values) if torso_values else None,
            )

        subjects_needed = {s.subject for s in posture_work}
        no_3d = sorted(subjects_needed - set(baseline_3d))
        if no_3d:
            print(f"[WARNING] 3D 기준점 없는 사람: {no_3d} → 3D 특징을 못 씁니다.")

        # ── 2단계: 자세 구간 ─────────────────────────────
        existing = load_existing(dataset_path)
        replaced = sorted({s.source for s in posture_work} & set(existing))
        if replaced:
            print(f"[INGEST] 재처리로 교체되는 영상 {len(replaced)}개")

        fresh = defaultdict(list)
        summary = []

        for segment in posture_work:
            features, total = scanner.get(segment)
            valid = [f for _, f in features if f is not None]

            # 2D 기준점은 같은 시점(view)에서 잰 것만 쓴다.
            # signed_delta는 화면 투영값이므로 카메라 각도가 바뀌면
            # 똑같은 자세라도 값이 달라진다. 다른 시점의 기준점을 빌려오면
            # posture_error가 오류 없이 조용히 틀린 값이 된다.
            # 없으면 None으로 두고 make_row가 posture_error를 비운다.
            #
            # 3D 기준점은 신체 좌표계라 시점과 무관하므로 사람 단위로 공유한다.
            base2 = baseline_2d.get((segment.subject, segment.view))
            base3 = baseline_3d.get(segment.subject)

            want_pose = segment.label != "NO_POSE"
            rows = []
            for index, (_, feature) in enumerate(features):
                if want_pose and feature is None:
                    continue
                if not want_pose and feature is not None:
                    continue
                rows.append(make_row(segment, index, feature, base2, base3))

            fresh[segment.source].extend(rows)
            world = (
                sum(1 for f in valid if f.get("world_ok")) / len(valid) if valid else 0.0
            )
            summary.append((segment, total, len(valid), len(rows), world, base2))

        # ── 3단계: 저장 ──────────────────────────────────
        merged = []
        for source, rows in existing.items():
            if source not in fresh:
                merged.extend(rows)
        for rows in fresh.values():
            merged.extend(rows)

        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with dataset_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(merged)

        # ── 보고 ────────────────────────────────────────
        print()
        print(f"{'영상':<34}{'라벨':<9}{'프레임':>7}{'포즈%':>7}{'3D%':>7}{'저장':>7}")
        print("-" * 71)
        for segment, total, valid, saved, world, base2 in summary:
            print(f"{segment.path.name:<34}{segment.label:<9}{total:>7}"
                  f"{valid / max(total, 1) * 100:>6.1f}%{world * 100:>6.1f}%{saved:>7}")

        print()
        print(f"[INGEST] 저장: {dataset_path}  총 {len(merged)}행")

        low_pose = [s for s, t, v, _, _, _ in summary
                    if s.label != "NO_POSE" and v / max(t, 1) < 0.9]
        low_3d = [s for s, _, v, _, w, _ in summary
                  if s.label != "NO_POSE" and v and w < 0.9]
        no_base2 = [s for s, _, _, _, _, b in summary if b is None]

        if low_pose:
            print(f"[WARNING] 포즈 검출률 90% 미만: {[str(s) for s in low_pose]}")
            print("          조명과 프레이밍을 확인하세요.")
        if low_3d:
            print(f"[WARNING] 3D 유효율 90% 미만: {[str(s) for s in low_3d]}")
            print("          골반이 프레임에 들어오는지 확인하세요.")
        if no_base2:
            views = sorted({(s.subject, s.view) for s in no_base2})
            print(f"[WARNING] 2D 기준점이 없는 (사람, 시점): {views}")
            print("          해당 시점의 BASELINE 영상이 없어 2D 특징이 비어 있습니다.")
            print("          3D 특징만으로 학습됩니다. 문제되지 않습니다.")

        print()
        print("[NEXT] python3 check_dataset.py 로 학습 가능 여부를 확인하세요.")

    finally:
        pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="영상 파일에서 학습 데이터를 만든다",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--videos", default=str(DEFAULT_VIDEO_DIR),
                        help="영상 폴더 (기본: ../videos)")
    parser.add_argument("--segments", default=DEFAULT_SEGMENTS,
                        help="구간 표 파일명. 영상 폴더 안에 둔다")
    parser.add_argument("--fps", type=float, default=VIDEO_SAMPLE_FPS,
                        help="초당 몇 프레임을 CSV에 저장할지. "
                             "낮추면 중복이 준다 (처리 속도와는 무관)")
    parser.add_argument("--process-fps", type=float, default=VIDEO_PROCESS_FPS,
                        help="초당 몇 프레임을 MediaPipe에 통과시킬지. "
                             "런타임 처리 속도와 같아야 한다. "
                             "0을 주면 원본 전 프레임을 처리한다")
    parser.add_argument("--output", default=str(DATASET_PATH))
    return parser.parse_args()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[PROGRAM] 사용자가 중단했습니다.")
        sys.exit(1)
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}")
        sys.exit(1)
