#!/usr/bin/env python3
"""
check_dataset.py - 학습이 가능한 상태인지 즉시 알려준다

    python3 check_dataset.py

train_model.py를 몇 분씩 돌려보고 나서야 데이터가 부족한 것을 알게 되는
상황을 막기 위한 도구다. 몇 초 안에 끝난다.

확인하는 것
    스키마 버전이 맞는가
    세 자세가 다 있는가
    클래스마다 독립 세션이 2개 이상인가
    사람이 2명 이상인가 (cross-subject 평가에 필요)
    시점이 2종 이상인가 (cross-view 평가에 필요)
    3D 특징을 쓸 수 있는가
    학습에 얼마나 걸릴 것인가

마지막에 학습 가능 / 불가를 한 줄로 판정한다.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 프로젝트 루트를 import 경로에 넣는다.
# tools/는 패키지가 아니라 실행 스크립트 모음이므로, 어디서 실행하든
# posture 패키지를 찾을 수 있어야 한다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posture.contract import (
    ALL_FEATURES,
    FEATURE_CANDIDATES,
    POSTURE_LABELS,
    SCHEMA_VERSION,
    VIEWS,
)

from posture.paths import DATASET_PATH

# train_model.py의 탐색 규모 (8 조합 x 8 파라미터 x 5 fold)
SEARCH_FITS = 8 * 8 * 5
# 1행당 1회 학습 시간(초) 실측 근사. 머신에 따라 다르다.
SECONDS_PER_ROW_PER_FIT = 9.2e-5 / 1000


class Report:
    def __init__(self):
        self.blocking = []
        self.warnings = []
        self.notes = []

    def block(self, message, fix):
        self.blocking.append((message, fix))

    def warn(self, message, fix):
        self.warnings.append((message, fix))

    def note(self, message):
        self.notes.append(message)


def section(title):
    print()
    print(title)
    print("-" * max(len(title), 40))


def load(path: Path, report: Report):
    if not path.exists():
        report.block(
            f"데이터 파일이 없습니다: {path.name}",
            "ingest_video.py 또는 collect_data.py로 먼저 데이터를 만드세요.",
        )
        return None

    df = pd.read_csv(path)
    if df.empty:
        report.block("데이터가 비어 있습니다.", "수집을 다시 하세요.")
        return None

    if "schema_version" in df.columns:
        versions = pd.to_numeric(df["schema_version"], errors="coerce").dropna()
        stale = sorted({int(v) for v in versions if int(v) != SCHEMA_VERSION})
        if stale:
            report.block(
                f"스키마 버전이 다른 행이 있습니다: {stale} (현재 {SCHEMA_VERSION})",
                "예전 포맷 데이터입니다. 해당 행을 지우거나 다시 수집하세요.",
            )

    df["label"] = df["label"].astype(str).str.upper().str.strip()
    for column in ALL_FEATURES:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    df["subject"] = df["subject"].astype(str).str.strip()
    df["session"] = df["session"].astype(str).str.strip()
    df["group"] = df["subject"] + "::" + df["session"]
    return df


def check_labels(df, report):
    section("자세 분포")
    counts = df["label"].value_counts()
    for label in POSTURE_LABELS + ["NO_POSE"]:
        n = int(counts.get(label, 0))
        mark = "OK " if n > 0 or label == "NO_POSE" else "★ "
        print(f"  {mark}{label:<9} {n:>7} 행")

    missing = [l for l in POSTURE_LABELS if counts.get(l, 0) == 0]
    if missing:
        report.block(
            f"없는 자세: {missing}",
            "세 자세가 모두 있어야 학습됩니다. 해당 자세를 촬영하세요.",
        )

    posture = df[df["label"].isin(POSTURE_LABELS)]
    if not posture.empty:
        share = posture["label"].value_counts(normalize=True)
        if share.max() > 0.6:
            report.warn(
                f"한 자세가 전체의 {share.max() * 100:.0f}%를 차지합니다 "
                f"({share.idxmax()})",
                "자세별 촬영 시간을 비슷하게 맞추면 좋습니다. 치명적이지는 않습니다.",
            )
    return posture


def check_sessions(posture, report):
    section("세션 (train/test 분할의 단위)")
    table = posture.groupby("label")["group"].nunique()
    for label in POSTURE_LABELS:
        n = int(table.get(label, 0))
        mark = "OK " if n >= 2 else "★ "
        print(f"  {mark}{label:<9} 세션 {n}개")

    thin = [l for l in POSTURE_LABELS if table.get(l, 0) < 2]
    if thin:
        report.block(
            f"세션이 2개 미만인 자세: {thin}",
            "같은 자세를 다른 세팅에서 한 번 더 촬영하세요.\n"
            "        세션이 1개면 train/test를 나눌 수 없습니다.",
        )

    small = posture.groupby("group").size()
    tiny = small[small < 30]
    if not tiny.empty:
        report.warn(
            f"행이 30개 미만인 세션 {len(tiny)}개: {list(tiny.index)[:5]}",
            "촬영이 너무 짧거나 포즈 검출이 실패했을 수 있습니다.",
        )


def check_subjects(posture, report):
    section("사람 (cross-subject 평가)")
    subjects = sorted(posture["subject"].unique())
    print(f"  {len(subjects)}명: {subjects}")

    if len(subjects) < 2:
        report.block(
            "사람이 1명뿐입니다.",
            "cross-subject 평가가 불가능합니다. 최소 2명, 권장 3명.\n"
            "        이 숫자가 대회에서 가장 중요한 성능 지표입니다.",
        )
    elif len(subjects) < 3:
        report.warn(
            f"사람이 {len(subjects)}명입니다.",
            "3명 이상을 권장합니다. 2명이면 일반화 성능을 신뢰하기 어렵습니다.",
        )

    per_subject = posture.groupby("subject")["label"].nunique()
    incomplete = per_subject[per_subject < 3]
    if not incomplete.empty:
        report.warn(
            f"세 자세가 다 있지 않은 사람: {list(incomplete.index)}",
            "LOSO 평가에서 해당 사람이 제외됩니다.",
        )


def check_views(posture, report):
    section("촬영 시점 (cross-view 평가)")
    if "view" not in posture.columns:
        report.warn("view 열이 없습니다.", "예전 방식으로 수집된 데이터입니다.")
        return

    views = sorted(v for v in posture["view"].dropna().astype(str).unique() if v)
    print(f"  {len(views)}종: {views}")

    unknown = [v for v in views if v not in VIEWS]
    if unknown:
        report.warn(f"알 수 없는 시점: {unknown}", f"허용: {VIEWS}")

    if len(views) < 2:
        report.warn(
            "시점이 1종뿐입니다.",
            "cross-view 평가를 못 합니다. 각도 다양성이 목표라면\n"
            "        다른 각도에서도 촬영하세요. 학습 자체는 됩니다.",
        )


def check_features(posture, report):
    section("특징 가용성")
    usable = []
    for columns in FEATURE_CANDIDATES:
        have = [c for c in columns if c in posture.columns]
        if len(have) < len(columns):
            print(f"  ★  {str(columns):<45} 열 없음")
            continue
        ratio = float(posture[columns].notna().all(axis=1).mean())
        mark = "OK " if ratio >= 0.5 else "★  "
        print(f"  {mark}{str(columns):<45} 유효 {ratio * 100:5.1f}%")
        if ratio >= 0.5:
            usable.append(columns)

    if not usable:
        report.block(
            "쓸 수 있는 특징 조합이 없습니다.",
            "기준점(BASELINE) 영상이 없거나 포즈 검출이 실패했습니다.",
        )
        return

    has_3d = any("fwd_error" in c or "cva_error" in c for c in usable)
    has_2d = any("posture_error" in c for c in usable)
    if not has_3d:
        report.warn(
            "3D 특징을 쓸 수 없습니다.",
            "골반이 프레임에 안 들어왔거나 기준점 영상이 없습니다.\n"
            "        카메라 각도가 바뀌면 성능이 크게 떨어집니다.",
        )
    if not has_2d:
        report.note(
            "2D 특징이 없습니다. 시점별 BASELINE 영상이 없어서입니다. "
            "3D만으로 학습되며 문제되지 않습니다."
        )


def check_quality(df, posture, report):
    section("데이터 품질")
    if "world_ok" in df.columns:
        detected = df[pd.to_numeric(df["pose_detected"], errors="coerce") == 1]
        if not detected.empty:
            ratio = float(
                pd.to_numeric(detected["world_ok"], errors="coerce").fillna(0).mean()
            )
            mark = "OK " if ratio >= 0.9 else "★  "
            print(f"  {mark}3D world landmark 유효율  {ratio * 100:5.1f}%")
            if ratio < 0.9:
                report.warn(
                    f"3D 유효율이 {ratio * 100:.1f}%입니다.",
                    "골반이 프레임에 들어오는지 확인하고 다시 촬영하세요.",
                )

    if "obliquity" in posture.columns:
        obl = pd.to_numeric(posture["obliquity"], errors="coerce").dropna()
        if not obl.empty:
            print(f"  -  시점 지표(obliquity) 중앙값  {obl.median():.3f} "
                  f"(0=정측면)")

    if "source" in df.columns:
        sources = df["source"].dropna().astype(str)
        sources = sources[sources != ""]
        if not sources.empty:
            print(f"  -  출처 {sources.nunique()}개")


def estimate_time(posture):
    section("예상 학습 시간")
    rows = len(posture)
    seconds = rows * SECONDS_PER_ROW_PER_FIT * SEARCH_FITS
    print(f"  {rows:,}행 → 약 {seconds / 60:.1f}분 (대략적 추정)")
    if seconds > 900:
        print("  오래 걸립니다. ingest_video.py --fps 를 낮춰 프레임을 줄이면")
        print("  학습이 빨라지고 중복 데이터도 줄어듭니다.")


def main():
    report = Report()
    print("=" * 60)
    print("데이터셋 점검")
    print("=" * 60)
    print(f"파일: {DATASET_PATH}")

    df = load(DATASET_PATH, report)
    if df is not None:
        posture = check_labels(df, report)
        if not posture.empty:
            check_sessions(posture, report)
            check_subjects(posture, report)
            check_views(posture, report)
            check_features(posture, report)
            check_quality(df, posture, report)
            estimate_time(posture)

    section("판정")
    if report.blocking:
        print("  ★ 학습 불가")
        for message, fix in report.blocking:
            print(f"\n  문제: {message}")
            print(f"  해결: {fix}")
    else:
        print("  OK 학습 가능")
        print("\n  다음: python3 train_model.py")

    if report.warnings:
        print()
        print("  경고 (학습은 되지만 성능에 영향)")
        for message, fix in report.warnings:
            print(f"\n  - {message}")
            print(f"    {fix}")

    if report.notes:
        print()
        for message in report.notes:
            print(f"  참고: {message}")

    print()
    return 1 if report.blocking else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}")
        sys.exit(2)
