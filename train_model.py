#!/usr/bin/env python3
"""
train_model.py - 수집한 CSV로 모델을 학습한다

    python3 train_model.py

사람이 정할 숫자가 없다. 특징 조합과 하이퍼파라미터를 교차검증으로
자동 선택하고, 결과를 models/ 아래 두 파일로 남긴다.

    posture-rf.joblib      학습된 모델
    split_manifest.json    특징 목록, 파라미터, 임계값, 성능 기록

선택 기준
    score = 균형정확도 - 0.5 x (BAD를 NORMAL로 놓친 비율)
    거북목을 놓치는 비용이 헛경고보다 크기 때문이다.
    성능이 비슷하면(0.005 이내) 더 가벼운 모델을 고른다.

교차검증은 반드시 세션 단위로 나눈다. 같은 세션의 프레임끼리는
거의 동일하므로, 프레임을 섞어 나누면 정답을 보고 시험 치는 셈이 된다.

출력에서 봐야 할 두 숫자
    [SUBJECT] Mean cross-subject   처음 보는 사람에 대한 성능
    [VIEW]    Mean cross-view      처음 보는 카메라 각도에 대한 성능
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

# 프로젝트 루트를 import 경로에 넣는다.
# tools/는 패키지가 아니라 실행 스크립트 모음이므로, 어디서 실행하든
# posture 패키지를 찾을 수 있어야 한다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posture.contract import (
    ALL_FEATURES,
    FEATURE_CANDIDATES,
    LABEL_MAP,
    POSTURE_LABELS,
    RUNTIME_STATUSES,
    SCHEMA_VERSION,
)

from posture.paths import DATASET_PATH, MODEL_DIR, MODEL_PATH, SPLIT_PATH

TEST_SIZE = 0.25
RANDOM_STATE = 42

# 트리 수는 그대로 추론 지연이 된다. 60이 120과 같은 점수를 내면 60을 쓴다.
# 그리드는 성글게 잡는다. 동점 시 더 가벼운 쪽을 고르는 규칙이 있으므로
# 촘촘한 격자는 탐색 시간만 늘리고 결과를 거의 바꾸지 않는다.
# 특징 후보가 8종으로 늘어난 만큼 파라미터 쪽을 줄여 균형을 맞춘다.
PARAM_GRID = [
    {"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf}
    for n in (60, 120)
    for d in (4, 8)
    for leaf in (5, 20)
]

MISS_PENALTY = 0.5  # BAD → NORMAL 오분류에 붙이는 벌점 가중치

# 탐색 단계에서 쓸 최대 행 수.
# 영상 데이터는 인접 프레임이 거의 동일해 전부 쓸 이유가 없다.
# 조합 비교에는 이 정도면 충분하고, 최종 학습은 전체 데이터로 한다.
SEARCH_MAX_ROWS = 12000


# ─────────────────────────────────────────────────────────────
# 데이터
# ─────────────────────────────────────────────────────────────
def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    required = {"subject", "session", "label"} | set(ALL_FEATURES)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            "collect_data.py v2로 다시 수집했는지 확인하세요."
        )

    if "schema_version" in df.columns:
        versions = pd.to_numeric(df["schema_version"], errors="coerce").dropna().unique()
        stale = [int(v) for v in versions if int(v) != SCHEMA_VERSION]
        if stale:
            raise ValueError(
                f"CSV schema_version {stale} != {SCHEMA_VERSION}. "
                "예전 포맷 행이 섞여 있습니다. 파일을 비우고 다시 수집하세요."
            )

    df = df.copy()
    df["label"] = df["label"].astype(str).str.upper().str.strip()

    no_pose_rows = int((df["label"] == "NO_POSE").sum())
    if no_pose_rows:
        print(f"[DATASET] NO_POSE rows: {no_pose_rows} (excluded from RF training)")

    df = df[df["label"].isin(POSTURE_LABELS)].copy()

    for column in ALL_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    # 포즈만 잡히면 항상 계산되는 값만 필수로 둔다.
    #
    # posture_error(2D)는 그 시점의 기준점 영상이, fwd_error/cva_error(3D)는
    # 사람별 기준점 영상이 있어야 생긴다. 이것들을 여기서 요구하면
    # 기준점이 없는 시점의 데이터가 통째로 사라지고,
    # "3D 특징만으로 학습" 경로가 실제로는 동작하지 않게 된다.
    #
    # 어떤 조합을 쓸지는 search_best_config가 유효 행 비율을 보고 정하고,
    # 최종 학습 직전에 선택된 열에 대해서만 dropna한다.
    required = ["signed_delta", "abs_delta"]
    df = df.dropna(subset=required + ["subject", "session", "label"])

    world_ratio = float(df["fwd_error"].notna().mean()) if "fwd_error" in df else 0.0
    error_ratio = float(df["posture_error"].notna().mean()) if "posture_error" in df else 0.0
    print(f"[DATASET] 3D world landmark 유효 비율: {world_ratio * 100:.1f}%")
    print(f"[DATASET] 2D 기준점(posture_error) 유효 비율: {error_ratio * 100:.1f}%")
    if 0.0 < world_ratio < 0.9:
        print("[WARNING] 3D 좌표가 자주 실패합니다. 조명과 상반신 프레이밍을 확인하세요.")
    if 0.0 < error_ratio < 0.9:
        print("[WARNING] 2D 기준점이 없는 시점이 있습니다. "
              "해당 행은 2D 조합 탐색에서만 빠지고 3D 조합에는 그대로 쓰입니다.")

    if df.empty:
        raise ValueError("No valid NORMAL/WARNING/BAD rows remain after cleaning.")

    df["target"] = df["label"].map(LABEL_MAP).astype(int)
    df["subject"] = df["subject"].astype(str).str.strip()
    df["session"] = df["session"].astype(str).str.strip()
    df["group"] = df["subject"] + "::" + df["session"]

    absent = [label for label in POSTURE_LABELS if label not in set(df["label"])]
    if absent:
        raise ValueError(f"Missing posture classes in dataset: {absent}")

    return df


def find_group_split(df):
    counts = df.groupby("label")["group"].nunique().to_dict()
    thin = {label: counts.get(label, 0) for label in POSTURE_LABELS if counts.get(label, 0) < 2}
    if thin:
        raise ValueError(
            f"Need at least 2 independent sessions per class. Current: {counts}"
        )

    y = df["target"].to_numpy()
    groups = df["group"].to_numpy()
    dummy_x = np.zeros((len(df), 1), dtype=np.float32)
    required = set(LABEL_MAP.values())

    for offset in range(500):
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE + offset
        )
        train_idx, test_idx = next(splitter.split(dummy_x, y, groups))
        if set(y[train_idx]) == required and set(y[test_idx]) == required:
            return train_idx, test_idx, RANDOM_STATE + offset

    raise ValueError(
        "Could not create a session-level split containing all three classes. "
        "Collect more independent sessions."
    )


# ─────────────────────────────────────────────────────────────
# 점수
# ─────────────────────────────────────────────────────────────
def bad_to_normal_rate(y_true, y_pred):
    labels = [LABEL_MAP[label] for label in POSTURE_LABELS]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    bad_row = matrix[LABEL_MAP["BAD"]]
    total = int(bad_row.sum())
    return (int(bad_row[LABEL_MAP["NORMAL"]]) / total) if total else 0.0


def combined_score(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred) - MISS_PENALTY * bad_to_normal_rate(
        y_true, y_pred
    )


def make_cv(df):
    """세션 단위 층화 K-fold. 세션이 적으면 fold 수를 줄인다."""
    n_groups = df["group"].nunique()
    n_splits = int(min(5, max(2, n_groups // 2)))
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def cross_val_score_config(df, columns, params, cv):
    df = df.dropna(subset=columns)
    if len(df) < 60:
        return float("-inf")
    x = df[columns].to_numpy(dtype=np.float32)
    y = df["target"].to_numpy(dtype=np.int64)
    groups = df["group"].to_numpy()

    scores = []
    for train_idx, test_idx in cv.split(x, y, groups):
        if set(y[train_idx]) != set(LABEL_MAP.values()):
            continue
        if len(set(y[test_idx])) < 2:
            continue
        model = RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params,
        )
        model.fit(x[train_idx], y[train_idx])
        scores.append(combined_score(y[test_idx], model.predict(x[test_idx])))

    return float(np.mean(scores)) if scores else float("-inf")


def subsample_for_search(df):
    """
    탐색용 부분집합. 세션 구성은 그대로 두고 세션 안에서만 솎아낸다.
    세션을 통째로 빼면 교차검증 구조가 바뀌므로 그렇게 하지 않는다.
    """
    if len(df) <= SEARCH_MAX_ROWS:
        return df

    # groupby.apply는 그룹 키를 열에서 지운다. 인덱스만 골라내 원본에서 뽑는다.
    fraction = SEARCH_MAX_ROWS / len(df)
    keep = []
    for _, part in df.groupby(["group", "label"], sort=False):
        if len(part) <= 20:
            keep.append(part.index)
        else:
            take = max(20, int(len(part) * fraction))
            keep.append(
                part.sample(n=min(take, len(part)),
                            random_state=RANDOM_STATE).index
            )
    sampled = df.loc[np.concatenate([np.asarray(k) for k in keep])]
    print(f"[SEARCH] 탐색용 부분집합 {len(df)} → {len(sampled)}행 "
          f"(최종 학습은 전체 데이터 사용)")
    return sampled


def search_best_config(full_df):
    """특징 조합 × 하이퍼파라미터를 세션 단위 CV로 비교한다."""
    df = subsample_for_search(full_df)
    cv = make_cv(df)
    print(f"[SEARCH] StratifiedGroupKFold n_splits={cv.get_n_splits()}")
    print(f"[SEARCH] {len(FEATURE_CANDIDATES)} feature sets x {len(PARAM_GRID)} params")

    # 후보마다 쓸 수 있는 행이 다르다.
    # 기준점이 없어 posture_error가 빈 시점이 있으면 2D 조합은 그 행을 못 쓴다.
    # 즉 후보들의 CV 점수는 서로 다른 데이터에서 나온 값이므로
    # 그대로 비교하면 "적은 데이터에서 잰 쉬운 점수"가 유리해진다.
    # 비율을 반드시 같이 출력해서 이 비대칭이 보이게 한다.
    usable = []
    coverage = {}
    for columns in FEATURE_CANDIDATES:
        ratio = float(df[columns].notna().all(axis=1).mean())
        coverage[tuple(columns)] = ratio
        if ratio < 0.5:
            print(f"[SEARCH] {str(columns):<45} 건너뜀 (유효 행 {ratio * 100:.0f}%)")
            continue
        usable.append(columns)

    if not usable:
        raise ValueError("사용 가능한 특징 조합이 없습니다. 데이터를 확인하세요.")

    results = []
    for columns in usable:
        best_for_set = None
        for params in PARAM_GRID:
            score = cross_val_score_config(df, columns, params, cv)
            results.append((score, columns, params))
            if best_for_set is None or score > best_for_set[0]:
                best_for_set = (score, params)
        print(
            f"[SEARCH] {str(columns):<40} 행 {coverage[tuple(columns)] * 100:5.1f}%  "
            f"best CV score={best_for_set[0]:.4f} "
            f"{best_for_set[1]}"
        )

    top = max(result[0] for result in results)
    # 동점(0.005 이내)이면 특징 수 → 트리 수 → 깊이 순으로 더 가벼운 쪽을 고른다.
    tied = [r for r in results if r[0] >= top - 0.005]
    tied.sort(
        key=lambda r: (
            len(r[1]),
            r[2]["n_estimators"],
            r[2]["max_depth"] or 99,
            -r[0],
        )
    )
    score, columns, params = tied[0]

    print(f"[SEARCH] Selected features : {columns}")
    print(f"[SEARCH] Selected params   : {params}")
    print(f"[SEARCH] CV score          : {score:.4f}  (top was {top:.4f})")
    if len(tied) > 1:
        print(f"[SEARCH] {len(tied)} configs were within 0.005 - picked the cheapest.")

    # 선택된 조합이 다른 후보보다 훨씬 적은 행에서 평가됐다면 경고한다.
    # 점수가 높은 이유가 "더 좋아서"가 아니라 "더 쉬운 부분집합에서 쟀기 때문"일 수 있다.
    picked_ratio = coverage[tuple(columns)]
    best_ratio = max(coverage[tuple(c)] for c in usable)
    if picked_ratio < best_ratio - 0.05:
        print(
            f"[WARNING] 선택된 조합은 전체의 {picked_ratio * 100:.0f}% 행만 씁니다 "
            f"(다른 후보는 최대 {best_ratio * 100:.0f}%)."
        )
        print(
            "          기준점이 없는 시점의 데이터가 학습에서 빠졌다는 뜻입니다. "
            "CV 점수가 더 쉬운 부분집합에서 나온 값일 수 있으니"
        )
        print(
            "          시점별 BASELINE 영상을 추가하거나, 3D 전용 조합의 점수와 "
            "직접 비교해 보세요."
        )

    # 특징이 1개면 런타임에서 구간 테이블(LUT)로 컴파일되어 추론이 수백 배 빨라진다.
    # 선택된 조합이 2개 이상이면, 가장 좋은 단일 특징과의 차이를 알려준다.
    if len(columns) > 1:
        singles = [r for r in results if len(r[1]) == 1]
        if singles:
            best_single = max(singles, key=lambda r: r[0])
            gap = score - best_single[0]
            print(
                f"[SEARCH] 참고: 단일 특징 {best_single[1]} 는 CV {best_single[0]:.4f} "
                f"(선택안보다 {gap:+.4f})."
            )
            print(
                "[SEARCH]       단일 특징은 LUT로 컴파일되어 추론이 수백 배 빠르다. "
                "Pi에서 프레임이 모자라면 이쪽을 고려할 것."
            )

    return columns, params, float(score)


# ─────────────────────────────────────────────────────────────
# 평가 / threshold
# ─────────────────────────────────────────────────────────────
def evaluate(y_true, y_pred):
    labels = [LABEL_MAP[label] for label in POSTURE_LABELS]
    balanced = balanced_accuracy_score(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=POSTURE_LABELS,
        output_dict=True,
        zero_division=0,
    )
    miss = bad_to_normal_rate(y_true, y_pred)

    print(f"[EVAL] Balanced Accuracy: {balanced:.4f}")
    for label in POSTURE_LABELS:
        print(
            f"[EVAL] {label:<7} Precision: {report[label]['precision']:.4f}  "
            f"Recall: {report[label]['recall']:.4f}  "
            f"F1: {report[label]['f1-score']:.4f}"
        )
    print(f"[EVAL] BAD -> NORMAL Rate: {miss:.4f}")
    print("[EVAL] Confusion Matrix rows=true, columns=predicted (NORMAL, WARNING, BAD)")
    print(matrix)

    return {
        "balanced_accuracy": float(balanced),
        "classification_report": report,
        "bad_to_normal_rate": float(miss),
        "confusion_matrix": matrix.tolist(),
    }


def apply_threshold_rule(error, warning_enter, bad_enter):
    out = np.zeros(error.shape, dtype=np.int64)
    out[error >= warning_enter] = LABEL_MAP["WARNING"]
    out[error >= bad_enter] = LABEL_MAP["BAD"]
    return out


def pick_threshold_axis(columns):
    """
    threshold 폴백이 쓸 축을 고른다.
    모델이 실제로 쓰는 특징 중 하나여야 한다.
    2D 모델에 3D 축의 임계값을 물리면 전부 NORMAL로 판정된다.
    """
    for preferred in ("posture_error", "cva_error", "fwd_error",
                      "torso_error", "signed_delta", "cva_deg", "fwd_ratio"):
        if preferred in columns:
            return preferred
    return columns[0]


def tune_thresholds(df, axis="posture_error"):
    """
    지정한 축에서 threshold 폴백 경계를 격자 탐색으로 구한다.

    부호 방향도 자동으로 판정한다.
    posture_error는 나쁠수록 커지지만, cva_error/fwd_error는 나쁠수록
    음수로 작아진다. 방향을 고정하면 3D 축에서 전부 NORMAL이 나온다.
    """
    error = df[axis].to_numpy(dtype=np.float64)
    y_all = df["target"].to_numpy(dtype=np.int64)

    # BAD의 중앙값이 NORMAL보다 작으면 축을 뒤집는다.
    bad_med = float(np.median(error[y_all == LABEL_MAP["BAD"]]))
    normal_med = float(np.median(error[y_all == LABEL_MAP["NORMAL"]]))
    sign = -1.0 if bad_med < normal_med else 1.0
    if sign < 0:
        print(f"[THRESHOLD] 축 '{axis}' 는 나쁠수록 작아짐 → 부호 반전 적용")
    error = error * sign
    y = df["target"].to_numpy(dtype=np.int64)

    lo, hi = np.percentile(error, [1, 99])
    grid = np.linspace(lo, hi, 160)

    best = None
    for i, warning_enter in enumerate(grid[:-1]):
        for bad_enter in grid[i + 1 :]:
            score = combined_score(y, apply_threshold_rule(error, warning_enter, bad_enter))
            if best is None or score > best[0]:
                best = (score, float(warning_enter), float(bad_enter))

    score, warning_enter, bad_enter = best

    # 히스테리시스 폭 = 두 경계 간격의 30%. 경계에서 떨리는 것을 막는다.
    margin = max((bad_enter - warning_enter) * 0.30, 1e-3)
    warning_exit = warning_enter - margin
    bad_exit = bad_enter - margin

    print(f"[THRESHOLD] {axis} distribution (sign={sign:+.0f} 적용 후)")
    for label in POSTURE_LABELS:
        values = error[y == LABEL_MAP[label]]
        print(
            f"[THRESHOLD] {label:<7} n={values.size:<6} "
            f"p10={np.percentile(values, 10):+.4f}  "
            f"median={np.percentile(values, 50):+.4f}  "
            f"p90={np.percentile(values, 90):+.4f}"
        )
    print(f"[THRESHOLD] Grid-searched rule score: {score:.4f}")
    print("[THRESHOLD] Suggested values (posture/judge가 manifest에서 자동으로 읽습니다)")
    print(f"[THRESHOLD]   warning_enter = {warning_enter:+.4f}")
    print(f"[THRESHOLD]   warning_exit  = {warning_exit:+.4f}")
    print(f"[THRESHOLD]   bad_enter     = {bad_enter:+.4f}")
    print(f"[THRESHOLD]   bad_exit      = {bad_exit:+.4f}")

    if bad_enter - warning_enter < 0.02:
        print(
            "[WARNING] WARNING과 BAD 경계가 거의 붙어 있습니다. "
            "두 자세를 더 뚜렷하게 나눠 다시 수집하세요."
        )

    return {
        "axis": axis,
        "sign": float(sign),
        "warning_enter": warning_enter,
        "warning_exit": warning_exit,
        "bad_enter": bad_enter,
        "bad_exit": bad_exit,
        "rule_score": float(score),
    }


def report_view_generalization(df, columns, params, all_views=None):
    """
    촬영 시점(view)을 하나씩 빼고 평가한다.
    "다른 각도에서 찍어도 되는가"에 대한 답이다.

    all_views는 특징 dropna 이전의 시점 목록이다.
    촬영은 여러 각도로 했는데 선택된 특징 때문에 일부 시점이 통째로
    빠지는 경우가 있어, 그 둘을 구분해서 안내한다.
    """
    if "view" not in df.columns:
        return None

    views = sorted(v for v in df["view"].dropna().astype(str).unique() if v)
    if len(views) < 2:
        if all_views and len(all_views) > len(views):
            missing = sorted(set(all_views) - set(views))
            print(
                f"[VIEW] 시점 {sorted(all_views)} 로 촬영했지만, 선택된 특징 {columns} 은 "
                f"{missing} 시점에서 계산할 수 없어 학습에서 빠졌습니다."
            )
            print(
                "[VIEW] 그래서 각도 일반화를 측정할 수 없습니다. "
                "해당 시점의 BASELINE 영상을 추가하세요."
            )
        else:
            print(
                f"[VIEW] 시점이 {len(views)}종뿐입니다. 각도 일반화를 측정할 수 없습니다. "
                "collect_data.py --view 로 여러 각도에서 수집하세요."
            )
        return None

    scores = []
    print("[VIEW] Leave-One-View-Out evaluation")
    for held_out in views:
        mask = df["view"].astype(str) == held_out
        train_part, test_part = df[~mask], df[mask]
        if train_part.empty or test_part.empty:
            continue
        if set(train_part["target"]) != set(LABEL_MAP.values()):
            print(f"[VIEW]   {held_out:<10} skipped (train missing a class)")
            continue
        if len(set(test_part["target"])) < 2:
            print(f"[VIEW]   {held_out:<10} skipped (test has <2 classes)")
            continue

        fold = RandomForestClassifier(
            class_weight="balanced_subsample", random_state=RANDOM_STATE,
            n_jobs=-1, **params,
        )
        fold.fit(train_part[columns].to_numpy(dtype=np.float32),
                 train_part["target"].to_numpy(dtype=np.int64))
        pred = fold.predict(test_part[columns].to_numpy(dtype=np.float32))
        truth = test_part["target"].to_numpy()
        score = balanced_accuracy_score(truth, pred)
        scores.append(float(score))
        print(f"[VIEW]   held-out={held_out:<10} BA: {score:.4f}  "
              f"BAD->NORMAL: {bad_to_normal_rate(truth, pred):.4f}")

    if not scores:
        return None

    mean_score = float(np.mean(scores))
    print(f"[VIEW] Mean cross-view Balanced Accuracy: {mean_score:.4f}")
    print("[VIEW] 이 숫자가 처음 보는 카메라 각도에 대한 실제 성능입니다.")
    return {"per_view_balanced_accuracy": scores, "mean_balanced_accuracy": mean_score}


def report_subject_generalization(df, columns, params):
    subjects = sorted(df["subject"].unique().tolist())
    if len(subjects) < 2:
        print(
            f"[SUBJECT] Only {len(subjects)} subject(s). Cross-subject generalization "
            "cannot be measured. Collect data from at least 2 people."
        )
        return None

    scores, misses = [], []
    print("[SUBJECT] Leave-One-Subject-Out evaluation")
    for held_out in subjects:
        mask = df["subject"] == held_out
        train_part, test_part = df[~mask], df[mask]

        if train_part.empty or test_part.empty:
            continue
        if set(train_part["target"]) != set(LABEL_MAP.values()):
            print(f"[SUBJECT]   {held_out:<10} skipped (train missing a class)")
            continue
        if len(set(test_part["target"])) < 2:
            print(f"[SUBJECT]   {held_out:<10} skipped (test has <2 classes)")
            continue

        fold = RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params,
        )
        fold.fit(
            train_part[columns].to_numpy(dtype=np.float32),
            train_part["target"].to_numpy(dtype=np.int64),
        )
        pred = fold.predict(test_part[columns].to_numpy(dtype=np.float32))
        truth = test_part["target"].to_numpy()
        score = balanced_accuracy_score(truth, pred)
        miss = bad_to_normal_rate(truth, pred)
        scores.append(float(score))
        misses.append(float(miss))
        print(
            f"[SUBJECT]   held-out={held_out:<10} BA: {score:.4f}  "
            f"BAD->NORMAL: {miss:.4f}"
        )

    if not scores:
        return None

    mean_score = float(np.mean(scores))
    print(f"[SUBJECT] Mean cross-subject Balanced Accuracy: {mean_score:.4f}")
    print("[SUBJECT] 이 숫자가 처음 보는 사람에 대한 실제 성능입니다.")
    return {
        "per_subject_balanced_accuracy": scores,
        "per_subject_bad_to_normal": misses,
        "mean_balanced_accuracy": mean_score,
        "mean_bad_to_normal_rate": float(np.mean(misses)),
    }


# ─────────────────────────────────────────────────────────────
def main():
    df = load_dataset()
    print(f"[DATASET] Rows: {len(df)}  Sessions: {df['group'].nunique()}  "
          f"Subjects: {df['subject'].nunique()}")
    summary = df.groupby(["group", "label"], as_index=False).size()
    print(summary.sort_values(["label", "group"]).to_string(index=False))

    columns, params, cv_score = search_best_config(df)

    # 특징 dropna 이전의 시점 목록. 어떤 시점이 특징 때문에 빠졌는지 알려면 필요하다.
    all_views = (
        sorted(v for v in df["view"].dropna().astype(str).unique() if v)
        if "view" in df.columns else []
    )
    rows_before = len(df)

    df = df.dropna(subset=columns).reset_index(drop=True)
    if len(df) < rows_before:
        print(
            f"[DATASET] 선택된 특징 기준으로 {rows_before} → {len(df)}행 "
            f"({(rows_before - len(df)) / rows_before * 100:.0f}% 제외)"
        )
    train_idx, test_idx, split_seed = find_group_split(df)
    train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

    model = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **params,
    )
    model.fit(
        train_df[columns].to_numpy(dtype=np.float32),
        train_df["target"].to_numpy(dtype=np.int64),
    )

    metrics = evaluate(
        test_df["target"].to_numpy(dtype=np.int64),
        model.predict(test_df[columns].to_numpy(dtype=np.float32)),
    )

    threshold_hint = tune_thresholds(df, axis=pick_threshold_axis(columns))
    subject_report = report_subject_generalization(df, columns, params)
    view_report = report_view_generalization(df, columns, params, all_views)

    # 런타임에서 스레드 풀을 띄우지 않도록 저장 전에 고정한다.
    # 1행 예측에서는 병렬화가 오히려 손해다.
    model.n_jobs = 1

    complexity = None
    if "model_complexity" in df.columns:
        values = pd.to_numeric(df["model_complexity"], errors="coerce").dropna().unique()
        if len(values) == 1:
            complexity = int(values[0])
        elif len(values) > 1:
            print(f"[WARNING] Dataset mixes model_complexity {sorted(values)}.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "runtime_statuses": RUNTIME_STATUSES,
        "no_pose_rule": "Return NO_POSE before model inference when ear/shoulder pose is unavailable.",
        "model_classes": POSTURE_LABELS,
        "feature_columns": columns,
        "label_map": LABEL_MAP,
        "model_params": params,
        "cv_score": cv_score,
        "model_complexity": complexity,
        "threshold_hint": threshold_hint,
        "subject_generalization": subject_report,
        "view_generalization": view_report,
        "split_seed": split_seed,
        "test_size": TEST_SIZE,
        "train_groups": sorted(train_df["group"].unique().tolist()),
        "test_groups": sorted(test_df["group"].unique().tolist()),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "metrics": metrics,
    }
    SPLIT_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[MODEL] Saved: {MODEL_PATH}")
    print(f"[MODEL] Manifest: {SPLIT_PATH}")
    print(f"[MODEL] Features: {columns}  Params: {params}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}")
        raise
