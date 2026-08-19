#!/usr/bin/env python3
"""
evaluate_model.py - 학습 결과를 검증한다

    python3 evaluate_model.py

manifest에 저장된 test 세션으로 재평가한다.
같은 데이터로 학습했으므로 재현 확인용이며 독립 검증이 아니다.

세 가지를 함께 확인한다.

    [RUNTIME] agreement    런타임 경로가 sklearn 원본과 100% 일치하는지.
                           LUT 컴파일이 정확도를 바꾸지 않았음을 매번 검증한다.
    [RUNTIME] latency      프레임당 실측 추론 지연
    [FALLBACK]             임계값 방식이 RF 대비 얼마나 차이 나는지

FALLBACK 점수가 RF보다 높으면 임계값 방식을 쓰는 것이 맞다.
더 단순하고, 더 정확하고, 모델 파일도 필요 없다.
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

# 프로젝트 루트를 import 경로에 넣는다.
# tools/는 패키지가 아니라 실행 스크립트 모음이므로, 어디서 실행하든
# posture 패키지를 찾을 수 있어야 한다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posture.contract import ALL_FEATURES, LABEL_MAP, POSTURE_LABELS

from posture.paths import DATASET_PATH, MODEL_PATH, SPLIT_PATH


def print_block(tag, y_true, y_pred):
    labels = [LABEL_MAP[label] for label in POSTURE_LABELS]
    balanced = balanced_accuracy_score(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=POSTURE_LABELS,
        output_dict=True, zero_division=0,
    )
    bad_row = matrix[LABEL_MAP["BAD"]]
    total = int(bad_row.sum())
    miss = (int(bad_row[LABEL_MAP["NORMAL"]]) / total) if total else 0.0

    print(f"[{tag}] Balanced Accuracy: {balanced:.4f}")
    for label in POSTURE_LABELS:
        print(
            f"[{tag}] {label:<7} Precision: {report[label]['precision']:.4f}  "
            f"Recall: {report[label]['recall']:.4f}  F1: {report[label]['f1-score']:.4f}"
        )
    print(f"[{tag}] BAD -> NORMAL Rate: {miss:.4f}")
    print(f"[{tag}] Confusion Matrix rows=true, cols=pred (NORMAL, WARNING, BAD)")
    print(matrix)
    return balanced


def main():
    for path in (DATASET_PATH, MODEL_PATH, SPLIT_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}")

    manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    features = manifest["feature_columns"]
    test_groups = set(manifest["test_groups"])

    df = pd.read_csv(DATASET_PATH)
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    df = df[df["label"].isin(POSTURE_LABELS)].copy()

    # manifest가 실제로 쓰는 특징만 필수로 본다.
    # ALL_FEATURES 전체를 요구하면 3D가 없는 데이터에서 전 행이 삭제된다.
    for column in ALL_FEATURES:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=list(features) + ["subject", "session", "label"])

    df["target"] = df["label"].map(LABEL_MAP).astype(int)
    df["group"] = (
        df["subject"].astype(str).str.strip() + "::" + df["session"].astype(str).str.strip()
    )

    test_df = df[df["group"].isin(test_groups)].copy()
    if test_df.empty:
        raise ValueError("No rows matched the saved test sessions.")
    if set(test_df["target"]) != set(LABEL_MAP.values()):
        raise ValueError("Saved test split does not contain all three classes.")

    x_test = test_df[features].to_numpy(dtype=np.float32)
    y_test = test_df["target"].to_numpy(dtype=np.int64)

    model = joblib.load(MODEL_PATH)
    if int(getattr(model, "n_features_in_", -1)) != len(features):
        raise ValueError("Model feature count does not match the saved contract.")

    print(f"[EVAL] Test rows: {len(test_df)}  Test sessions: {test_df['group'].nunique()}")
    print(f"[EVAL] Features: {features}  Params: {manifest.get('model_params')}")
    print_block("EVAL", y_test, model.predict(x_test))

    # ── 런타임 경로 대조 ────────────────────────────────────
    from posture.judge import RandomForestPostureClassifier, ThresholdClassifier

    classifier = RandomForestPostureClassifier()
    available = [c for c in ALL_FEATURES if c in test_df.columns]
    rows = test_df[available].to_dict("records")

    runtime_pred = np.array(
        [int(np.argmax(classifier.predict(row))) for row in rows], dtype=np.int64
    )
    reference_pred = model.predict(x_test)
    agree = float(np.mean(runtime_pred == reference_pred))
    mode = "LUT" if classifier.compiled else "tree walk"
    print(f"[RUNTIME] mode={mode}  agreement with sklearn predict: {agree * 100:.4f}%")
    if agree < 1.0:
        print("[ERROR] 런타임 경로가 원본 모델과 다른 예측을 냅니다. LUT 컴파일을 확인하세요.")

    start = time.perf_counter()
    for row in rows:
        classifier.predict(row)
    latency = (time.perf_counter() - start) / len(rows) * 1000.0
    print(f"[RUNTIME] inference latency: {latency:.4f} ms/frame ({mode})")

    # ── threshold 폴백 비교 ────────────────────────────────
    threshold = ThresholdClassifier()
    threshold_pred = np.array(
        [int(np.argmax(threshold.predict(row))) for row in rows], dtype=np.int64
    )
    print(f"[FALLBACK] {threshold.describe()}")
    print_block("FALLBACK", y_test, threshold_pred)

    print("[EVAL] NO_POSE는 RandomForest 추론 이전에 처리됩니다.")
    print("[EVAL] 같은 데이터로 학습했으므로 재현 확인용이며 독립 검증이 아닙니다.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}")
        raise
