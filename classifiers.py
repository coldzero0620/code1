#!/usr/bin/env python3
"""
judge/classifiers.py - 특징 행 → 클래스 확률

두 가지를 제공한다.

    RandomForestPostureClassifier   학습된 모델. 정상 경로
    ThresholdClassifier             단일 축 임계값. 모델이 없을 때의 폴백

둘 다 predict(row) -> 확률벡터 인터페이스를 가지므로
PostureJudge는 어느 쪽인지 몰라도 된다.

특징 순서는 반드시 manifest에서 읽는다. 이 규칙이 깨지면
학습과 추론이 다른 값을 같은 자리에 넣게 되고, 오류 없이 조용히 틀린다.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..contract import MODEL_COMPLEXITY, POSTURE_LABELS
from ..features import to_vector
from ..paths import MODEL_PATH, SPLIT_PATH

# manifest가 없을 때만 쓰는 최후의 기본값.
# 정상 경로에서는 tools/train_model.py가 격자 탐색으로 구한 값이 manifest에 들어간다.
FALLBACK_THRESHOLDS = {
    "warning_enter": 0.15,
    "warning_exit": 0.10,
    "bad_enter": 0.32,
    "bad_exit": 0.25,
}


def load_manifest() -> Optional[dict]:
    """학습 결과 manifest. 없거나 깨졌으면 None."""
    if not SPLIT_PATH.exists():
        return None
    try:
        return json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


class ThresholdClassifier:
    """
    단일 축 임계값 기반. 자체 히스테리시스를 가진다.

    축과 부호는 기본적으로 manifest에서 읽는다.
    posture_error는 나쁠수록 커지지만 cva_error/fwd_error는 나쁠수록
    음수로 작아진다. 축을 고정하면 3D 모델에서 전부 NORMAL이 나온다.

    인자를 주면 manifest보다 우선한다. axis/sign까지 넘길 수 있게 해둔
    이유는, 이 판정기가 manifest 유무에 따라 조용히 다른 축으로 동작하기
    때문이다. 시험 코드처럼 결정적 동작이 필요한 곳에서는
    use_manifest=False로 완전히 고정할 수 있어야 한다.
    """

    name = "threshold"

    def __init__(
        self,
        warning_enter=None,
        warning_exit=None,
        bad_enter=None,
        bad_exit=None,
        axis=None,
        sign=None,
        use_manifest=True,
    ):
        hint = FALLBACK_THRESHOLDS.copy()
        self.axis = "posture_error"
        self.sign = 1.0
        manifest = load_manifest() if use_manifest else None
        if manifest and isinstance(manifest.get("threshold_hint"), dict):
            raw = manifest["threshold_hint"]
            for key in hint:
                value = raw.get(key)
                if isinstance(value, (int, float)):
                    hint[key] = float(value)
            if isinstance(raw.get("axis"), str):
                self.axis = raw["axis"]
            if isinstance(raw.get("sign"), (int, float)):
                self.sign = float(raw["sign"])
            self.source = "manifest"
        else:
            self.source = "fallback"

        # 명시 인자가 manifest를 덮어쓴다
        if axis is not None:
            self.axis = axis
            self.source = "명시"
        if sign is not None:
            self.sign = float(sign)

        self.warning_enter = hint["warning_enter"] if warning_enter is None else warning_enter
        self.warning_exit = hint["warning_exit"] if warning_exit is None else warning_exit
        self.bad_enter = hint["bad_enter"] if bad_enter is None else bad_enter
        self.bad_exit = hint["bad_exit"] if bad_exit is None else bad_exit
        self.feature_columns = [self.axis]
        self._last = "NORMAL"

        if not (self.warning_exit <= self.warning_enter <= self.bad_exit <= self.bad_enter):
            print(
                "[WARNING] threshold 경계 순서가 뒤집혔습니다 "
                f"(W {self.warning_exit:+.3f}/{self.warning_enter:+.3f}, "
                f"B {self.bad_exit:+.3f}/{self.bad_enter:+.3f}). "
                "manifest를 확인하세요. 판정 결과를 신뢰할 수 없습니다."
            )

    def reset(self) -> None:
        self._last = "NORMAL"

    def predict(self, row: Dict[str, float]) -> np.ndarray:
        error = row[self.axis] * self.sign

        if self._last == "BAD":
            if error < self.bad_exit:
                out = "WARNING" if error >= self.warning_exit else "NORMAL"
            else:
                out = "BAD"
        elif self._last == "WARNING":
            if error >= self.bad_enter:
                out = "BAD"
            elif error < self.warning_exit:
                out = "NORMAL"
            else:
                out = "WARNING"
        else:
            if error >= self.bad_enter:
                out = "BAD"
            elif error >= self.warning_enter:
                out = "WARNING"
            else:
                out = "NORMAL"

        self._last = out
        proba = np.zeros(len(POSTURE_LABELS), dtype=np.float32)
        proba[POSTURE_LABELS.index(out)] = 1.0
        return proba

    def describe(self) -> str:
        return (
            f"threshold({self.source}) axis={self.axis} sign={self.sign:+.0f} "
            f"W>={self.warning_enter:+.3f}/<{self.warning_exit:+.3f} "
            f"B>={self.bad_enter:+.3f}/<{self.bad_exit:+.3f}"
        )


class RandomForestPostureClassifier:
    """
    joblib 모델 로드. 특징 순서는 manifest에서 가져온다.

    특징이 1개면 트리 앙상블을 구간 테이블로 컴파일한다.
    1차원 입력에서 RF는 계단 함수이므로, 모든 트리의 분할 임계값을 모아
    경계를 만들고 각 구간의 대표점 확률을 미리 계산해두면
    런타임 추론이 np.searchsorted 한 번으로 끝난다. 출력은 완전히 동일하다.
    """

    name = "rf"

    def __init__(self, model_path: Path = MODEL_PATH, split_path: Path = SPLIT_PATH):
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("joblib이 없습니다. pip install joblib") from exc

        if not model_path.exists():
            raise FileNotFoundError(f"모델이 없습니다: {model_path}")
        if not split_path.exists():
            raise FileNotFoundError(f"manifest가 없습니다: {split_path}")

        manifest = json.loads(split_path.read_text(encoding="utf-8"))
        self.feature_columns: List[str] = manifest["feature_columns"]
        self.model_classes: List[str] = manifest["model_classes"]
        self.metrics = manifest.get("metrics", {})

        if self.model_classes != POSTURE_LABELS:
            raise ValueError(
                f"클래스 불일치 manifest={self.model_classes} contract={POSTURE_LABELS}"
            )

        self.model = joblib.load(model_path)
        n_in = int(getattr(self.model, "n_features_in_", -1))
        if n_in != len(self.feature_columns):
            raise ValueError(f"모델 입력 차원 {n_in} != 특징 수 {len(self.feature_columns)}")

        # 1행 예측에서 스레드 풀은 순수 오버헤드다.
        try:
            self.model.n_jobs = 1
        except AttributeError:
            pass

        # 모델의 클래스 인덱스 → POSTURE_LABELS 순서로 재배열하는 순열
        model_order = [int(c) for c in self.model.classes_]
        self._reorder = np.array(
            [model_order.index(index) for index in range(len(POSTURE_LABELS))]
        )

        trained_complexity = manifest.get("model_complexity")
        if trained_complexity is not None and trained_complexity != MODEL_COMPLEXITY:
            print(
                f"[WARNING] 학습 데이터 model_complexity={trained_complexity}, "
                f"런타임={MODEL_COMPLEXITY}. 분포가 어긋납니다."
            )

        self._boundaries: Optional[np.ndarray] = None
        self._table: Optional[np.ndarray] = None
        if len(self.feature_columns) == 1:
            self._compile_lookup_table()

    def _compile_lookup_table(self) -> None:
        """
        주의: sklearn 트리는 입력을 float32로 캐스팅한 뒤 float64 임계값과
        `x <= threshold`로 비교한다. 따라서
          - 경계는 반드시 float64 원본을 유지해야 하고
          - 조회 값은 반드시 float32로 캐스팅한 뒤 비교해야 한다.
        경계를 float32로 저장하면 변환 오차(~1e-8)만큼 구간이 어긋나
        경계에 정확히 걸린 입력에서 다른 답이 나온다.
        """
        thresholds = []
        for estimator in self.model.estimators_:
            tree = estimator.tree_
            mask = tree.feature >= 0  # -2 = leaf
            thresholds.append(tree.threshold[mask])

        if not thresholds:
            return
        boundaries = np.unique(np.concatenate(thresholds)).astype(np.float64)
        if boundaries.size == 0 or boundaries.size > 20000:
            return  # 컴파일 이득이 없거나 메모리가 과한 경우 일반 경로 사용

        # 각 구간의 대표점. 구간 i는 boundaries[i-1] < x <= boundaries[i].
        points = np.empty(boundaries.size + 1, dtype=np.float64)
        points[0] = boundaries[0] - 1.0
        points[1:-1] = (boundaries[:-1] + boundaries[1:]) / 2.0
        points[-1] = boundaries[-1] + 1.0

        # 대표점이 float32 캐스팅 후에도 제 구간에 남는지 확인한다.
        # 인접한 경계가 float32 해상도보다 촘촘하면 어긋날 수 있다.
        landed = np.searchsorted(
            boundaries, points.astype(np.float32).astype(np.float64), side="left"
        )
        if not np.array_equal(landed, np.arange(points.size)):
            return  # 안전하게 일반 경로 사용

        table = self.model.predict_proba(points.reshape(-1, 1).astype(np.float32))
        table = table[:, self._reorder].astype(np.float32)

        # 자체 검증: 모든 경계에서 정확히, 그리고 바로 앞뒤에서 원본과 대조한다.
        probe = np.concatenate([
            boundaries,
            np.nextafter(boundaries.astype(np.float32), np.float32(-np.inf)).astype(np.float64),
            np.nextafter(boundaries.astype(np.float32), np.float32(np.inf)).astype(np.float64),
            points,
        ])
        # 원본 모델의 예측을 POSTURE_LABELS 인덱스로 변환해 LUT 결과와 대조한다.
        model_order = [int(c) for c in self.model.classes_]
        class_to_label = {value: index for index, value in enumerate(model_order)}
        reference = np.array([
            class_to_label[int(value)]
            for value in self.model.predict(probe.reshape(-1, 1).astype(np.float32))
        ])

        index = np.searchsorted(
            boundaries, probe.astype(np.float32).astype(np.float64), side="left"
        )
        compiled = np.argmax(table[index][:, self._reorder.argsort()], axis=1)

        if not np.array_equal(compiled, reference):
            print("[WARNING] LUT 자체 검증 실패 → 일반 트리 순회 경로를 사용합니다.")
            return

        self._boundaries = boundaries
        self._table = table

    @property
    def compiled(self) -> bool:
        return self._table is not None

    def reset(self) -> None:
        pass

    def predict(self, row: Dict[str, float]) -> np.ndarray:
        if self._table is not None:
            # sklearn과 동일하게 float32로 캐스팅한 뒤 경계와 비교한다.
            value = float(np.float32(row[self.feature_columns[0]]))
            index = int(np.searchsorted(self._boundaries, value, side="left"))
            return self._table[index]

        x = to_vector(row, self.feature_columns)
        return self.model.predict_proba(x)[0][self._reorder].astype(np.float32)

    def describe(self) -> str:
        ba = self.metrics.get("balanced_accuracy")
        extra = f" BA={ba:.3f}" if isinstance(ba, float) else ""
        mode = f"LUT({self._boundaries.size + 1} bins)" if self.compiled else "tree walk"
        return f"RandomForest{extra} features={self.feature_columns} {mode}"
