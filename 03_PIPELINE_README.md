# 자세 판정 학습 파이프라인

거북목 자세를 실시간 판정하는 시스템의 머신러닝 부분.
데이터 수집 → 학습 → 검증 → 라즈베리파이 추론까지 6개 파일로 완결된다.

---

## 빠른 시작 — 영상으로 학습하기

```bash
# 1. videos/ 폴더에 영상을 넣는다 (파일명 규칙은 아래)
# 2. 특징 추출
python3 ingest_video.py

# 3. 학습 가능한지 확인 (몇 초)
python3 check_dataset.py

# 4. 학습 (파라미터는 자동 선택)
python3 train_model.py

# 5. 검증
python3 evaluate_model.py
```

파이 카메라로 직접 수집하려면 1~2단계 대신:

```bash
python3 collect_data.py --subject s01 --session a --label NORMAL --view side --seconds 20
```

---

## 디렉토리 배치

```
project/
├── models/                    ← train_model.py가 자동 생성
│   ├── posture-rf.joblib
│   └── split_manifest.json
├── videos/                    ← 영상을 여기에 넣는다
│   ├── s01_BASELINE.mp4
│   ├── s01_side_NORMAL_a.mp4
│   └── segments.csv           (선택) 구간이 섞인 영상만
└── scripts/                   ← 파일 9개 + requirements.txt
    ├── contract.py            공유 상수 (의존성 없음)
    ├── features.py            프레임 → 특징
    ├── ingest_video.py        영상 → CSV
    ├── collect_data.py        파이 카메라 → CSV
    ├── measure_range.py       자세 범위 측정 (촬영 현장용)
    ├── check_dataset.py       학습 가능 여부 점검
    ├── train_model.py         학습 + 자동 튜닝
    ├── evaluate_model.py      검증
    ├── posture_runtime.py     실시간 판정 + 밴드 연결
    ├── requirements.txt
    └── posture_dataset.csv    자동 생성
```

`train_model.py`가 `parents[1] / "models"`를 쓰므로 **스크립트는 반드시 하위 폴더**에 둔다.
루트에 두면 `models/`가 프로젝트 밖으로 나간다.

---

## 판정 상태

| 상태 | 의미 |
|---|---|
| `NO_POSE` | 사람이 없거나 귀·어깨가 안 보임 |
| `NORMAL` | 바른 자세 |
| `WARNING` | 주의 |
| `BAD` | 거북목 |

`NO_POSE`는 성격이 다르다. 모델이 판단하는 것이 아니라 **입력이 없는 상태**다.
그래서 추론 이전에 걸러내고, 학습 데이터에서도 제외한다.

---

## 1-A. 영상으로 데이터 만들기

### 파일명 규칙

```
{subject}_{view}_{label}_{session}.mp4      s01_side_BAD_a.mp4
{subject}_BASELINE.mp4                      모든 시점 공통 기준점
{subject}_{view}_BASELINE.mp4               해당 시점 전용 기준점
```

`view`는 `side` `oblique` `low` `high` 중 하나.
규칙에 안 맞는 파일은 경고를 내고 건너뛴다.

### 여러 자세가 섞인 영상

`videos/segments.csv`에 구간을 적는다.

```csv
file,subject,view,session,start,end,label
s01_side_MIXED_a.mp4,s01,side,a,0:00,0:30,BASELINE
s01_side_MIXED_a.mp4,s01,side,a,0:40,1:20,NORMAL
s01_side_MIXED_a.mp4,s01,side,a,1:30,2:10,BAD
```

여기 등록된 파일은 파일명 스캔에서 제외되므로 충돌하지 않는다.
구간 사이(0:30~0:40)는 적지 않으면 버려진다. 자세 전환 프레임을 빼기 위한 것이다.

### 기준점(BASELINE)에 관한 제약

3D 특징은 사람당 기준점 하나면 충분하다. 신체 좌표계라 각도와 무관하다.
2D 특징은 시점마다 기준점이 다르다. 정측면 기준점을 비스듬한 영상에 쓰면 틀린다.

`{subject}_BASELINE.mp4` 하나만 찍으면 2D 특징이 비어 학습에서 자동 제외된다.
**문제되지 않는다.** 각도 다양성이 목표라면 3D만 쓰는 것이 오히려 맞다.

### 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--videos` | `../videos` | 영상 폴더 |
| `--fps` | 5.0 | 초당 몇 프레임을 **CSV에 저장**할지 |
| `--process-fps` | 20.0 | 초당 몇 프레임을 **MediaPipe에 통과**시킬지 |
| `--segments` | `segments.csv` | 구간 표 파일명 |

이 둘은 다른 개념이다.

`--fps`는 저장 간격이다. 30fps 원본에서 인접 프레임은 거의 동일하므로
전부 저장할 이유가 없다. 낮추면 중복이 준다.

`--process-fps`는 MediaPipe 호출 간격이며 **런타임과 같아야 한다.**
MediaPipe는 `smooth_landmarks=True`에서 직전 호출의 추적 상태를 이어 쓰는데,
Python solutions API는 호출마다 내부 타임스탬프를 33333us씩 고정 증가시킨다.
실제 경과 시간은 보지 않는다. 따라서 평활 결과를 좌우하는 것은
"호출 사이에 몸이 얼마나 움직였는가"이고, 그것은 호출 간격이 결정한다.

    런타임 20fps        호출 간 50ms
    영상을 5fps로 처리   호출 간 200ms  → 움직임 4배, 다른 값이 나온다
    영상 전 프레임 처리  호출 간 33ms   → 이번엔 반대로 촘촘하다

그래서 전부 처리하는 것이 아니라 `CAMERA_FPS`에 맞춘다.
30fps 원본에서 20fps를 뽑으면 3프레임 중 2개를 쓰게 된다.

실기에서 잰 런타임 처리 속도가 20fps와 다르면 그 값으로 바꾼다.
`collect_data.py`가 실측치를 출력한다.

### 재처리는 안전하다

`source` 열에 영상 파일명이 기록된다.
같은 영상을 다시 처리하면 기존 행을 지우고 새로 쓴다. 중복이 쌓이지 않는다.

---

## 1-A-2. 자세 범위 측정 (촬영 현장용)

WARNING을 어느 정도로 잡을지는 사람마다 다르다. 측정해서 정한다.

본 촬영 전에 아래 영상을 하나 찍는다.

```
바른 자세 5초 유지
  → 천천히 거북목 끝까지 (5초에 걸쳐)
  → 끝에서 5초 유지
  → 천천히 원위치
```

```bash
python3 measure_range.py ../videos/s01_range.mp4
```

```
cva_deg 기준 자세 범위
  NORMAL      -9.52도
  WARNING    -30.30도   ← 이 자세를 취하게 한다
  BAD        -51.07도
```

**WARNING = 바른 자세에서 거북목 끝까지 가는 거리의 딱 절반.**
각도를 눈대중으로 맞추긴 어렵지만 "끝까지의 절반"은 몸으로 감이 온다.

35~65% 사이 어디든 분리가 뚜렷하므로 정확히 50%를 맞추려 애쓸 필요는 없다.
중요한 것은 **위치가 아니라 일관성**이다.

### 촬영 순서

`NORMAL → BAD → WARNING` 순서를 권한다.
`NORMAL → WARNING → BAD` 순으로 하면 점점 더 숙이면서
WARNING이 BAD 쪽으로 밀린다. BAD를 먼저 해봐야 절반의 감이 잡힌다.

**목만 움직이고 어깨와 골반 방향은 고정할 것.**
자세를 바꿀 때 몸통이 함께 돌아가면 `obliquity`가 자세와 상관관계를 갖게 되고,
모델이 그것을 지름길로 삼는다. 다른 사람에게는 통하지 않는다.

---

## 1-B. 학습 가능 여부 점검

```bash
python3 check_dataset.py
```

몇 초 안에 끝난다. 학습을 몇 분 돌려보고 나서야 데이터가 부족한 것을
알게 되는 상황을 막는다.

```
판정
----------------------------------------
  ★ 학습 불가

  문제: 세션이 2개 미만인 자세: ['WARNING']
  해결: 같은 자세를 다른 세팅에서 한 번 더 촬영하세요.
```

학습 불가면 종료 코드 1을 낸다. 스크립트로 묶을 때 쓸 수 있다.

---

## 1-C. 파이 카메라로 직접 수집

메인 서비스가 카메라를 잡고 있으면 실패한다. 먼저 정지시킨다.

```bash
sudo systemctl stop posture
```

### 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--subject` | 필수 | 사람 ID (`s01` 등 익명 ID 권장) |
| `--session` | 필수 | 세션 ID. 다시 세팅했을 때 바꾼다 |
| `--label` | 필수 | `NORMAL` / `WARNING` / `BAD` / `NO_POSE` |
| `--view` | `side` | 촬영 시점. `side` `oblique` `low` `high` |
| `--seconds` | 20 | **실제 저장 시간.** settle은 여기 포함 안 됨 |
| `--calibration` | 3.0 | 기준점 측정 시간 (바른 자세 유지) |
| `--prepare` | 3.0 | 자세 전환 대기 |
| `--settle` | 2.0 | 수집 시작 후 버릴 시간 |
| `--headless` | off | 디스플레이 없이 실행 (SSH 전용 접속) |

### 세션이란

**카메라를 안 옮기고 자세만 바꿔 이어 찍으면 세션 1개다.**
학습이 세션 단위로 train/test를 나누므로, 이러면 정답을 보고 시험 치는 셈이 된다.

세션을 바꿀 때는 일어났다가 카메라를 조금 옮기고 다시 앉는다.

### 권장 수집량

| | 최소 | 충분 |
|---|---|---|
| 사람 | 3명 | 4~5명 |
| 시점 | side, oblique | + low, high |
| 시점당 세션 | 2회 | 2~3회 |

**시점을 늘리는 것보다 사람을 늘리는 것이 우선이다.**
`Mean cross-subject`가 실제 성능을 나타내는 숫자이고, 3명이 최소선이다.

### 프레이밍 — 골반이 보여야 한다

3D 각도 불변 특징이 골반-어깨로 몸의 수직축을 세운다.
상반신만 찍으면 이 특징이 전부 실패하고, 각도 다양성이라는 목표가 무너진다.

수집 후 출력을 확인한다.

```
[DATASET] view=side  정측면 판정 비율: 98.2%  3D 유효: 96.7%
```

3D 유효 비율이 90% 아래면 프레이밍 문제다. 프레임을 넓히고 다시 찍는다.

---

## 2. 학습

```bash
python3 train_model.py
```

**사람이 정할 숫자가 없다.** 특징 조합과 하이퍼파라미터를 교차검증으로 자동 선택한다.

### 출력에서 봐야 할 것

```
[SEARCH] Selected features : ['fwd_error', 'cva_error']
[SEARCH] Selected params   : {'n_estimators': 60, 'max_depth': 4, ...}
[EVAL]   BAD -> NORMAL Rate: 0.0000
[SUBJECT] Mean cross-subject Balanced Accuracy: 0.9562
[VIEW]    Mean cross-view Balanced Accuracy: 0.9929
```

| 항목 | 의미 |
|---|---|
| `Selected features` | 자동 선택된 특징 조합 |
| `BAD -> NORMAL Rate` | 가장 위험한 오분류. 낮을수록 좋다 |
| `Mean cross-subject` | **처음 보는 사람에 대한 성능. 이 숫자가 진짜다** |
| `Mean cross-view` | 처음 보는 카메라 각도에 대한 성능 |

session 분할 점수보다 낮게 나오는 것이 정상이며, 낮다고 실패한 것이 아니다.

피험자가 1명이거나 시점이 1종이면 해당 평가를 건너뛰고 경고를 출력한다.

### 선택 기준

```
score = 균형정확도 - 0.5 x (BAD를 NORMAL로 놓친 비율)
```

거북목을 놓치는 비용이 헛경고보다 크기 때문이다.
성능이 비슷하면(0.005 이내) **더 가벼운 모델**을 고른다.
Pi 4에서는 트리 개수가 곧 프레임 지연이다.

### 특징 조합

| 계열 | 특징 | 성격 |
|---|---|---|
| 2D 투영 | `posture_error`, `abs_delta` | 가볍고 정측면에서 정확. 각도 변하면 무너짐 |
| 3D 신체 | `fwd_error`, `cva_error` | 카메라 위치 무관. z 추정 노이즈에 취약 |
| 시점 | `obliquity` | 0=정측면, 커질수록 정면 |

3D 특징을 쓰려면 `MODEL_COMPLEXITY=1` 이상이 필요하다 (contract.py).
`complexity=0`은 z 추정이 부정확해 3D 특징이 제 성능을 못 낸다.

**Pi 4에서 프레임 예산을 반드시 직접 측정할 것.**

---

## 3. 검증

```bash
python3 evaluate_model.py
```

```
[EVAL]    Balanced Accuracy: 0.9963
[RUNTIME] agreement with sklearn predict: 100.0000%
[RUNTIME] inference latency: 2.68 ms/frame (tree walk)
[FALLBACK] Balanced Accuracy: 0.9691
```

manifest의 test 세션으로 재평가한다. 같은 데이터이므로 **재현 확인용**이며
독립 검증이 아니다. 새 세션을 따로 모아 돌리면 훨씬 의미 있다.

**FALLBACK이 RF보다 높으면 임계값 방식을 쓰는 것이 맞다.**
더 단순하고, 더 정확하고, 모델 파일도 필요 없다.
실패가 아니라 실험으로 얻은 근거 있는 판단이다.

---

## 4. 런타임 연결

```python
import time
from features import create_pose, process_pose, extract_feature, BaselineCalibrator
from posture_runtime import build_judge, BandLink

pose  = create_pose()
judge = build_judge("rf")            # 또는 "threshold"
link  = BandLink(send_fn=lambda c: client.write_gatt_char(CMD_UUID, c.encode()))
calib = BaselineCalibrator(seconds=3.0)

# 캘리브레이션 — 루프 안에서 반드시 프레임을 새로 읽어야 한다
calib.start(time.monotonic())
while True:
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("camera stopped")
    if not calib.feed(extract_feature(process_pose(pose, frame)), time.monotonic()):
        break

# 판정 루프
ret, frame = cap.read()
feature = extract_feature(process_pose(pose, frame))
status, info = judge.decide(feature, calib.baseline, now=time.monotonic())
link.update(status)
```

BLE 연결이 될 때마다 `link.on_connected()`를 부른다.
세션을 다시 시작할 때는 `judge.reset(now=time.monotonic())`을 부른다.

`build_judge("rf")`는 모델 로드에 실패하면 **자동으로 임계값 방식으로 폴백**한다.

---

## 설계 원칙

### 계약은 contract.py에만

`contract.py`는 cv2도 mediapipe도 sklearn도 import하지 않는다.
그래서 노트북과 파이가 같은 파일을 읽는다.

**특징을 추가할 때는 `ALL_FEATURES`와 `FEATURE_CANDIDATES`만 고치면 된다.**
나머지 스크립트는 자동으로 따라온다.

### 특징 계산은 features.py에만

`build_feature_row()`가 이름-값 딕셔너리를 만들고
`to_vector()`가 manifest 순서대로 뽑는다.
**순서 불일치가 구조적으로 불가능하다.**

### RF 출력은 반드시 안정화를 거친다

프레임마다 독립 예측이라 그대로 쓰면 떨린다. 2단으로 막는다.

| 단계 | 방식 |
|---|---|
| 1단 | 최근 7프레임 **확률 평균** 후 argmax |
| 2단 | 후보 유지 시간 — **악화 0.6초 / 완화 2.0초** |

경고를 놓치는 비용이 헛경고보다 크므로 비대칭이다.
`NO_POSE`는 안정화를 거치지 않는다.

### 1차원 모델은 구간 테이블로 컴파일된다

특징이 1개면 RandomForest는 수학적으로 계단 함수다.
모든 분기점을 모아 구간별 확률을 미리 계산해두면
추론이 `np.searchsorted` 한 번으로 끝난다 (30ms → 0.002ms).

**근사가 아니라 완전히 동일한 출력**이며,
`evaluate_model.py`가 매번 100% 일치를 확인한다.

`[JUDGE] ... LUT(265 bins)`가 뜨면 컴파일된 것이고,
`tree walk`면 특징이 2개 이상이라 일반 경로를 쓰는 것이다.

---

## 진동 밴드 연결

`BandLink`가 판정 상태를 밴드 명령으로 옮긴다.

| 상태 | 명령 |
|---|---|
| NORMAL | `N` |
| WARNING | `W` |
| BAD | `B` |
| NO_POSE | `P` |
| PAUSED | `S` |

**상태 문자열을 그대로 보내면 안 된다.** 펌웨어가 첫 글자만 읽으므로
`NO_POSE`가 `N`(=NORMAL)으로, `PAUSED`가 `P`(=NO_POSE)로 오해석된다.
`NORMAL`/`WARNING`/`BAD`는 우연히 맞아떨어져 테스트가 통과하므로 특히 위험하다.

유지 중에는 현재 명령을 재전송한다. `N`/`B`/`P`/`S`는 펌웨어에서 멱등이므로,
통신 이상이나 재연결로 밴드가 초기화돼도 2초 안에 자동 복구된다.
`W`만 예외로 heartbeat(`H`)를 보낸다. 재전송하면 경고 패턴이 재발동하기 때문이다.

캘리브레이션 중에는 `link.pause()`, 끝나면 `link.resume()`.

---

## 자주 만나는 문제

| 증상 | 원인 | 대처 |
|---|---|---|
| `기존 CSV 헤더가 다릅니다` | 예전 스키마 파일에 이어쓰기 시도 | 파일을 옮기고 새로 수집 |
| `schema_version 불일치` | v2 데이터가 섞임 | 해당 행 제거 또는 재수집 |
| `Need at least 2 independent sessions` | 세션이 부족 | 세션을 나눠 재수집 |
| 3D 유효 비율 낮음 | 골반이 프레임 밖 | 프레임을 넓혀 재촬영 |
| `RandomForest 로드 실패` | 모델/manifest 없음·손상 | 임계값 방식으로 자동 폴백됨 |
| 학습이 오래 걸림 | 데이터가 많음 | 6만 행이면 약 30분. 정상 |

---

## 스키마 버전

현재 `SCHEMA_VERSION = 3`.

v2에서 3D 특징(`fwd_ratio`, `cva_deg`), 시점 기술자(`obliquity`),
촬영 시점 라벨(`view`)이 추가됐다.
**v2 이전에 수집한 데이터는 재수집해야 한다.**

---

## 실제 영상에서 확인된 것

두 개의 실제 영상으로 검증한 결과다.

| 항목 | 결과 |
|---|---|
| 포즈 검출 | 100% |
| 3D 유효율 (골반 보이는 영상) | 100% |
| 3D 유효율 (상반신만 나온 영상) | 0% — 정직하게 거부 |
| NORMAL vs BAD 분리도 | d' = 4.8 ~ 10.7, 겹침 없음 |
| 학습 → 검증 전 구간 | 정상 동작 |

`posture_error`, `fwd_error`, `cva_error` 모두 실제 사람에게서 자세를 뚜렷하게
구분했다. 특징 설계가 작동한다는 증거다.

### 골반 visibility 검사가 왜 필요했나

MediaPipe는 화면에 보이지 않는 관절도 **추정해서 좌표를 내놓는다.**
좌표만 보고 판단하면 상상해낸 골반 위치로 몸통 축을 세우고
"3D 정상"이라고 보고하게 된다. 각도 불변성이 통째로 거짓이 된다.

`_body_frame`이 골반과 어깨의 visibility를 확인하는 이유다.
합성 데이터로는 절대 드러나지 않는 종류의 문제였다.

---

## 아직 검증 안 된 것

| 항목 | 상태 |
|---|---|
| Pi 4 실시간 성능 | **미측정.** complexity=1은 데스크톱에서 10.7ms/frame. Pi에서는 예산 초과 위험 |
| 실제 사람 데이터 성능 | **미측정.** 지금까지 수치는 전부 합성 데이터 기준 |
| 카메라 수집 경로 | 실기기에서 실행된 적 없음 |

### mediapipe 버전 고정 필요

```
mediapipe 0.10.33 → mp.solutions 제거됨. 전부 실패
mediapipe 0.10.14 → 정상
```

`pip install mediapipe`를 그냥 치면 최신 버전이 깔려 코드가 전부 깨진다.

```bash
pip install mediapipe==0.10.14
```

`complexity=0`은 첫 실행 시 모델을 인터넷에서 내려받는다.
`complexity=1`은 패키지에 포함돼 있어 오프라인에서 동작한다.
