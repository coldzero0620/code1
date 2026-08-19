# 구조

## 왜 이렇게 나눴는가

모듈화 전에는 두 덩어리가 따로 있었다.

- `python_code12_2.py` (2189줄) — 라즈베리파이 런타임
- `scripts/` — 학습 파이프라인

둘 다 특징을 계산하고, 둘 다 자세를 분류하고, 둘 다 카메라를 열었다.
그런데 계산식이 서로 달랐다. 같은 자세에서 다른 숫자가 나왔고,
학습한 모델을 런타임에 넣으면 **에러 없이 조용히 틀린 답**이 나왔다.

같은 일을 하는 코드가 두 벌 있으면 언젠가 갈라진다.
그래서 겹치는 것을 전부 한 벌로 줄이는 것이 이 구조의 목적이다.

## 세 계층

```
        ┌──────────────────────────────┐
        │            app               │  실행 계층
        │  상태 공유 · 캘리브레이션      │
        │  UI · 키 입력 · 메인 루프      │
        └───────┬──────────────┬───────┘
                │              │
        ┌───────▼──────┐  ┌────▼─────────┐
        │    judge     │  │   hardware   │
        │  판정 계층    │◄─┤  주변장치     │
        │              │  │ (명령 규약만) │
        └───────┬──────┘  └────┬─────────┘
                │              │
        ┌───────▼──────────────▼───────┐
        │   features  ·  contract      │  계약 계층
        └──────────────────────────────┘
```

의존은 위에서 아래로만 흐른다.

- `judge`는 카메라도 GPIO도 BLE도 import하지 않는다
- `hardware`는 `judge.band`(명령 규약) 하나만 가져온다. 판정 로직은 모른다
- `hardware`는 `RuntimeState`를 모른다. 상태 변화는 전부 콜백으로 위에 알린다

그래서 이런 것이 가능하다.

| 하고 싶은 것 | 필요한 것 |
|---|---|
| 판정 로직 시험 | 노트북. 카메라 불필요 |
| LED/밴드 배선 확인 | 라파. 모델 불필요 |
| 학습 | 노트북. 하드웨어 전부 불필요 |

## 파일

```
posture-project/
├── run_monitor.py              라즈베리파이 실행 진입점
├── test_integration.py         가짜 하드웨어로 전체 검증
│
├── posture/
│   ├── contract.py             모든 설정의 단일 출처
│   ├── paths.py                파일 위치의 단일 출처
│   ├── features.py             MediaPipe → 특징
│   │
│   ├── judge/                  ── 판정 계층 ──
│   │   ├── stabilizer.py       확률 평균 + 유지시간
│   │   ├── classifiers.py      RandomForest / 임계값 폴백
│   │   ├── judge.py            NO_POSE 선처리 + 지속시간 악화
│   │   └── band.py             상태 → 밴드 명령 규약
│   │
│   ├── hardware/               ── 주변장치 ──
│   │   ├── gpio.py             GPIO 추상화 (없으면 더미)
│   │   ├── camera.py           CSI 카메라 + 최신 프레임 워커
│   │   ├── led.py              RGB LED
│   │   ├── band.py             BLE 연결 · 배터리 수신
│   │   └── switch.py           GPIO23 슬라이드 스위치
│   │
│   └── app/                    ── 실행 계층 ──
│       ├── state.py            스레드 간 공유 상태
│       ├── stats.py            세션 통계 · 점수
│       ├── calibration.py      3초 기준자세 측정
│       ├── ui.py               창 · 오버레이 · 요약
│       └── monitor.py          메인 루프
│
├── tools/                      ── 학습 도구 ──
│   ├── collect_data.py         카메라로 직접 수집
│   ├── ingest_video.py         영상 파일에서 수집
│   ├── check_dataset.py        학습 전 검증
│   ├── train_model.py          특징 조합 탐색 + 학습
│   ├── evaluate_model.py       저장된 모델 재평가
│   └── measure_range.py        특징 분포 진단
│
├── data/    posture_dataset.csv
├── models/  posture-rf.joblib, split_manifest.json
├── videos/  학습용 영상
└── docs/
```

## 제거한 중복

| 기능 | 예전 | 지금 |
|---|---|---|
| 특징 추출 | 2벌 (계산식이 달랐음) | `features.extract_feature` |
| 카메라 | 2벌 (화각·품질이 달랐음) | `hardware/camera.py` |
| MediaPipe 생성 | 2벌 | `features.create_pose` |
| 기준자세 측정 | 2벌 (한쪽은 축 1개만) | `features.BaselineCalibrator` |
| 자세 분류 | 2벌 | `judge/classifiers.py` |
| 시간 안정화 | 2벌 | `judge/stabilizer.py` |
| 밴드 명령 문자 | 2벌 (한쪽은 코드에 직접 박음) | `contract.BAND_COMMANDS` |
| 설정 상수 | 2벌 | `contract.py` |
| 파일 경로 | 3~4곳 | `paths.py` |

## 학습 결과가 런타임에 전달되는 경로

```
tools/train_model.py
        │
        ├─→ models/posture-rf.joblib        학습된 모델
        └─→ models/split_manifest.json      쓴 특징과 그 순서, 임계값, 성능
                    │
                    ▼
        judge/classifiers.py
            manifest에서 feature_columns를 읽는다
                    │
                    ▼
        features.to_vector(row, feature_columns)
            manifest 순서대로 값을 배열한다
```

**특징 순서를 코드에 적지 않는 것이 핵심이다.**
예전 런타임은 "4개 입력이면 내가 아는 그 4개"라고 가정했다.
학습이 다른 4개를 고르면 값이 뒤섞인 채로 예측이 계속됐다.
지금은 순서가 항상 manifest에서 오므로 그런 어긋남이 구조적으로 불가능하다.

`SCHEMA_VERSION`이 맞지 않는 CSV가 섞이면 학습이 시작 전에 멈춘다.

## 판정이 만들어지는 순서

```
프레임
  │
  ├─ features.extract_feature       포즈가 없으면 None
  │       │
  │       └─ None → NO_POSE (모델 추론 없이 즉시)
  │
  ├─ features.build_feature_row     baseline을 빼서 오차로 바꾼다
  │
  ├─ classifier.predict             확률 벡터
  │
  ├─ StatusStabilizer               최근 7프레임 평균 + 유지시간
  │                                 악화 0.6초 / 완화 2.0초
  │
  └─ 지속시간 악화                   WARNING이 2초 이어지면 BAD
          │
          ▼
   NO_POSE / NORMAL / WARNING / BAD
```

안정화와 지속시간 악화는 서로 다른 축이다.

- 안정화: "분류기 판정이 흔들리지 않는가"
- 지속시간 악화: "같은 경고가 얼마나 오래 갔는가"

## 상태 → 출력

| 상태 | LED | 밴드 |
|---|---|---|
| NORMAL | 초록 | `N` |
| WARNING | 파랑 | `W` |
| BAD | 빨강 | `B` |
| NO_POSE | 흰색 | `P` |
| PAUSED | 꺼짐 | `S` |

밴드 연결이 끊기면 LED 워커가 흰색 0.5초 점멸로 덮어쓴다.
진동이 안 오는 상태를 사용자가 알아야 하기 때문이다.

펌웨어는 수신 문자열의 **첫 글자만** 읽는다.
상태 문자열을 그대로 보내면 `NO_POSE`가 `N`(=NORMAL)로 오해석된다.
반드시 `contract.BAND_COMMANDS`를 거쳐야 한다.

## 일시정지

두 겹이다.

| 출처 | 해제 방법 |
|---|---|
| `ui_paused` | SPACE 키, 요약 화면 |
| `switch_paused` | GPIO23 슬라이드 스위치 |

**물리 스위치가 마스터다.** HIGH로 올리면 UI 쪽 정지까지 함께 풀린다.
그러지 않으면 "스위치는 켰는데 왜 안 움직이지" 상황이 생긴다.

스위치는 라즈베리파이 전원을 건드리지 않는다.
어느 위치에서도 OS와 VNC는 계속 살아 있다.

## 수집과 실기가 같아야 하는 것

학습 데이터와 실기 입력이 다르면 모델 성능이 그대로 떨어진다.
아래는 전부 `contract.py`에서 오므로 한쪽만 바뀔 수 없다.

| 항목 | 값 |
|---|---|
| 카메라 출력 | 640×480 |
| MediaPipe 입력 | 320×240, INTER_AREA |
| 센서 모드 | `1640:1232:10:P` (IMX219 전체 화각) |
| MJPEG 품질 | 60 |
| 프레임 간격 | 20fps |
| `model_complexity` | 1 |
| `smooth_landmarks` | True |

**프레임 간격이 특히 중요하다.** MediaPipe Python API는 `process()`
호출마다 내부 타임스탬프를 33333μs씩 고정 증가시킨다(`solution_base.py`).
실제 경과 시간은 보지 않는다. 따라서 평활 강도를 정하는 것은
"호출 사이에 몸이 얼마나 움직였는가", 즉 호출 간격이다.

영상에서 학습 데이터를 만들 때 `--process-fps`를 `CAMERA_FPS`에 맞추는
이유가 이것이다. 전 프레임을 처리하면 오히려 반대로 어긋난다.

## 확인 방법

```bash
python3 test_integration.py
```

가짜 카메라·가짜 BLE·더미 GPIO로 다음을 검사한다.

- 상태 → LED 색 / 밴드 명령 대응
- 밴드 명령이 상태 문자열이 아닌 규약 문자인지
- NO_POSE에서 진동이 확실히 꺼지는지
- WARNING 지속 시 BAD 승격
- 일시정지와 마스터 재개
- 세션 통계와 점수 축 정규화
- 종료 시 밴드 정지 명령

모델이 있을 때와 없을 때(임계값 폴백) 양쪽에서 통과해야 한다.
