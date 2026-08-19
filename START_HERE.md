# 시작하기

거북목(전방두부자세) 실시간 감지 시스템.
라즈베리파이 + CSI 카메라 + RGB LED + BLE 진동 밴드로 동작한다.

## 무엇부터 보면 되나

| 하려는 일 | 문서 |
|---|---|
| 노트북에 개발 환경 만들기 | `docs/01_WINDOWS_SETUP.md` |
| 학습용 영상 찍기 | `docs/02_SHOOTING_GUIDE.md` |
| 데이터 만들고 학습하기 | `docs/03_PIPELINE_README.md` |
| 코드 구조 이해하기 | `docs/04_ARCHITECTURE.md` |

## 전체 흐름

```
영상 촬영
   │
   ├─ tools/ingest_video.py      영상 → data/posture_dataset.csv
   │  (또는 tools/collect_data.py 로 라파에서 직접 수집)
   │
   ├─ tools/check_dataset.py     학습 가능한지 먼저 확인
   │
   ├─ tools/train_model.py       models/ 에 모델과 manifest 생성
   │
   └─ models/ 를 라즈베리파이로 복사
          │
          └─ python3 run_monitor.py
```

## 빠른 확인

하드웨어 없이 코드가 제대로 조립됐는지 보려면:

```bash
python3 test_integration.py
```

가짜 카메라와 가짜 BLE로 판정부터 출력까지 한 바퀴 돌린다.

## 라즈베리파이에서 실행

```bash
python3 run_monitor.py              # 학습된 모델 사용
python3 run_monitor.py --headless   # 화면 없이
python3 run_monitor.py --model threshold   # 임계값 폴백 강제
```

모델 파일이 없으면 자동으로 임계값 방식으로 내려간다.
시연 중 모델 문제로 전체가 죽지 않도록 한 장치다.

## 조작

| 입력 | 동작 |
|---|---|
| GPIO23 스위치 LOW | 자세 기능 일시정지 (OS/VNC는 유지) |
| GPIO23 스위치 HIGH | 강제 재개. UI 정지까지 함께 해제 |
| SPACE | 일시정지 / 재개 |
| ESC | 요약 화면 / 라이브 복귀 |
| 창 닫기 | 화면만 끔. 감지·LED·진동은 계속 |

## LED 색

| 색 | 의미 |
|---|---|
| 초록 | 정상 |
| 파랑 | 경고, 또는 캘리브레이션 중 |
| 빨강 | 불량 |
| 흰색 | 사람이 안 잡힘 |
| 흰색 점멸 | 밴드 연결 끊김 |
| 꺼짐 | 일시정지 |

## 라파에서 먼저 확인할 것

**1. 카메라가 전체 화각 모드를 지원하는가**

```bash
rpicam-vid --list-cameras
```

출력에 `1640x1232`가 있어야 한다. 없으면 IMX219가 아닌 것이므로
`posture/contract.py`의 `CAMERA_SENSOR_MODE`를 빈 문자열로 두어야 한다.
단, 그러면 수집과 실기의 화각을 따로 맞춰야 한다.

**2. 실제 처리 속도가 몇 fps인가**

```bash
python3 tools/collect_data.py --subject test --session a --label NORMAL --seconds 20
```

출력에 `[DATASET] 실측 처리 속도`가 찍힌다.
20fps에 못 미치면 `contract.CAMERA_FPS`와 `VIDEO_PROCESS_FPS`를
그 값으로 낮춰야 학습 데이터와 실기의 프레임 간격이 맞는다.

## 설정을 바꾸려면

`posture/contract.py` 한 곳만 고치면 된다.
학습 도구와 런타임이 같은 파일을 읽으므로 한쪽만 바뀌는 일이 없다.

CSV 스키마를 바꿨다면 `SCHEMA_VERSION`을 올린다.
그러면 예전 포맷이 섞였을 때 학습이 시작 전에 멈춘다.
