#!/usr/bin/env python3
"""
contract.py - 모든 스크립트가 공유하는 상수

cv2도 mediapipe도 sklearn도 import하지 않는다.
그래서 학습용 노트북과 라즈베리파이가 같은 파일을 읽을 수 있고,
양쪽 설정이 어긋나는 사고가 구조적으로 불가능해진다.

여기 값을 고치면 6개 스크립트 전부에 동시에 반영된다.
반대로 말하면, 여기 값을 고칠 때는 이미 수집한 데이터와
호환되는지 반드시 확인해야 한다.

스키마를 바꿨다면 SCHEMA_VERSION을 올린다.
그러면 예전 포맷 CSV가 섞였을 때 학습이 시작 전에 중단된다.
"""

# ─────────────────────────────────────────────────────────────
# 카메라 / MediaPipe 설정
# ─────────────────────────────────────────────────────────────
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
INFERENCE_WIDTH = 320
INFERENCE_HEIGHT = 240

# 수집과 런타임이 같은 프레임 간격을 봐야 한다.
# MediaPipe는 smooth_landmarks=True에서 직전 프레임 상태를 이어 쓰므로,
# 프레임 간격이 다르면 같은 자세에서도 다른 값이 나온다.
CAMERA_FPS = 20

# ─────────────────────────────────────────────────────────────
# IMX219(카메라 모듈 2) 센서 모드
#
# rpicam-vid에 --mode 없이 640x480을 요청하면 드라이버가 센터 크롭
# 센서 모드를 고를 수 있다. 그러면 화각이 좁아져서 골반이 프레임 밖으로
# 나가고, 3D 특징(fwd_ratio, cva_deg)이 통째로 NaN이 된다.
# obliquity도 어깨 간격 기반이라 크롭 여부에 따라 값이 달라진다.
#
# 1640x1232는 2x2 비닝 전체 화각 모드다. 출력은 그대로 640x480이지만
# 화각이 센서 전체가 된다. 수집과 런타임이 반드시 같은 모드여야 한다.
#
# 다른 카메라(IMX708 등)를 쓰면 이 모드가 없어 rpicam-vid가 실패한다.
# 그럴 때는 빈 문자열로 두면 드라이버 자동 선택으로 돌아간다.
# 단, 그 경우 수집과 런타임 양쪽 다 빈 문자열이어야 한다.
CAMERA_SENSOR_MODE = "1640:1232:10:P"

# MJPEG 인코딩 품질. 압축 아티팩트가 랜드마크 좌표에 영향을 주므로
# 수집과 런타임이 같은 값을 써야 한다.
MJPEG_QUALITY = 60

VISIBILITY_THRESHOLD = 0.50

# 각도 불변 특징(3D 신체 좌표계)을 쓰려면 z 추정 정확도가 중요하다.
#   complexity=0  z가 가장 부정확. 2D 특징 전용.
#   complexity=1  3D 특징 사용 가능. Pi 4에서 프레임 예산을 반드시 확인할 것.
#   complexity=2  가장 정확하나 Pi 4에서는 대체로 과하다.
# 수집과 런타임이 반드시 같은 값이어야 한다.
MODEL_COMPLEXITY = 1
SMOOTH_LANDMARKS = True
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# ─────────────────────────────────────────────────────────────
# 라벨 / 상태
# ─────────────────────────────────────────────────────────────
RUNTIME_STATUSES = ["NO_POSE", "NORMAL", "WARNING", "BAD"]
POSTURE_LABELS = ["NORMAL", "WARNING", "BAD"]
LABEL_MAP = {"NORMAL": 0, "WARNING": 1, "BAD": 2}
SEVERITY = {"NORMAL": 0, "WARNING": 1, "BAD": 2}

# ─────────────────────────────────────────────────────────────
# 특징
# ─────────────────────────────────────────────────────────────
# build_feature_row()가 만들어내는 전체 특징 이름.
# 실제 학습에 쓸 조합은 train_model.py가 교차검증으로 자동 선택하고
# manifest에 기록한다. 런타임은 manifest 순서를 그대로 따른다.
# 2D 투영 기반 - 카메라가 정측면에서 벗어나면 급격히 무너진다.
# 3D 신체 좌표계 기반 - 카메라 위치와 무관하나 z 추정 노이즈에 취약하다.
ALL_FEATURES = [
    "posture_error",     # 2D: signed_delta - baseline
    "signed_delta",      # 2D: 귀-어깨 수평 offset / 수직 거리
    "abs_delta",         # 2D: 부호 없는 버전
    "fwd_ratio",         # 3D: 전방 이동량 / 몸통 길이
    "fwd_error",         # 3D: fwd_ratio - baseline
    "cva_deg",           # 3D: 몸통 수직축 대비 목 각도 (도)
    "cva_error",         # 3D: cva_deg - baseline
    "torso_angle_deg",   # 3D: 카메라 수직축 대비 몸통 기울기 (도)
    "torso_error",       # 3D: torso_angle_deg - baseline
    "obliquity",         # 시점: 어깨 이미지 간격 / 몸통 길이 (0=정측면)
]

# torso_angle_deg 절대값은 후보에 넣지 않는다.
# 이 값은 카메라 수직축이 기준이라 카메라를 기울이면 통째로 이동한다.
# baseline을 뺀 torso_error만 쓴다. 그래서 baseline 촬영과 자세 촬영 사이에
# 카메라를 옮기면 안 된다. 촬영 가이드에 이미 삼각대 고정이 필수로 적혀 있다.
#
# torso_error를 넣는 이유:
#   cva_error는 몸통을 기준축으로 삼으므로 몸 전체가 앞으로 기울면
#   목과 몸통이 같이 움직여 값이 거의 안 변한다.
#   즉 "목만 뺐다"와 "몸 전체를 숙였다"를 구분하지 못한다.
#   torso_error가 그 축을 따로 준다.

# 자동 선택 후보. 앞쪽이 더 단순한 조합이며, 동점이면 단순한 쪽을 고른다.
FEATURE_CANDIDATES = [
    # 2D 계열 - 카메라를 정측면에 고정할 수 있을 때
    ["posture_error"],
    ["posture_error", "abs_delta"],
    # 3D 계열 - 카메라 각도가 변할 때
    ["cva_error"],
    ["fwd_error"],
    ["fwd_error", "cva_error"],
    ["fwd_error", "cva_error", "obliquity"],
    # 몸통 축 추가 - 목만 뺀 것과 몸 전체를 숙인 것을 구분한다
    ["cva_error", "torso_error"],
    ["fwd_error", "torso_error"],
    ["fwd_error", "cva_error", "torso_error"],
    ["fwd_error", "cva_error", "torso_error", "obliquity"],
    # 혼합 - 정측면 근처에서는 2D가 더 정밀하므로 모델이 상황별로 쓰게 둔다
    ["posture_error", "fwd_error"],
    ["posture_error", "fwd_error", "cva_error", "obliquity"],
    ["posture_error", "fwd_error", "cva_error", "torso_error", "obliquity"],
]

# ─────────────────────────────────────────────────────────────
# CSV 스키마 - collect_data.py가 쓰고 train/evaluate가 읽는다
# ─────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "timestamp",
    "subject",
    "session",
    "label",
    "frame_id",
    "pose_detected",
    "side",
    "facing",
    "view",
    "signed_delta",
    "abs_delta",
    "fwd_ratio",
    "cva_deg",
    "torso_angle_deg",
    "obliquity",
    "baseline",
    "baseline_fwd",
    "baseline_cva",
    "baseline_torso",
    "posture_error",
    "fwd_error",
    "cva_error",
    "torso_error",
    "world_ok",
    "source",
    "model_complexity",
    "schema_version",
]

# 5: torso_angle_deg / baseline_torso / torso_error 추가
SCHEMA_VERSION = 5

# ─────────────────────────────────────────────────────────────
# 영상 기반 수집
# ─────────────────────────────────────────────────────────────
# source 열에는 이 행을 만든 출처를 적는다.
#   영상  → 파일명 (예: "s01_side_MIXED_a.mp4")
#   카메라 → "camera:{subject}/{session}/{label}"
# 같은 source를 다시 처리하면 기존 행을 지우고 새로 쓴다(멱등).
CAMERA_SOURCE_PREFIX = "camera:"

VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".m4v", ".MP4", ".MOV"]

# 영상에서 초당 몇 프레임을 CSV에 저장할지.
# 30fps 원본에서 인접 프레임은 거의 동일하므로 전부 쓸 이유가 없다.
# 낮추면 학습이 빨라지고 중복도 줄어든다.
VIDEO_SAMPLE_FPS = 5.0

# 영상에서 초당 몇 프레임을 MediaPipe에 통과시킬지.
# 위의 저장 간격과 다른 개념이며, 반드시 구분해야 한다.
#
# MediaPipe Python solutions는 process() 호출마다 내부 타임스탬프를
# 33333us씩 고정 증가시킨다(solution_base.py). 실제 경과 시간은 안 본다.
# 따라서 smooth_landmarks 평활 강도를 좌우하는 것은
# "호출 사이에 몸이 실제로 얼마나 움직였는가"다.
#
# 즉 학습 영상과 런타임의 처리 간격이 같아야 같은 값이 나온다.
#   런타임 20fps  → 호출 간 50ms
#   영상 5fps로 처리 → 호출 간 200ms  (움직임 4배 → 평활 결과가 달라짐)
#   영상 30fps 전부 처리 → 호출 간 33ms (이번엔 반대로 과함)
#
# 그래서 전부 처리하는 것이 아니라 CAMERA_FPS에 맞추는 것이 맞다.
# 30fps 원본에서 20fps를 뽑으면 3프레임 중 2개를 쓰게 된다.
#
# 실기에서 측정한 런타임 처리 속도가 CAMERA_FPS와 다르면
# 그 실측값으로 바꾸는 것이 좋다. collect_data.py가 실측치를 출력한다.
VIDEO_PROCESS_FPS = CAMERA_FPS

# 라벨 구간 앞뒤로 버릴 시간(초). 자세 전환 프레임이 섞이는 것을 막는다.
SEGMENT_TRIM_SEC = 0.5

# BASELINE은 라벨이 아니라 기준점 측정용 구간이다.
BASELINE_LABEL = "BASELINE"

# ─────────────────────────────────────────────────────────────
# 촬영 시점(view) 라벨
# ─────────────────────────────────────────────────────────────
# 각도 일반화를 측정하려면 어느 각도에서 찍었는지 기록해야 한다.
# train_model.py가 leave-one-view-out으로 이 축을 검증한다.
VIEWS = ["side", "oblique", "low", "high"]

# 시점 품질 게이트.
# obliquity가 이 값을 넘으면 정면에 가까워 2D 특징을 신뢰할 수 없다.
OBLIQUITY_2D_LIMIT = 0.45

# ─────────────────────────────────────────────────────────────
# 런타임 안정화 기본값
# ─────────────────────────────────────────────────────────────
# 나빠지는 방향(NORMAL→WARNING→BAD)은 빠르게, 좋아지는 방향은 느리게.
# 경고를 놓치는 비용이 헛경고 비용보다 크다.
HOLD_ESCALATE_SEC = 0.6
HOLD_RELAX_SEC = 2.0
PROBA_WINDOW = 7

# ─────────────────────────────────────────────────────────────
# 진동 밴드(XIAO ESP32-C3) 통신 계약
# ─────────────────────────────────────────────────────────────
# 펌웨어는 수신 문자열의 첫 글자만 읽는다.
# 상태 문자열을 그대로 보내면 NO_POSE가 'N'(=NORMAL)으로,
# PAUSED가 'P'(=NO_POSE)로 오해석된다.
# NORMAL/WARNING/BAD는 우연히 맞아떨어져 테스트가 통과하므로
# 반드시 이 표를 거쳐야 한다.
BAND_COMMANDS = {
    "NORMAL": "N",
    "WARNING": "W",
    "BAD": "B",
    "NO_POSE": "P",
    "PAUSED": "S",
}

# 상태 유지 중 재전송할 문자.
# 펌웨어에서 N/B/P/S는 멱등이므로 현재 명령을 그대로 다시 보낸다.
# 이렇게 하면 통신 이상이나 재연결로 밴드 상태가 초기화돼도
# 다음 주기에 자동 복구된다.
# W만 예외다. 600ms짜리 경고 패턴이 끝난 뒤 다시 받으면 재발동하므로
# 유지 중에는 heartbeat 문자를 보낸다.
BAND_HEARTBEAT_CHAR = "H"
BAND_NON_IDEMPOTENT = {"WARNING"}

# 펌웨어 타임아웃은 5초. 한 번 놓쳐도 버티도록 2초로 잡는다.
BAND_KEEPALIVE_SEC = 2.0

# WARNING은 2펄스 뒤 멈춘다. 계속 나쁜 자세면 이 주기로 다시 알린다.
# None이면 진입 시 1회만 알린다.
BAND_WARNING_REPEAT_SEC = 30.0


# ─────────────────────────────────────────────────────────────
# 하드웨어 핀 배치 (라즈베리파이 BCM 번호)
# ─────────────────────────────────────────────────────────────
# 실물 LED에서 확인한 채널 배선이다. 색이 바뀌면 여기만 고친다.
LED_PIN_RED = 22
LED_PIN_GREEN = 27
LED_PIN_BLUE = 17

# 3핀 SPDT 슬라이드 스위치를 2선 active-LOW 모드 스위치로 쓴다.
#   가운데 핀 → GPIO23 (물리 16번)
#   바깥 핀 하나 → GND
#   나머지 바깥 핀 → 미연결
# LOW  = 자세 기능 일시정지 (OS/VNC는 계속 살아 있다)
# HIGH = 자세 기능 활성 + 모든 일시정지 해제 (마스터 재개)
# 내부 풀업이 스위치가 열려 있을 때 HIGH를 유지한다.
MODE_SWITCH_PIN = 23
MODE_SWITCH_DEBOUNCE_SEC = 0.25
MODE_SWITCH_POLL_SEC = 0.05
MODE_SWITCH_STARTUP_GRACE_SEC = 0.50

# ─────────────────────────────────────────────────────────────
# BLE 진동 밴드 (XIAO ESP32-C3)
# ─────────────────────────────────────────────────────────────
# 명령 문자는 위쪽 BAND_COMMANDS가 단일 출처다. 여기에 다시 적지 않는다.
BLE_DEVICE_NAME = "Posture-Band"
BLE_SERVICE_UUID = "abcdefaa-1234-5678-1234-abcdefabcdef"
BLE_COMMAND_CHAR_UUID = "abcdefab-1234-5678-1234-abcdefabcdef"
BLE_BATTERY_CHAR_UUID = "abcdefac-1234-5678-1234-abcdefabcdef"
BLE_SCAN_TIMEOUT_SEC = 2.0
BLE_RETRY_SEC = 0.25
BLE_CONNECT_TIMEOUT_SEC = 8.0
BLE_POLL_SEC = 0.1

# P와 S는 앞에 N을 먼저 보낸다.
# N/W/B만 아는 구버전 펌웨어에서도 일시정지·종료 시 모터가 꺼지도록 하기 위함이다.
BAND_PREFIX_WITH_NORMAL = {"P", "S"}

# ─────────────────────────────────────────────────────────────
# 런타임 카메라 워커
# ─────────────────────────────────────────────────────────────
CAMERA_REOPEN_SEC = 1.0
CAMERA_WAIT_SEC = 0.75
CAMERA_FATAL_TIMEOUT_SEC = 20.0

# ─────────────────────────────────────────────────────────────
# 런타임 캘리브레이션
# ─────────────────────────────────────────────────────────────
CALIBRATION_SEC = 3.0
CALIBRATION_RETRY_SEC = 2.0
CALIBRATION_MIN_SAMPLES = 12

# 캘리브레이션 중 signed_delta IQR 상한.
# 넘으면 자세가 흔들린 것으로 보고 다시 잰다.
CALIBRATION_MAX_SPREAD = 0.08

# 캘리브레이션 중 3D 좌표 유효 비율이 이 값 미만이면 경고한다.
# 3D 특징을 쓰는 모델이라면 baseline 자체가 부정확해진다.
CALIBRATION_MIN_WORLD_RATIO = 0.90

# ─────────────────────────────────────────────────────────────
# 지속시간 기반 악화
# ─────────────────────────────────────────────────────────────
# StatusStabilizer는 "분류기가 BAD라고 말한 상태가 얼마나 유지됐나"를 본다.
# 이것과 별개로, 분류기가 WARNING만 계속 말하더라도 그 상태가 오래 지속되면
# BAD로 올리는 규칙이 있다. V12.2부터 있던 동작이며 사용자 입장에서는
# "계속 나쁜 자세면 결국 강하게 알린다"에 해당한다.
#
# None으로 두면 이 규칙이 꺼지고 분류기 판정만 쓴다.
WARNING_TO_BAD_SEC = 2.0

# ─────────────────────────────────────────────────────────────
# 자세 점수 (UI 표시용)
# ─────────────────────────────────────────────────────────────
# 100점에서 시작해 오차에 비례해 깎는다. 판정에는 쓰이지 않는다.
# 축마다 단위가 달라(비율 vs 도) 하나의 계수로는 맞출 수 없으므로
# manifest의 threshold_hint 축을 기준으로 정규화한다.
SCORE_MAX = 100
