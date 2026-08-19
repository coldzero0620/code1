# 윈도우 + VS Code 실행 가이드

**결론: 학습 파이프라인 전체가 윈도우에서 됩니다.**
영상을 넣어 학습시키는 것이 목적이라면 라즈베리파이 없이 노트북만으로 끝납니다.

단 하나, `collect_data.py`(파이 카메라 실시간 수집)만 안 됩니다.
`rpicam-vid`라는 라즈베리파이 전용 명령을 쓰기 때문입니다.
윈도우에서 실행하면 시작하자마자 오류로 멈춥니다.

| 스크립트 | 윈도우 | 비고 |
|---|---|---|
| `ingest_video.py` | 된다 | 영상 → CSV |
| `measure_range.py` | 된다 | 사람별 자세 범위 측정 |
| `check_dataset.py` | 된다 | 학습 가능 여부 점검 |
| `train_model.py` | 된다 | 학습 |
| `evaluate_model.py` | 된다 | 검증 |
| `posture_runtime.py` | 된다 | 판정 로직 (BLE는 별도) |
| `collect_data.py` | **안 된다** | 라즈베리파이 카메라 전용 |

---

## 시작하기 전에

### 함정 하나: 파이썬 버전

**python.org에서 "최신"을 받으면 3.13이 깔리고, 설치가 실패합니다.**

`mediapipe 0.10.14`의 윈도우 휠은 3.9 ~ 3.12까지만 있습니다.
3.13용은 없습니다.

**3.12를 받으세요.**

### 함정 둘: mediapipe 버전

`pip install mediapipe`를 그냥 치면 최신 버전(1.0.1)이 깔립니다.
최신 버전에는 우리 코드가 쓰는 `mediapipe.solutions`가 **제거돼 있습니다.**
전부 실패합니다.

`requirements.txt`로 설치하면 자동으로 0.10.14가 깔립니다.

---

## STEP 1 — 파이썬 3.12 설치

1. https://www.python.org/downloads/windows/ 접속
2. **Python 3.12.x** 항목을 찾는다 (3.13이 아니다)
3. `Windows installer (64-bit)` 다운로드
4. 실행하고 **"Add python.exe to PATH" 체크** (중요)
5. `Install Now`

확인:

```
Win + R → cmd → 엔터
py -3.12 --version
```

`Python 3.12.x`가 나오면 성공입니다.

---

## STEP 2 — VS Code 설치

1. https://code.visualstudio.com 에서 다운로드 후 설치
2. VS Code 실행
3. 왼쪽 사이드바 확장(Extensions) 아이콘 클릭 — 네모 4개 모양
4. `Python` 검색 → Microsoft가 만든 것 설치

---

## STEP 3 — 프로젝트 폴더 만들기

원하는 위치에 아래 구조로 폴더를 만듭니다.
예를 들어 `C:\posture` 아래에:

```
C:\posture\
├── videos\           ← 영상을 넣을 곳 (비어 있어도 됨)
└── scripts\          ← 코드를 넣을 곳
```

`scripts` 폴더에 받은 파일 9개를 넣습니다.

```
contract.py
features.py
ingest_video.py
measure_range.py
collect_data.py
check_dataset.py
train_model.py
evaluate_model.py
posture_runtime.py
requirements.txt
```

압축 파일을 풀었다면 이미 이 구조로 되어 있으니 그대로 쓰면 된다.

**파일명이 `.py.txt`로 되어 있다면 `.txt`를 지우세요.**

> 윈도우가 확장자를 숨기고 있을 수 있습니다.
> 탐색기 상단 `보기` → `표시` → `파일 확장명`을 켜면 보입니다.

---

## STEP 4 — VS Code에서 폴더 열기

1. VS Code 상단 `File` → `Open Folder`
2. `C:\posture` 선택 (scripts가 아니라 상위 폴더)
3. "이 폴더의 작성자를 신뢰합니까?" → `예, 신뢰합니다`

---

## STEP 5 — 가상환경 만들기

터미널을 엽니다: 상단 `Terminal` → `New Terminal`
(단축키 `Ctrl` + `` ` ``)

터미널에 입력:

```
py -3.12 -m venv .venv
```

몇 초 걸립니다. 끝나면 왼쪽 파일 목록에 `.venv` 폴더가 생깁니다.

이어서 활성화:

```
.venv\Scripts\activate
```

프롬프트 앞에 `(.venv)`가 붙으면 성공입니다.

> **오류가 나는 경우**
> `이 시스템에서 스크립트를 실행할 수 없으므로...` 라고 나오면
> PowerShell 실행 정책 때문입니다. 아래를 한 번 실행하고 다시 시도하세요.
>
> ```
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```
>
> 또는 터미널 종류를 바꿔도 됩니다.
> 터미널 창 오른쪽 위 `+` 옆 아래화살표 → `Command Prompt` 선택

---

## STEP 6 — 패키지 설치

```
pip install -r scripts\requirements.txt
```

5~10분 정도 걸립니다. mediapipe가 큽니다.

확인:

```
python -c "import mediapipe as mp; print(mp.__version__, hasattr(mp,'solutions'))"
```

`0.10.14 True`가 나와야 합니다.

**`False`가 나오면** 최신 버전이 깔린 것입니다. 다시 설치하세요.

```
pip uninstall -y mediapipe
pip install mediapipe==0.10.14
```

---

## STEP 7 — VS Code에 인터프리터 알려주기

1. `Ctrl` + `Shift` + `P`
2. `Python: Select Interpreter` 입력 후 선택
3. 목록에서 `.venv` 가 붙은 것 선택
   (예: `Python 3.12.x ('.venv': venv)`)

이걸 해야 VS Code가 코드의 import를 인식하고 자동완성이 됩니다.

---

## STEP 8 — 영상 넣기

`C:\posture\videos\` 안에 영상을 넣습니다.

파일명 규칙:

```
s01_BASELINE.mp4              기준점 영상
s01_side_NORMAL_a.mp4         s01, 정측면, NORMAL, 세션 a
s01_side_BAD_a.mp4
```

자세가 여러 개 섞인 영상은 같은 폴더에 `segments.csv`를 만들어 구간을 적습니다.

```csv
file,subject,view,session,start,end,label
s01_side_MIXED_a.mp4,s01,side,a,0:00,0:30,BASELINE
s01_side_MIXED_a.mp4,s01,side,a,0:40,1:20,NORMAL
s01_side_MIXED_a.mp4,s01,side,a,1:30,2:10,BAD
```

> `segments.csv`를 엑셀로 만들 때는
> `다른 이름으로 저장` → `CSV UTF-8(쉼표로 분리)` 를 고르세요.
> 그냥 `CSV`로 저장하면 한글이 깨질 수 있습니다.

---

## STEP 9 — 실행

터미널에서 `scripts` 폴더로 이동:

```
cd scripts
```

순서대로 실행합니다.

```
python ingest_video.py
python check_dataset.py
python train_model.py
python evaluate_model.py
```

촬영 전에 사람별 자세 범위를 재려면:

```
python measure_range.py ../videos/s01_range.mp4
```

각 단계에서 무엇을 봐야 하는지는 `README_PIPELINE.md`에 있습니다.

---

## 자주 만나는 문제

| 증상 | 원인 | 해결 |
|---|---|---|
| `ERROR: Could not find a version that satisfies mediapipe==0.10.14` | 파이썬 3.13 | 3.12를 설치하고 가상환경을 다시 만든다 |
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | 최신 mediapipe | `pip install mediapipe==0.10.14` |
| `이 시스템에서 스크립트를 실행할 수 없으므로` | PowerShell 정책 | STEP 5의 안내 참고 |
| `ModuleNotFoundError: No module named 'contract'` | 잘못된 폴더에서 실행 | `cd scripts` 후 실행 |
| `rpicam-vid command was not found` | `collect_data.py`를 실행함 | 윈도우에서는 안 된다. `ingest_video.py`를 쓸 것 |
| `영상 폴더가 없습니다` | `videos` 폴더 위치 | `scripts`의 **상위** 폴더에 있어야 한다 |
| 한글이 `???`로 나옴 | 터미널 인코딩 | 아래 참고 |

### 한글 깨짐

VS Code 통합 터미널은 보통 문제없습니다.
만약 깨진다면 터미널에서 한 번 실행하세요.

```
chcp 65001
```

---

## 파일 경로 정리

```
C:\posture\
├── .venv\                    가상환경 (STEP 5에서 생성)
├── videos\
│   ├── s01_BASELINE.mp4
│   ├── s01_side_NORMAL_a.mp4
│   └── segments.csv          (선택)
├── models\                   train_model.py가 자동 생성
│   ├── posture-rf.joblib
│   └── split_manifest.json
└── scripts\
    ├── contract.py
    ├── features.py
    ├── ingest_video.py
    ├── measure_range.py
    ├── collect_data.py       (윈도우에서는 사용 불가)
    ├── check_dataset.py
    ├── train_model.py
    ├── evaluate_model.py
    ├── posture_runtime.py
    ├── requirements.txt
    └── posture_dataset.csv   ingest_video.py가 자동 생성
```

`models\`가 `scripts\`의 상위에 생기는 것이 정상입니다.
그래서 스크립트를 반드시 `scripts` 하위 폴더에 두어야 합니다.

---

## 팀원에게 공유할 때

폴더 전체를 압축해 보낼 때는 **`.venv` 폴더를 빼고** 보내세요.
용량이 크고, 다른 컴퓨터에서 동작하지 않습니다.

받는 사람은 STEP 1, 2, 5, 6, 7만 하면 됩니다.
