"""
posture - 거북목 실시간 감지 시스템

세 계층으로 나뉜다.

    posture.judge      특징 → 상태 판정. 하드웨어 없이 동작하며 노트북에서 테스트 가능
    posture.hardware   카메라 / LED / BLE 밴드 / 슬라이드 스위치. 라즈베리파이 전용
    posture.app        두 계층을 묶는 실행 계층. 상태 공유, 캘리브레이션, UI, 메인 루프

계층 간 의존 방향은 한쪽이다.

    app  →  hardware  →  contract
     └───→  judge     →  features → contract

judge는 hardware를 모르고, hardware는 judge를 모른다.
그래서 판정 로직은 카메라 없이, 하드웨어 제어는 모델 없이 각각 시험할 수 있다.

contract.py가 모든 설정의 단일 출처다.
학습 도구(tools/)와 런타임이 같은 파일을 읽으므로 양쪽 설정이 어긋날 수 없다.
"""

__all__ = ["contract", "features", "judge", "hardware", "app"]
