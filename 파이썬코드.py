import cv2
import mediapipe as mp
import time
import numpy as np

# -------------------------------------------------------------
# [시뮬레이션 전용] 라즈베리파이 GPIO 에러 방지용 가상 클래스
# -------------------------------------------------------------
class DummyGPIO:
    BCM = OUT = LOW = HIGH = None
    def setmode(self, mode): pass
    def setup(self, pins, mode): pass
    def output(self, pin, state): pass
    def cleanup(self): pass

GPIO = DummyGPIO()

# -------------------------------------------------------------
# 하드웨어 핀 및 파라미터 설정
# -------------------------------------------------------------
MOTOR_PIN = 18
LED_R, LED_G, LED_B = 17, 27, 22

MARGIN = 0.1                
WEIGHT = 1.0                # Y축 정규화 적용으로 가중치 1.0 설정
BAD_POSTURE_LIMIT = 2.0      
VIBRATION_DURATION = 2.0     
VISIBILITY_THRESHOLD = 0.5   

# 현재 LED 상태 저장용 글로벌 변수
current_led_status = "GREEN"

GPIO.setmode(GPIO.BCM)
GPIO.setup([MOTOR_PIN, LED_R, LED_G, LED_B], GPIO.OUT)

def set_led(color):
    global current_led_status
    current_led_status = color
    GPIO.output(LED_R, GPIO.LOW); GPIO.output(LED_G, GPIO.LOW); GPIO.output(LED_B, GPIO.LOW)
    if color == "GREEN": GPIO.output(LED_G, GPIO.HIGH)
    elif color == "YELLOW": GPIO.output(LED_R, GPIO.HIGH); GPIO.output(LED_G, GPIO.HIGH)
    elif color == "RED": GPIO.output(LED_R, GPIO.HIGH)

def motor_on():
    GPIO.output(MOTOR_PIN, GPIO.HIGH)

def motor_off():
    GPIO.output(MOTOR_PIN, GPIO.LOW)

# -------------------------------------------------------------
# 실시간 웹캠 소스 선택 (노트북 내장 웹캠: 0)
# -------------------------------------------------------------
cap = cv2.VideoCapture(0)

# -------------------------------------------------------------
# MediaPipe 초기화
# -------------------------------------------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 귀와 어깨 좌표 및 'Y축 높이 대비 X축 오차 비율' 계산 함수
def get_ear_shoulder_data(results, frame_shape):
    if results.pose_landmarks is None:
        return None, None, None
    
    lm = results.pose_landmarks.landmark
    ear_lm = lm[mp_pose.PoseLandmark.LEFT_EAR.value]
    shoulder_lm = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    
    # 인식 신뢰도 검증
    if ear_lm.visibility < VISIBILITY_THRESHOLD or shoulder_lm.visibility < VISIBILITY_THRESHOLD:
        return None, None, None
    
    h, w, _ = frame_shape
    ear_pt = (int(ear_lm.x * w), int(ear_lm.y * h))
    shoulder_pt = (int(shoulder_lm.x * w), int(shoulder_lm.y * h))
    
    # 1. 귀-어깨 간 X축 거리 (거북목 시 늘어나는 거리)
    x_delta = abs(ear_lm.x - shoulder_lm.x)
    
    # 2. 귀-어깨 간 Y축 높이 차이 (카메라와 거리에 비례하여 증감하는 세로 축)
    y_height = abs(ear_lm.y - shoulder_lm.y)
    if y_height == 0:
        y_height = 0.001
        
    # 3. [핵심] Y축 높이로 X축 오차를 정규화하여 거리 변화 영향 차단
    normalized_delta = (x_delta / y_height) * WEIGHT
    
    return normalized_delta, ear_pt, shoulder_pt

def calibrate_baseline():
    print("\n=== [1단계] 사용자 맞춤형 0점 조절 시작 ===")
    print("3초간 바른 자세를 유지하세요. 기준값을 세팅합니다...")
    baseline_list = []
    start_calib = time.time()

    while time.time() - start_calib < 3.0:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (800, 600))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        weighted_delta, ear_pt, shoulder_pt = get_ear_shoulder_data(results, frame.shape)
        if weighted_delta is not None:
            baseline_list.append(weighted_delta)
            cv2.line(frame, ear_pt, shoulder_pt, (255, 255, 0), 2)
            cv2.circle(frame, ear_pt, 6, (0, 0, 255), -1)
            cv2.circle(frame, shoulder_pt, 6, (0, 255, 0), -1)
            
        cv2.putText(frame, "Calibrating... Keep good posture", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Posture Simulation", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    if len(baseline_list) == 0:
        raise RuntimeError("보정 실패: 영상에서 귀와 어깨 인식을 실패했습니다.")

    baseline = sum(baseline_list) / len(baseline_list)
    print(f"✔️ 기준값 세팅 완료 (정규화 기준 거리 비율: {baseline:.4f})\n")
    return baseline

# -------------------------------------------------------------
# 통계 집계용 변수 초기화
# -------------------------------------------------------------
total_monitored_time = 0.0   # 총 모니터링 시간
total_bad_time = 0.0         # 총 누적 나쁜 자세 시간
score_history = []           # 평균 점수 산출용 데이터 리스트

# -------------------------------------------------------------
# 메인 가동부
# -------------------------------------------------------------
try:
    baseline = calibrate_baseline()
    bad_timer = 0.0
    alert_triggered = False
    motor_start_time = None
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("\n카메라 연결이 중단되었습니다. 통계 화면으로 이동합니다...")
            break

        frame = cv2.resize(frame, (800, 600))
        current_time = time.time()
        elapsed_time = current_time - prev_time
        prev_time = current_time

        # 전체 모니터링 누적 시간
        total_monitored_time += elapsed_time

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        current_delta, ear_pt, shoulder_pt = get_ear_shoulder_data(results, frame.shape)

        if current_delta is not None:
            posture_error = current_delta - baseline
            bad_posture = posture_error > MARGIN
            
            # 관절 포인트 시각화
            cv2.line(frame, ear_pt, shoulder_pt, (255, 255, 0), 2)
            cv2.circle(frame, ear_pt, 7, (0, 0, 255), -1)      # 귀
            cv2.circle(frame, shoulder_pt, 7, (0, 255, 0), -1) # 어깨
        else:
            posture_error = 0.0
            bad_posture = False

        # 오차 점수 반영 (오차 1당 감점 비율 700)
        if posture_error > 0:
            posture_score = max(0, int(100 - (posture_error * 700)))
        else:
            posture_score = 100

        score_history.append(posture_score)

        # 상태 판단 및 하드웨어 피드백 제어
        if bad_posture:
            bad_timer += elapsed_time
            total_bad_time += elapsed_time  
            if bad_timer < BAD_POSTURE_LIMIT:
                set_led("YELLOW")
                motor_off()
            elif bad_timer >= BAD_POSTURE_LIMIT:
                set_led("RED")
                if not alert_triggered:
                    motor_on()
                    motor_start_time = time.time()
                    alert_triggered = True
        else:
            bad_timer = 0.0
            alert_triggered = False
            motor_start_time = None
            set_led("GREEN")
            motor_off()

        # 진동 모터 타이머 해제
        if motor_start_time is not None:
            if time.time() - motor_start_time >= VIBRATION_DURATION:
                motor_off()

        # -------------------------------------------------------------
        # UI 정보 출력 (배경 박스 없이 투명 출력)
        # -------------------------------------------------------------
        good_ratio = max(0.0, ((total_monitored_time - total_bad_time) / total_monitored_time) * 100) if total_monitored_time > 0 else 100.0

        status_text = "BAD" if bad_posture else "NORMAL"
        color_bgr = (0, 0, 255) if bad_posture else (0, 255, 0)

        cv2.putText(frame, f"STATUS : {status_text}", (35, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
        cv2.putText(frame, f"SCORE  : {posture_score} / 100", (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 215, 0), 2)
        cv2.putText(frame, f"TIMER  : {bad_timer:.1f}s", (35, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"TOTAL BAD : {total_bad_time:.1f}s", (35, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 1)
        cv2.putText(frame, f"GOOD RATIO: {good_ratio:.1f}%", (35, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)

        # -------------------------------------------------------------
        # 우측 하단 가상 LED UI (단일 원)
        # -------------------------------------------------------------
        if current_led_status == "RED":
            led_circle_color = (0, 0, 255)
        elif current_led_status == "YELLOW":
            led_circle_color = (0, 255, 255)
        else:
            led_circle_color = (0, 255, 0)

        cv2.circle(frame, (740, 540), 22, led_circle_color, -1)

        # -------------------------------------------------------------
        # 최종 화면 출력 및 종료 이벤트 처리
        # -------------------------------------------------------------
        cv2.imshow("Posture Simulation", frame)
        
        # 'q' 키를 누르거나 마우스로 창의 X 버튼을 클릭하면 종료 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or cv2.getWindowProperty("Posture Simulation", cv2.WND_PROP_VISIBLE) < 1:
            print("\n사용자에 의해 모니터링이 종료되었습니다. 통계 화면으로 이동합니다...")
            break

    # -------------------------------------------------------------
    # 최종 통계 요약 카드 출력
    # -------------------------------------------------------------
    if total_monitored_time > 0:
        avg_score = sum(score_history) / len(score_history) if score_history else 0
        final_good_ratio = max(0.0, ((total_monitored_time - total_bad_time) / total_monitored_time) * 100)

        summary_bg = np.zeros((600, 800, 3), dtype=np.uint8)
        # 카드 배경 박스
        cv2.rectangle(summary_bg, (100, 80), (700, 520), (35, 35, 35), -1)
        cv2.rectangle(summary_bg, (100, 80), (700, 520), (0, 215, 255), 2)

        cv2.putText(summary_bg, "=== POSTURE ANALYSIS REPORT ===", (160, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2)
        
        cv2.putText(summary_bg, f"1. Total Time       : {total_monitored_time:.1f} sec", (150, 220), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(summary_bg, f"2. Total Bad Time   : {total_bad_time:.1f} sec", (150, 280), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 255), 2)
        cv2.putText(summary_bg, f"3. Average Score    : {avg_score:.1f} / 100", (150, 340), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 215, 0), 2)
        cv2.putText(summary_bg, f"4. Good Posture Rate: {final_good_ratio:.1f} %", (150, 400), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

        cv2.putText(summary_bg, "Press ANY KEY (or Spacebar) to Exit...", (220, 480), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        cv2.imshow("Posture Simulation", summary_bg)
        # 사용자가 아무 키나 누를 때까지 영구 대기
        cv2.waitKey(0)

except RuntimeError as e:
    print(f"\n❌ 에러 발생: {e}")
finally:
    motor_off()
    set_led("OFF")
    cap.release()
    cv2.destroyAllWindows()
    print("\n=== 시뮬레이션 프로그램이 안전하게 종료되었습니다 ===")