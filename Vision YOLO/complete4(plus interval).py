import cv2
from ultralytics import YOLO  # YOLO 클래스 임포트
import pygame
import os
import time
import sounddevice as sd
import numpy as np
import librosa
print("Program started...")


# YOLO 모델 로드
model = YOLO('D:/Temp/YOLO4/best.pt')
print("Model loaded...")

# 내장 카메라 사용
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print("Video stream opened...")

# 음성 파일 경로 설정
voice_path = 'D:/Temp/YOLO4/voice'

# 실시간 음정 분석 시작
print("Starting pitch detection...")
pitch_thread = threading.Thread(target=start_pitch_detection)
pitch_thread.start()
print("Pitch detection started...")

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    print("Frame read successfully...")

    # YOLO 모델로 예측 수행
    results = model.predict(frame)
    print("YOLO prediction complete...")

    # 예측 결과를 프레임에 표시
    annotated_frame = frame.copy()
    detected_labels = set()

    for result in results:
        boxes = result.boxes.xyxy.numpy()
        scores = result.boxes.conf.numpy()
        class_ids = result.boxes.cls.numpy()
        labels = model.names

        for box, score, class_id in zip(boxes, scores, class_ids):
            if score >= threshold:
                x1, y1, x2, y2 = map(int, box)
                label = labels[int(class_id)]
                detected_labels.add(label)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(annotated_frame, f'{label} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    print(f"Detected labels: {detected_labels}")

    # 프레임 표시
    cv2.imshow('YOLO Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exit command received...")
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("Resources released, program ending...")
