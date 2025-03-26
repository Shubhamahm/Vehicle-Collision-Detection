import torch
import cv2
import numpy as np
from time import time

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define object classes to detect
TARGET_CLASSES = ['car', 'motorcycle', 'truck', 'bus', 'pedestrian', 'traffic light', 'dog', 'cat', 'pothole', 'brake light', 'danger sign']

# Initialize video capture
cap = cv2.VideoCapture('video.mp4')  # Replace with 0 for webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    start_time = time()
    results = model(frame)
    
    # Extract detection results
    detections = results.pandas().xyxy[0]  # Convert tensor to Pandas DataFrame
    for _, row in detections.iterrows():
        class_name = row['name']
        if class_name in TARGET_CLASSES:
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            confidence = row['confidence']
            
            # Draw bounding box
            color = (0, 255, 0) if class_name != 'danger sign' else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{class_name} {confidence:.2f}', (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Show FPS
    fps = 1 / (time() - start_time)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imshow('Collision Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
