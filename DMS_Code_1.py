# file: dms_realtime.py
# Requirements: ultralytics, opencv-python (or opencv-python-headless), cvzone, numpy

import time
import cv2
import numpy as np
from ultralytics import YOLO
from cvzone import putTextRect

# --- Config ---
WEIGHTS = "yolov8n.pt"  # small model; change to a custom model if you train one
CONF_THRESH = 0.35
TARGET_CLASS = "person"   # default target: 'person' (for diver)
CAM_INDEX = 0             # /dev/video0 or libcamera pipeline

# --- Initialize camera ---
cap = cv2.VideoCapture(CAM_INDEX)  # if using libcamera, you may need different capture method
if not cap.isOpened():
    raise RuntimeError("Could not open camera. If using Raspberry Pi Camera, ensure it's enabled and accessible.")

# Warmup FPS calc
prev_time = time.time()
frame_count = 0
fps = 0.0

# --- Load model ---
model = YOLO(WEIGHTS)  # will download weights if you don't have them locally

# Optionally limit classes to speed up inference (only detect person)
# find class index for person from model.names
person_cls_idx = None
for idx, name in model.model.names.items():
    if name == "person":
        person_cls_idx = idx
        break

# --- Main loop ---
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed, retrying...")
            time.sleep(0.1)
            continue

        # Optionally resize for speed
        h, w = frame.shape[:2]
        target_w = 640
        if w > target_w:
            scale = target_w / float(w)
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

        # Run detection (ultralytics YOLO object)
        # set conf and device if needed: model(frame, conf=CONF_THRESH, device='cpu')
        results = model(frame, conf=CONF_THRESH, verbose=False)  # returns Results object(s)

        # results can be a list per batch — take first
        res = results[0]
        boxes = res.boxes  # Boxes object
        # Extract detections
        detections = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.model.names[cls_id]
                # Filter by target class if required
                if label == TARGET_CLASS:
                    detections.append((x1, y1, x2, y2, label, conf))

        # Draw detections + overlays
        for (x1, y1, x2, y2, label, conf) in detections:
            text = f"{label} {conf:.2f}"
            # cvzone utility to draw with rectangle & background
            putTextRect(frame, text, (x1, y1 - 10), scale=1, colorR=(0,255,0), thickness=1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        # FPS
        frame_count += 1
        if time.time() - prev_time >= 1.0:
            fps = frame_count / (time.time() - prev_time)
            prev_time = time.time()
            frame_count = 0

        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        # Display (if you have X11 / screen)
        cv2.imshow("Diver Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopping...")

finally:
    cap.release()
    cv2.destroyAllWindows()
