"""
DMS (Driver Monitoring System) – Improved Version

UPGRADES:
 - Uses YOLOv8m (MUCH better phone detection)
 - Lowered confidence threshold for small objects
 - Improved phone class detection (matches many names)
 - Phone near face detection using bounding box overlap
 - More accurate & stable alerts
"""

import time
import math
from collections import deque, defaultdict

import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from cvzone import putTextRect

# ---------------------- CONFIG ----------------------
CAM_INDEX = 0
WEIGHTS = "yolov8m.pt"          # MUCH more accurate for phone detection
CONF_THRESH = 0.20              # lower threshold catches small objects

EAR_THRESHOLD = 0.23
EAR_CONSEC_FRAMES = 3
SUSTAINED_ALERT_SECONDS = 1.6

mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

L_EYE_IDX = [33, 160, 158, 133, 153, 144]
R_EYE_IDX = [362, 385, 387, 263, 373, 380]


# ---------------------- UTILS ----------------------
def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def eye_aspect_ratio(eye_pts):
    p1, p2, p3, p4, p5, p6 = eye_pts
    A = dist(p2, p6)
    B = dist(p3, p5)
    C = dist(p1, p4)
    if C == 0:
        return 0
    return (A + B) / (2.0 * C)

def is_eye_covered(eye_pts, hand_landmarks, w, h):
    if hand_landmarks is None:
        return False

    ex = [p[0] for p in eye_pts]
    ey = [p[1] for p in eye_pts]
    x1, y1 = min(ex), min(ey)
    x2, y2 = max(ex), max(ey)

    for hand in hand_landmarks:
        for lm in hand.landmark:
            hx, hy = int(lm.x * w), int(lm.y * h)
            if x1 <= hx <= x2 and y1 <= hy <= y2:
                return True
    return False

def boxes_overlap(boxA, boxB):
    (x1, y1, x2, y2) = boxA
    (X1, Y1, X2, Y2) = boxB
    return not (x2 < X1 or x1 > X2 or y2 < Y1 or y1 > Y2)


# ---------------------- MODEL LOAD ----------------------
print("Loading YOLO model...")
yolo = YOLO(WEIGHTS)

cls_map = yolo.model.names

# detect many phone class names
phone_cls_idx = []
for idx, name in cls_map.items():
    n = name.lower()
    if any(k in n for k in ["phone", "cell", "mobile"]):
        phone_cls_idx.append(idx)

print("Detected phone-related class IDs:", phone_cls_idx)

face_mesh = mp_face.FaceMesh(refine_landmarks=True)
hands = mp_hands.Hands(max_num_hands=2)

# State variables
eye_closed_counter = 0
alerts = {"eyes_closed": False, "eyes_covered": False, "phone_near_face": False}
last_alert_time = defaultdict(lambda: 0)
ear_history = deque(maxlen=6)


# ---------------------- MAIN LOOP ----------------------
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Camera not found!")

prev_time = time.time()
fps = 0
frame_count = 0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        h, w = frame.shape[:2]

        # YOLO detection
        results = yolo(frame, conf=CONF_THRESH, verbose=False)[0]

        phone_boxes = []
        if results.boxes is not None:
            for box in results.boxes:
                cid = int(box.cls[0])

                # keep only phone classes
                if cid in phone_cls_idx:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    phone_boxes.append((x1, y1, x2, y2))

        # Mediapipe face mesh
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_res = face_mesh.process(rgb)

        left_ear = right_ear = None
        face_box = None

        if face_res.multi_face_landmarks:
            lm = face_res.multi_face_landmarks[0].landmark
            pts = [(int(p.x*w), int(p.y*h)) for p in lm]

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            fx1, fy1, fx2, fy2 = min(xs), min(ys), max(xs), max(ys)
            face_box = (fx1, fy1, fx2, fy2)

            # draw face box
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255,200,0), 1)

            left_pts = [pts[i] for i in L_EYE_IDX]
            right_pts = [pts[i] for i in R_EYE_IDX]

            left_ear = eye_aspect_ratio(left_pts)
            right_ear = eye_aspect_ratio(right_pts)

            # draw eye landmarks
            for (x, y) in left_pts + right_pts:
                cv2.circle(frame, (x,y), 1, (0,255,0), -1)

        # Hand tracking (no drawing)
        hand_res = hands.process(rgb)
        hand_landmarks = hand_res.multi_hand_landmarks if hand_res.multi_hand_landmarks else None

        # EYE CLOSED LOGIC
        now = time.time()
        if left_ear and right_ear:
            avg_ear = (left_ear + right_ear) / 2
            if avg_ear < EAR_THRESHOLD:
                eye_closed_counter += 1
            else:
                eye_closed_counter = 0

        if eye_closed_counter >= EAR_CONSEC_FRAMES:
            if now - last_alert_time["eyes_closed"] > SUSTAINED_ALERT_SECONDS:
                alerts["eyes_closed"] = True
                last_alert_time["eyes_closed"] = now
        else:
            alerts["eyes_closed"] = False

        # EYE COVERED LOGIC
        eyes_covered = False
        if face_res.multi_face_landmarks and hand_landmarks:
            if is_eye_covered(left_pts, hand_landmarks, w, h) or \
               is_eye_covered(right_pts, hand_landmarks, w, h):
                eyes_covered = True

        if eyes_covered:
            if now - last_alert_time["eyes_covered"] > 0.8:
                alerts["eyes_covered"] = True
                last_alert_time["eyes_covered"] = now
        else:
            alerts["eyes_covered"] = False

        # PHONE NEAR FACE USING OVERLAP (most reliable)
        phone_near_face_flag = False
        if face_box and phone_boxes:
            for pb in phone_boxes:
                if boxes_overlap(pb, face_box):
                    phone_near_face_flag = True
                    break

        if phone_near_face_flag:
            if now - last_alert_time["phone_near_face"] > SUSTAINED_ALERT_SECONDS:
                alerts["phone_near_face"] = True
                last_alert_time["phone_near_face"] = now
        else:
            alerts["phone_near_face"] = False

        # DRAW PHONE BOXES ONLY
        for (x1, y1, x2, y2) in phone_boxes:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,120,255), 2)
            putTextRect(frame, "PHONE", (x1, y1-25), scale=2, colorR=(0,120,255))

        # DRAW ALERTS
        sy = 80
        if alerts["eyes_closed"]:
            putTextRect(frame, "ALERT: Eyes Closed!", (10, sy), scale=2, colorR=(0,0,255))
            sy += 45
        if alerts["eyes_covered"]:
            putTextRect(frame, "ALERT: Eyes Covered!", (10, sy), scale=2, colorR=(0,0,255))
            sy += 45
        if alerts["phone_near_face"]:
            putTextRect(frame, "ALERT: Phone Near Face!", (10, sy), scale=2, colorR=(255,150,0))
            sy += 45

        # FPS
        frame_count += 1
        tnow = time.time()
        if tnow - prev_time >= 1:
            fps = frame_count / (tnow - prev_time)
            prev_time = tnow
            frame_count = 0

        cv2.putText(frame, f"FPS: {fps:.1f}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,0), 2)

        cv2.imshow("DMS Monitor", frame)
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
