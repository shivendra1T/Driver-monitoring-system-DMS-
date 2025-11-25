"""
dms_eye_phone.py

Real-time Diver Monitoring System (DMS) with:
 - YOLOv8 (ultralytics) for person & phone detection
 - MediaPipe Face Mesh for eye tracking and EAR-based blink/eye-closed detection
 - MediaPipe Hands to infer if hand is holding phone (talking on phone)
 - Simple heuristics + timers to avoid spurious alerts
"""

import time
import math
from collections import deque, defaultdict

import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from cvzone import putTextRect

# -------------------- Config --------------------
CAM_INDEX = 0          # OpenCV camera index
WEIGHTS = "yolov8n.pt" # YOLOv8 small model
CONF_THRESH = 0.35

# EAR thresholds
EAR_THRESHOLD = 0.23      # below -> eye closed
EAR_CONSEC_FRAMES = 3     # number of frames to confirm closed eye

# Timers (seconds)
SUSTAINED_ALERT_SECONDS = 1.8  # must persist this long to alert

# Phone proximity thresholds (fraction of face width / height)
PHONE_NEAR_FACE_IOU_THRESH = 0.02  # tiny overlap or close centroid distance will count
PHONE_NEAR_FACE_CENTER_DIST_RATIO = 0.7  # phone center within this times face diagonal

# Mediapipe setups
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# Face mesh eye landmark indices (MediaPipe)
# Left eye landmarks: [33, 160, 158, 133, 153, 144]
# Right eye landmarks: [362, 385, 387, 263, 373, 380]
L_EYE_IDX = [33, 160, 158, 133, 153, 144]
R_EYE_IDX = [362, 385, 387, 263, 373, 380]

# -------------------- Helpers --------------------
def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def eye_aspect_ratio(eye_landmarks):
    # eye_landmarks: list of 6 (x,y) coordinates in pixel space: p1..p6
    # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    p1, p2, p3, p4, p5, p6 = eye_landmarks
    A = dist(p2, p6)
    B = dist(p3, p5)
    C = dist(p1, p4)
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def box_from_xyxy(xyxy):
    x1, y1, x2, y2 = xyxy
    return int(x1), int(y1), int(x2), int(y2)

def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2-x1) * max(0, y2-y1)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interA = interW * interH
    unionA = box_area(boxA) + box_area(boxB) - interA
    return interA / unionA if unionA > 0 else 0

# -------------------- Initialize models --------------------
print("Loading YOLO model...")
yolo = YOLO(WEIGHTS)  # will download weights if not present

# get class map from model (ultralytics)
cls_map = {}
if hasattr(yolo.model, "names"):
    cls_map = yolo.model.names  # index->name

# find numeric indices for person and phone if present
person_cls_idx = None
phone_cls_idx = None
for idx, name in cls_map.items():
    n = name.lower()
    if n == "person":
        person_cls_idx = idx
    if "phone" in n or "cell" in n or "mobile" in n:
        phone_cls_idx = idx

print("person_idx:", person_cls_idx, "phone_idx:", phone_cls_idx)

# MediaPipe instances
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# State variables
eye_closed_counters = {"left": 0, "right": 0}
eye_last_closed_time = {"left": 0.0, "right": 0.0}
alert_state = {"eye_closed": False, "on_phone": False, "talking_on_phone": False}
last_alert_time = defaultdict(lambda: 0.0)

# For smoothing EAR values
ear_history = deque(maxlen=6)

# -------------------- Main Loop --------------------
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera. Change CAM_INDEX or check camera access.")

prev_time = time.time()
fps = 0
frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]
        # Resize to limit processing (adjust for your machine)
        scale_w = 960
        if w > scale_w:
            scale = scale_w / w
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
            h, w = frame.shape[:2]

        # ---- YOLO detection ----
        results = yolo(frame, conf=CONF_THRESH, verbose=False)
        res = results[0]
        detections = []
        phones = []
        persons = []
        if res.boxes is not None and len(res.boxes) > 0:
            for box in res.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = cls_map.get(cls_id, str(cls_id)).lower()
                if cls_id == person_cls_idx or "person" in label:
                    persons.append(((x1,y1,x2,y2), conf))
                if phone_cls_idx is not None and cls_id == phone_cls_idx or ("phone" in label or "cell" in label):
                    phones.append(((x1,y1,x2,y2), conf))

        # ---- MediaPipe face mesh for eyes ----
        # convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb)

        left_ear = None
        right_ear = None
        face_boxes = []
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # derive pixel coordinates for selected landmarks
                lm = face_landmarks.landmark
                coords = [(int(l.x * w), int(l.y * h)) for l in lm]
                # compute bounding rect of face landmarks
                xs = [p[0] for p in coords]
                ys = [p[1] for p in coords]
                fx1, fy1, fx2, fy2 = min(xs), min(ys), max(xs), max(ys)
                face_boxes.append((fx1, fy1, fx2, fy2))

                # left eye
                try:
                    left_pts = [coords[i] for i in L_EYE_IDX]
                    right_pts = [coords[i] for i in R_EYE_IDX]
                except IndexError:
                    left_pts = right_pts = []
                if left_pts and right_pts:
                    left_ear = eye_aspect_ratio(left_pts)
                    right_ear = eye_aspect_ratio(right_pts)
                    ear_history.append((left_ear + right_ear)/2.0)

                    # draw eyes
                    for (x,y) in left_pts + right_pts:
                        cv2.circle(frame, (x,y), 1, (0,255,0), -1)

                    # draw face bbox
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 200, 0), 1)

        # ---- MediaPipe hands for hand detection ----
        hand_results = hands.process(rgb)
        hand_boxes = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # compute bounding rect of hand
                lms = hand_landmarks.landmark
                xs = [int(l.x * w) for l in lms]
                ys = [int(l.y * h) for l in lms]
                hx1, hy1, hx2, hy2 = min(xs), min(ys), max(xs), max(ys)
                hand_boxes.append((hx1, hy1, hx2, hy2))
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (200, 100, 255), 1)

        # ---- logic: eye closed detection ----
        now = time.time()
        # update counters for each eye using EAR values
        if left_ear is not None and right_ear is not None:
            if left_ear < EAR_THRESHOLD:
                eye_closed_counters["left"] += 1
            else:
                eye_closed_counters["left"] = 0

            if right_ear < EAR_THRESHOLD:
                eye_closed_counters["right"] += 1
            else:
                eye_closed_counters["right"] = 0

            # if both eyes closed enough frames => sustained closed
            if (eye_closed_counters["left"] >= EAR_CONSEC_FRAMES) and (eye_closed_counters["right"] >= EAR_CONSEC_FRAMES):
                # start timer for sustained alert
                if not alert_state["eye_closed"]:
                    if now - last_alert_time["eye_closed"] > SUSTAINED_ALERT_SECONDS:
                        alert_state["eye_closed"] = True
                        last_alert_time["eye_closed"] = now
                # else already alerted
            else:
                alert_state["eye_closed"] = False

            # overlay EAR
            avg_ear = (left_ear + right_ear) / 2.0
            putTextRect(frame, f"EAR: {avg_ear:.2f}", (10,60), scale=2, colorR=(0,200,200))
        else:
            # no face detected -> reset
            eye_closed_counters["left"] = eye_closed_counters["right"] = 0
            alert_state["eye_closed"] = False

        # ---- logic: phone / talk detection ----
        on_phone = False
        talking = False
        # For each detected person face box, check phone proximity
        for (fx1, fy1, fx2, fy2) in face_boxes:
            face_center = box_center((fx1, fy1, fx2, fy2))
            face_diag = math.hypot(fx2-fx1, fy2-fy1)
            for (pbox, pconf) in phones:
                phone_box = pbox
                # check overlap or center distance
                dcent = dist(face_center, box_center(phone_box))
                if dcent < face_diag * PHONE_NEAR_FACE_CENTER_DIST_RATIO:
                    # phone is close to face: mark as on_phone candidate
                    on_phone = True
                    # check if any hand overlaps the phone box
                    for hbox in hand_boxes:
                        if iou(hbox, phone_box) > 0.02:
                            # likely holding phone
                            talking = True
                            break
                else:
                    # also check IoU overlap small threshold
                    if iou((fx1,fy1,fx2,fy2), phone_box) > PHONE_NEAR_FACE_IOU_THRESH:
                        on_phone = True
                        for hbox in hand_boxes:
                            if iou(hbox, phone_box) > 0.02:
                                talking = True
                                break

        # sustain logic for phone alerts
        # on_phone/state transitions
        if on_phone:
            if not alert_state["on_phone"]:
                if now - last_alert_time["on_phone"] > SUSTAINED_ALERT_SECONDS:
                    alert_state["on_phone"] = True
                    last_alert_time["on_phone"] = now
        else:
            alert_state["on_phone"] = False

        if talking:
            if not alert_state["talking_on_phone"]:
                if now - last_alert_time["talking_on_phone"] > SUSTAINED_ALERT_SECONDS:
                    alert_state["talking_on_phone"] = True
                    last_alert_time["talking_on_phone"] = now
        else:
            alert_state["talking_on_phone"] = False

        # ---- draw YOLO boxes (persons, phones) ----
        for (person_box, conf) in persons:
            x1,y1,x2,y2 = person_box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            putTextRect(frame, f"Person {conf:.2f}", (x1, y1-20), scale=2, colorR=(0,200,0))

        for (phone_box, conf) in phones:
            x1,y1,x2,y2 = phone_box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (10,120,255), 2)
            putTextRect(frame, f"Phone {conf:.2f}", (x1, y1-20), scale=2, colorR=(10,120,255))

        # ---- overlays for alert states ----
        status_y = 100
        if alert_state["eye_closed"]:
            putTextRect(frame, "ALERT: Eyes Closed!", (10, status_y), scale=2, colorR=(0,0,255))
            status_y += 40
        if alert_state["on_phone"]:
            putTextRect(frame, "ALERT: On Phone (distracted)!", (10, status_y), scale=2, colorR=(255,140,0))
            status_y += 40
        if alert_state["talking_on_phone"]:
            putTextRect(frame, "ALERT: Talking on Phone!", (10, status_y), scale=2, colorR=(255,0,200))
            status_y += 40

        # fps
        frame_count += 1
        tnow = time.time()
        if tnow - prev_time >= 1.0:
            fps = frame_count / (tnow - prev_time)
            prev_time = tnow
            frame_count = 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,0), 2)

        # show
        cv2.imshow("DMS - Eye & Phone Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopping...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    hands.close()
