# dms_eye_phone.py
# Requirements: ultralytics, opencv-python, mediapipe, cvzone, numpy, imutils
# Test on PC webcam. Assumes ultralytics can load yolov8n.pt automatically.

import time
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from cvzone import putTextRect

# ----------------- Config -----------------
WEIGHTS = "yolov8n.pt"   # yolov8n (small) will be downloaded if missing
CONF_THRESH = 0.35

EAR_THRESH = 0.23
EAR_CONSEC_FRAMES = 6

MAR_THRESH = 0.45
MAR_CONSEC_FRAMES = 4

PHONE_OVERLAP_THRESH = 0.15  # fraction overlap of phone bbox with face bbox to count as near-head

CAM_INDEX = 0
# ------------------------------------------

# Mediapipe FaceMesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# YOLO model
model = YOLO(WEIGHTS)

# Helper: Euclidean distance
def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# Eye and mouth landmark indices (MediaPipe)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_UP = 13
MOUTH_DOWN = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

# EAR calculation
def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    # get coordinates
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((int(lm.x * img_w), int(lm.y * img_h)))
    # p1,p2,p3,p4,p5,p6 as per chosen indices
    p1,p2,p3,p4,p5,p6 = pts
    # vertical distances
    v1 = dist(p2, p6)
    v2 = dist(p3, p5)
    # horizontal
    h = dist(p1, p4)
    if h == 0:
        return 0.0
    ear = (v1 + v2) / (2.0 * h)
    return ear

# MAR calculation
def mouth_aspect_ratio(landmarks, img_w, img_h):
    up = landmarks[MOUTH_UP]
    down = landmarks[MOUTH_DOWN]
    left = landmarks[MOUTH_LEFT]
    right = landmarks[MOUTH_RIGHT]
    p_up = (int(up.x*img_w), int(up.y*img_h))
    p_down = (int(down.x*img_w), int(down.y*img_h))
    p_left = (int(left.x*img_w), int(left.y*img_h))
    p_right = (int(right.x*img_w), int(right.y*img_h))
    vert = dist(p_up, p_down)
    horiz = dist(p_left, p_right)
    if horiz == 0:
        return 0.0
    return vert / horiz

# Overlap / IoU helper (simple overlap ratio of phone area overlapping face bbox)
def bbox_overlap_fraction(boxA, boxB):
    # box = (x1,y1,x2,y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    if areaA == 0:
        return 0.0
    return interArea / areaA

# Camera
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")

# State counters
eye_counter = 0
mar_counter = 0
last_alert = None
fps_time = time.time()
frame_count = 0
fps=0.0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        # --- Run YOLO to detect person and phone ---
        results = model(frame, conf=CONF_THRESH, verbose=False)
        res = results[0]
        dets = []  # list of detections (class_name, conf, bbox)
        if hasattr(res, "boxes") and res.boxes is not None:
            for box in res.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                name = model.model.names[cls_id]
                dets.append((name,conf,(x1,y1,x2,y2)))

        # find phones
        phones = [d for d in dets if 'phone' in d[0].lower() or 'cell' in d[0].lower()]

        # Face mesh (we use mediapipe on entire frame to get primary face)
        mp_results = face_mesh.process(frame_rgb)
        status_texts = []

        face_box = None
        landmarks = None
        if mp_results.multi_face_landmarks:
            face_landmarks = mp_results.multi_face_landmarks[0].landmark
            # compute face bbox from landmarks
            xs = [lm.x for lm in face_landmarks]
            ys = [lm.y for lm in face_landmarks]
            min_x = int(min(xs) * w)
            max_x = int(max(xs) * w)
            min_y = int(min(ys) * h)
            max_y = int(max(ys) * h)
            # small padding
            pad_x = int(0.05 * (max_x-min_x))
            pad_y = int(0.08 * (max_y-min_y))
            face_box = (max(0,min_x-pad_x), max(0,min_y-pad_y), min(w, max_x+pad_x), min(h, max_y+pad_y))
            landmarks = face_landmarks

            # EAR for both eyes
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0

            # MAR
            mar = mouth_aspect_ratio(landmarks, w, h)

            # Eye closed logic
            if ear < EAR_THRESH:
                eye_counter += 1
            else:
                eye_counter = 0

            eyes_closed = eye_counter >= EAR_CONSEC_FRAMES
            if eyes_closed:
                status_texts.append("Eyes CLOSED")
                last_alert = "eyes_closed"

            # Mouth open logic
            if mar > MAR_THRESH:
                mar_counter += 1
            else:
                mar_counter = 0
            mouth_open = mar_counter >= MAR_CONSEC_FRAMES

            # Check phone near face
            phone_near = False
            phone_boxes = [p[2] for p in phones]
            for pb in phone_boxes:
                overlap = bbox_overlap_fraction(face_box, pb)
                # also check center of phone inside face bbox
                px_c = (pb[0]+pb[2])//2
                py_c = (pb[1]+pb[3])//2
                inside = (px_c >= face_box[0] and px_c <= face_box[2] and py_c >= face_box[1] and py_c <= face_box[3])
                # decide near if overlap or center inside
                if overlap > PHONE_OVERLAP_THRESH or inside:
                    phone_near = True
                    phone_box_near = pb
                    break

            # Decide talking vs distracted
            if phone_near and mouth_open:
                status_texts.append("Talking ON PHONE")
                last_alert = "talk_phone"
            elif phone_near:
                status_texts.append("Phone at ear / Distracted")
                last_alert = "phone_distracted"

            # Draw face box and landmarks if desired
            cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (255,200,0), 2)
            # draw a few landmark points for visual debug: eyes and mouth
            for idx in LEFT_EYE + RIGHT_EYE + [MOUTH_UP, MOUTH_DOWN, MOUTH_LEFT, MOUTH_RIGHT]:
                lm = landmarks[idx]
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(frame, (cx, cy), 2, (0,255,0), -1)

            # overlay EAR / MAR values
            putTextRect(frame, f"EAR:{ear:.2f}", (10,50), scale=1, thickness=1)
            putTextRect(frame, f"MAR:{mar:.2f}", (10,90), scale=1, thickness=1)

        else:
            # no face detected
            status_texts.append("No face detected")

        # Draw phone detections
        for name,conf,box in phones:
            x1,y1,x2,y2 = box
            color = (0,255,255)
            # if this phone was near, highlight
            if 'phone_box_near' in locals() and (box == phone_box_near):
                color = (0,0,255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            putTextRect(frame, f"{name} {conf:.2f}", (x1,y1-25), scale=0.7, thickness=1)

        # Draw overall status
        y0 = 130
        for i,txt in enumerate(status_texts):
            putTextRect(frame, txt, (10, y0 + i*35), scale=0.8, thickness=1, colorR=(255,100,100))

        # FPS
        frame_count += 1
        if time.time() - fps_time >= 1.0:
            fps = frame_count / (time.time() - fps_time)
            fps_time = time.time()
            frame_count = 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (w-150,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("DMS - Eye & Phone", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
