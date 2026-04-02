import cv2
import mediapipe as mp
import pandas as pd
import os
import time
import json
from pathlib import Path
import argparse

# CONFIG
CONFIG_PATH = "config/gesture_labels.json"

with open(CONFIG_PATH, "r" ,encoding="utf-8") as f:
    config = json.load(f)

GESTURES = config["gestures"]
HANDS = config["hands"]
LABEL_MAP = config["label_map"]
DISPLAY_NAMES = config["display_names"]

FRAMES_PER_SESSION = config["frames_per_session"] # 200
COOLDOWN_SECONDS = config["cooldown_seconds"] # 30
TARGET_PER_USER = 1000 # เป้าหมาย 1,000 เฟรมต่อ 1 คน

SAVE_PATH = "data/raw_landmarks"
THRESHOLD = 0.01

os.makedirs(SAVE_PATH, exist_ok=True)

# ARGUMENT
parser = argparse.ArgumentParser()
parser.add_argument("--gesture", required=True)
parser.add_argument("--hand", required=True)
parser.add_argument("--user", required=True)
args = parser.parse_args()

gesture = args.gesture
hand = args.hand
user_id = args.user

if gesture not in GESTURES:
    raise ValueError(f"Invalid gesture: {gesture}")
if hand not in HANDS:
    raise ValueError(f"Invalid hand: {hand}")

class_key = f"{gesture}_{hand}"
label = LABEL_MAP[class_key]

# FILE
file_path = Path(f"{SAVE_PATH}/{class_key}.csv")

# MEDIAPIPE
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands_model = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# CAMERA
cap = cv2.VideoCapture(0)

# STATE & VARIABLES
state = "COUNTDOWN"
countdown_time = 3
cooldown_time = COOLDOWN_SECONDS
frame_count = 0
total_saved_this_run = 0 # นับยอดรวมเฉพาะของรอบนี้
session_rows = []
prev_row = None

start_time = time.time()

# GUIDELINE
variations = [
    "1. Baseline (Normal)", 
    "2. Scale (Near-Far)", 
    "3. Translate (Edges)", 
    "4. Rotate (Angles)", 
    "5. Environment (Light/BG)"
]

# MAIN LOOP
while True:
    check, frame = cap.read()
    if not check:
        break

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_model.process(rgb)
    h, w, _ = image.shape # ดึงความสูงหน้าจอเพื่อจัดวาง UI

    # คำนวณความคืบหน้า และ UI หมวดหมู่
    current_total = total_saved_this_run + frame_count
    current_variation_idx = min(current_total // FRAMES_PER_SESSION, 4)
    current_variation = variations[current_variation_idx] if current_total < TARGET_PER_USER else "DONE (1000 Samples)"

    if current_total >= TARGET_PER_USER:
        state = "COMPLETE"

    # ควบคุมสถานะสีและข้อความ
    if state == "COMPLETE":
        color, status_text = (255, 0, 255), "COLLECTION COMPLETE! Press 'Esc'"
    elif state == "COUNTDOWN":
        elapsed = int(time.time() - start_time)
        remain = countdown_time - elapsed
        color, status_text = (0, 255, 0), f"READY IN: {remain}s"
        if remain <= 0:
            state = "RECORDING"
    elif state == "RECORDING":
        color, status_text = (0, 0, 255), f"RECORDING ({frame_count}/{FRAMES_PER_SESSION})"
        
        # Logic
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                detected_hand = handedness.classification[0].label.lower()
                if detected_hand != hand:
                    continue

                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                base_x = hand_landmarks.landmark[0].x
                base_y = hand_landmarks.landmark[0].y
                row = {}
                for i, lm in enumerate(hand_landmarks.landmark):
                    row[f"x{i}"] = lm.x - base_x
                    row[f"y{i}"] = lm.y - base_y
                    row[f"z{i}"] = lm.z
                row["label"] = label
                row["user_id"] = user_id

                if prev_row:
                    diff = sum(abs(row[k] - prev_row[k]) for k in row if k.startswith("x") or k.startswith("y"))
                    if diff < THRESHOLD:
                        continue

                prev_row = row
                session_rows.append(row)
                frame_count += 1

        # เมื่อครบ 200 เฟรม (1 session)
        if frame_count >= FRAMES_PER_SESSION:
            df = pd.DataFrame(session_rows)
            if file_path.exists():
                df.to_csv(file_path, mode="a", header=False, index=False)
            else:
                df.to_csv(file_path, index=False)

            print(f"Saved {len(session_rows)} samples to {file_path}")
            total_saved_this_run += FRAMES_PER_SESSION
            
            if total_saved_this_run >= TARGET_PER_USER:
                state = "COMPLETE"
            else:
                state = "COOLDOWN"
                start_time = time.time()
            
            session_rows = []
            frame_count = 0
            prev_row = None

    elif state == "COOLDOWN":
        elapsed = int(time.time() - start_time)
        remain = cooldown_time - elapsed
        color, status_text = (0, 255, 255), f"REST: {remain}s"
        if remain <= 0:
            state = "COUNTDOWN"
            start_time = time.time()

    # วาด UI บนหน้าจอ
    cv2.putText(image, f"[{class_key}] ID: {user_id} | Total: {current_total}/{TARGET_PER_USER}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(image, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(image, f"Mode: {current_variation}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)

    cv2.imshow("Data Collection System", image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()