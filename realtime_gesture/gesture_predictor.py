import os
import sys
import cv2
import socket
import json
import joblib
from pathlib import Path
from typing import Dict

from camera_capture import CameraCapture
from landmark_detector import LandmarkDetector
from feature_extractor import FeatureExtractor

# --- ฟังก์ชันช่วยหา Path สำหรับ PyInstaller (ต้องมีเพื่อให้รัน .exe ได้) ---
def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# --- CONFIG ---
MODEL_PATH = get_resource_path(os.path.join("models", "mlp_model.pkl"))
CONFIG_PATH = get_resource_path(os.path.join("config", "gesture_labels.json"))

UDP_IP = "127.0.0.1"
UDP_PORT = 5052
SHOW_VIDEO = False  # ปิดหน้าต่าง Python เพื่อความเร็วและไม่บังเกม

class GesturePredictorApp:
    def __init__(self):
        print("=== Initializing Rhythm Game Backend ===")
        self.index_to_label = self._load_config()
        self.model = self._load_model()
        
        self.detector = LandmarkDetector(max_hands=2) # แก้ให้รับได้ 2 มือ
        self.extractor = FeatureExtractor()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _load_config(self) -> Dict[int, str]:
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
            return {v: k for k, v in config["label_map"].items()}
        except Exception as e:
            raise RuntimeError(f"Config load error: {e}")

    def _load_model(self):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"Model load error: {e}")

    def run(self):
        print(f"[*] ท่อส่ง Socket เปิดแล้ว! เป้าหมาย -> {UDP_IP}:{UDP_PORT}")
        try:
            with CameraCapture() as cam:
                while True:
                    success, frame = cam.get_frame()
                    if not success: break

                    # สแกนหามือ (ส่งกลับมาเป็น List ของมือที่เจอ)
                    detected_hands, annotated_frame = self.detector.process_frame(frame)

                    # เตรียมสถานะสำหรับส่งให้ Unity
                    current_state = {"left": "none", "right": "none"}

                    if detected_hands:
                        for hand_info in detected_hands:
                            # 1. เช็คว่าเป็นมือซ้ายหรือขวา
                            raw_label = hand_info["label"]
                            real_hand = hand_info["label"].lower()
                            
                            # 2. ทำนายท่าทาง
                            features = self.extractor.extract_features(hand_info["landmarks"])
                            prediction_index = self.model.predict([features])[0]
                            gesture_name = self.index_to_label[prediction_index]
                            
                            # 3. บันทึกลงสถานะ
                            current_state[real_hand] = gesture_name

                    # --- จุดที่ต้องเปลี่ยน: ส่งเป็น JSON String ---
                    json_string = json.dumps(current_state)
                    self.sock.sendto(json_string.encode('utf-8'), (UDP_IP, UDP_PORT))

                    if SHOW_VIDEO:
                        status_text = f"L: {current_state['left']} | R: {current_state['right']}"
                        cv2.putText(annotated_frame, status_text, (10, 40), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.imshow("Rhythm Game Backend", annotated_frame)
                        if cv2.waitKey(1) & 0xFF == 27: break
        finally:
            self.cleanup()

    def cleanup(self):
        self.detector.release()
        self.sock.close()
        print("[*] ปิดการเชื่อมต่อเรียบร้อยแล้ว")

if __name__ == "__main__":
    app = GesturePredictorApp()
    app.run()