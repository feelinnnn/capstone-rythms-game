import os
import sys
import cv2
import socket
import json
import joblib
import sklearn
import sklearn.ensemble
import base64
from pathlib import Path
from typing import Dict

from camera_capture import CameraCapture
from landmark_detector import LandmarkDetector
from feature_extractor import FeatureExtractor

# --- ฟังก์ชันช่วยหา Path สำหรับ PyInstaller ---
def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# --- CONFIG ---
MODEL_PATH = get_resource_path(os.path.join("models", "mlp_model.pkl"))
CONFIG_PATH = get_resource_path(os.path.join("config", "gesture_labels.json"))

UDP_IP = "127.0.0.1"
UDP_PORT = 5052
SHOW_VIDEO = True  # เปิดไว้สำหรับ Debug ในฝั่ง Python

class GesturePredictorApp:
    def __init__(self):
        print("=== Initializing Rhythm Game Backend (With Frame Stream) ===")
        self.index_to_label = self._load_config()
        self.model = self._load_model()
        
        self.detector = LandmarkDetector(max_hands=2) 
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
        print(f"[*] ระบบเริ่มทำงาน! ส่งข้อมูลไปที่ -> {UDP_IP}:{UDP_PORT}")
        try:
            with CameraCapture() as cam:
                while True:
                    success, frame = cam.get_frame()
                    if not success: break

                    # 1. สแกนหามือ
                    detected_hands, annotated_frame = self.detector.process_frame(frame)

                    # เตรียมสถานะสำหรับส่งให้ Unity
                    current_state = {"left": "none", "right": "none"}

                    if detected_hands:
                        for hand_info in detected_hands:
                            # เช็คว่าเป็นมือซ้ายหรือขวา (และแก้ Mirror ตามตรรกะโค้ดชุดล่าง)
                            raw_label = hand_info["label"]
                            real_hand = "right" if raw_label == "left" else "left"
                            
                            # ทำนายท่าทาง
                            features = self.extractor.extract_features(hand_info["landmarks"])
                            prediction_index = self.model.predict([features])[0]
                            gesture_name = self.index_to_label.get(prediction_index, "unknown")
                            
                            # บันทึกลงสถานะ
                            current_state[real_hand] = gesture_name

                    # 2. เตรียมเฟรมภาพสำหรับส่ง (Base64)
                    # Resize ให้เล็กลง (160x90) เพื่อไม่ให้ Packet ใหญ่เกินไปสำหรับ UDP
                    small_frame = cv2.resize(annotated_frame, (160, 90))
                    _, buffer = cv2.imencode(".jpg", small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                    jpg_as_text = base64.b64encode(buffer).decode("utf-8")

                    # 3. รวมข้อมูล Gestures และ Frame เข้าด้วยกัน
                    data_to_send = {
                        "gestures": current_state,
                        "frame": jpg_as_text
                    }

                    # 4. ส่งเป็น JSON String ผ่าน UDP
                    json_string = json.dumps(data_to_send)
                    self.sock.sendto(json_string.encode('utf-8'), (UDP_IP, UDP_PORT))

                    if SHOW_VIDEO:
                        # ใส่ข้อความ Debug บนหน้าต่าง Python ด้วย
                        cv2.putText(annotated_frame, f"L:{current_state['left']} R:{current_state['right']}", 
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow("Rhythm Game Backend", annotated_frame)
                        if cv2.waitKey(1) & 0xFF == 27: break
        finally:
            self.cleanup()

    def cleanup(self):
        self.detector.release()
        self.sock.close()
        cv2.destroyAllWindows()
        print("[*] ปิดการเชื่อมต่อเรียบร้อยแล้ว")

if __name__ == "__main__":
    app = GesturePredictorApp()
    app.run()