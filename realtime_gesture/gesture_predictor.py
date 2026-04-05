import cv2
import socket
import json
import joblib
from pathlib import Path
from typing import Dict

from camera_capture import CameraCapture
from landmark_detector import LandmarkDetector
from feature_extractor import FeatureExtractor

# --- CONFIG ---
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "mlp_model.pkl"
CONFIG_PATH = ROOT_DIR / "config" / "gesture_labels.json"

UDP_IP = "127.0.0.1"
UDP_PORT = 5052

# สวิตช์ปิด/เปิดหน้าต่างวิดีโอ 
# (ตอนเทสให้เป็น True / ตอนเชื่อม Unity ให้แก้เป็น False เพื่อความเร็วสูงสุด)
SHOW_VIDEO = False  

class GesturePredictorApp:
    def __init__(self):
        print("=== Initializing Rhythm Game Backend ===")
        self.index_to_label = self._load_config()
        self.model = self._load_model()
        
        # สร้างดิกชันนารีเก็บ Bytes ที่แปลงรอไว้แล้ว (ไม่เสียเวลาแปลงตอนรันจริง)
        self.encoded_labels = {idx: label.encode('utf-8') for idx, label in self.index_to_label.items()}
        
        self.detector = LandmarkDetector()
        self.extractor = FeatureExtractor()
        
        # สร้างท่อ Socket รอไว้
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _load_config(self) -> Dict[int, str]:
        #โหลดไฟล์ JSON และสลับ Key-Value ให้พร้อมใช้งาน
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
            return {v: k for k, v in config["label_map"].items()}
        except Exception as e:
            raise RuntimeError(f"Config load error: {e}")

    def _load_model(self):
        #โหลดไฟล์ AI
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"Model load error: {e}")

    def run(self):
        #เริ่มประมวลผล
        print(f"[*] ท่อส่ง Socket เปิดแล้ว! เป้าหมาย -> {UDP_IP}:{UDP_PORT}")
        if SHOW_VIDEO:
            print("[*] โหมดแสดงภาพ: เปิด (กด 'Esc' เพื่อปิดโปรแกรม)")
        else:
            print("[*] โหมดแสดงภาพ: ปิด [HEADLESS MODE] (กด Ctrl+C ที่ Terminal เพื่อหยุด)")

        try:
            with CameraCapture() as cam:
                while True:
                    success, frame = cam.get_frame()
                    if not success:
                        break

                    # สแกนหามือ
                    hand_landmarks, annotated_frame = self.detector.process_frame(frame)

                    if hand_landmarks:
                        # จัดทรงข้อมูล 63 ตัวเลข
                        features = self.extractor.extract_features(hand_landmarks)
                        
                        # ทายผล
                        prediction_index = self.model.predict([features])[0]
                        
                        # ดึง Bytes ที่แปลงไว้แล้วมายิงเข้า Socket ได้เลย
                        byte_data = self.encoded_labels[prediction_index]
                        self.sock.sendto(byte_data, (UDP_IP, UDP_PORT))

                        # ถ้าปิดจอไว้ โค้ดส่วนวาดภาพจะไม่ทำงาน ลดภาระ CPU ได้มหาศาล
                        if SHOW_VIDEO:
                            gesture_name = self.index_to_label[prediction_index]
                            cv2.putText(annotated_frame, f"Predict: {gesture_name}", (10, 40), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # โชว์ภาพเฉพาะตอนที่เปิดสวิตช์
                    if SHOW_VIDEO:
                        cv2.imshow("Rhythm Game Backend", annotated_frame)
                        if cv2.waitKey(1) & 0xFF == 27:
                            break

        except KeyboardInterrupt:
            # ดักจับการกด Ctrl+C เผื่อตอนที่รันแบบปิดจอ
            print("\n[*] หยุดการทำงานโดยผู้ใช้")
        except Exception as e:
            print(f"[ERROR] ระบบขัดข้อง: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        self.detector.release()
        self.sock.close()
        print("[*] ปิดการเชื่อมต่อเรียบร้อยแล้ว")

# --- จุดสตาร์ทโปรแกรม ---
if __name__ == "__main__":
    try:
        app = GesturePredictorApp()
        app.run()
    except Exception as e:
        print(f"[FATAL ERROR] {e}")