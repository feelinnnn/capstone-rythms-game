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
SHOW_VIDEO = True  

class GesturePredictorApp:
    def __init__(self):
        print("=== Initializing Rhythm Game Backend ===")
        self.index_to_label = self._load_config()
        self.model = self._load_model()
        
        self.detector = LandmarkDetector(max_hands=2)
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
                    detected_hands, annotated_frame = self.detector.process_frame(frame)

                    current_state = {
                        "left": "none",
                        "right": "none"
                    }

                    if detected_hands:
                        # วนลูปประมวลผลทีละข้าง
                        for hand_info in detected_hands: 
                            # Mirror ภาพ ต้องสลับค่ากลับให้ตรงกับความจริง
                            raw_label = hand_info["label"]
                            real_hand = "right" if raw_label == "left" else "left"
                            
                            hand_landmarks = hand_info["landmarks"]
                            
                            # จัดทรงและให้ AI ทายผล
                            features = self.extractor.extract_features(hand_landmarks)
                            prediction_index = self.model.predict([features])[0]
                            
                            # ได้ชื่อท่าทาง เช่น "v_right", "rock_left" 
                            # เราสามารถตัดคำว่า _right หรือ _left ออกได้เพื่อให้ Unity เอาไปเช็กง่ายขึ้น
                            # หรือจะส่งไปเต็มๆ ก็ได้ ในที่นี้จะส่งเต็มๆ ก่อน
                            gesture_name = self.index_to_label[prediction_index]
                            
                            # จับคำตอบใส่กล่องให้ถูกข้าง
                            current_state[real_hand] = gesture_name

                    # แปลงกล่อง current_state เป็น JSON String
                    json_string = json.dumps(current_state)

                    # แปลงเป็น Bytes แล้วยิงเข้า Socket
                    self.sock.sendto(json_string.encode('utf-8'), (UDP_IP, UDP_PORT))

                    if SHOW_VIDEO:
                        # โชว์สถานะซ้าย-ขวา บนจอ
                        status_text = f"L: {current_state['left']} | R: {current_state['right']}"
                        cv2.putText(annotated_frame, status_text, (10, 40), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        
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