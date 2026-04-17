import cv2
import mediapipe as mp
from typing import Tuple, List, Dict, Any

class LandmarkDetector:
    def __init__(self, max_hands: int = 2, min_conf: float = 0.7):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_conf,
            min_tracking_confidence=min_conf
        )
    #List เก็บป้ายกำกับและพิกัด
    def process_frame(self, frame: Any) -> Tuple[List[Dict[str, Any]], Any]:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True

        detected_hands = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

                # ดึงป้ายกำกับ("Left" หรือ "Right") แปลงเป็นตัวพิมพ์เล็ก
                label = handedness.classification[0].label.lower()

                # แพ็กข้อมูลใส่ตะกร้า
                detected_hands.append({
                    "label": label,
                    "landmarks": hand_landmarks
                })

        return detected_hands, frame
    
    def release(self):
        self.hands.close()

# --- ส่วนทดสอบการทำงานกล้อง (Unit Test) ---
if __name__ == "__main__":
    from camera_capture import CameraCapture

    try:
        detector = LandmarkDetector()
        
        with CameraCapture() as cam:
            print("[*] Press 'Esc' to exit.")
            while True:
                success, img = cam.get_frame()
                if not success:
                    break
                
                # รับค่า Object มือดิบๆ
                detected_hands, annotated_img = detector.process_frame(img)
                
                hands_count = len(detected_hands)
                cv2.putText(annotated_img, f"Hands Detected: {hands_count}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Landmark Detection (Dual Hand)", annotated_img)
                
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                    
    except Exception as e:
        print(f"[ERROR] Test Failed: {e}")
    finally:
        if 'detector' in locals():
            detector.release()