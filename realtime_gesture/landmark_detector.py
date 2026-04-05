import cv2
import mediapipe as mp
from typing import Tuple, Optional, Any

class LandmarkDetector:
    def __init__(self, max_hands: int = 1, min_conf: float = 0.7):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_conf,
            min_tracking_confidence=min_conf
        )

    def process_frame(self, frame: Any) -> Tuple[Optional[Any], Any]:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True

        detected_hand_landmarks = None

        if results.multi_hand_landmarks:
            detected_hand_landmarks = results.multi_hand_landmarks[0]
            
            self.mp_draw.draw_landmarks(
                frame, 
                detected_hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

        return detected_hand_landmarks, frame

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
                
                # รับค่า Object มือดิบๆ มาจากฟังก์ชัน
                hand_landmarks, annotated_img = detector.process_frame(img)
                
                if hand_landmarks:
                    # นับจำนวนจุดให้ดึงจาก Object ของ MediaPipe
                    num_points = len(hand_landmarks.landmark)
                    cv2.putText(annotated_img, f"Hand Detected: {num_points} points", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Landmark Detection", annotated_img)
                
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                    
    except Exception as e:
        print(f"[ERROR] Test Failed: {e}")
    finally:
        if 'detector' in locals():
            detector.release()