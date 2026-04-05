import cv2
import mediapipe as mp
from typing import Tuple, Optional, List, Any

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

    def process_frame(self, frame: Any) -> Tuple[Optional[List[float]], Any]:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True

        landmarks_data = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            self.mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            landmarks_data = []
            for lm in hand_landmarks.landmark:
                landmarks_data.extend([lm.x, lm.y, lm.z])

        return landmarks_data, frame

    def release(self):
        self.hands.close()

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
                
                landmarks, annotated_img = detector.process_frame(img)
                
                if landmarks:
                    cv2.putText(annotated_img, f"Data points: {len(landmarks)}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Landmark Detection", annotated_img)
                
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                    
    except Exception as e:
        print(f"[ERROR] Test Failed: {e}")
    finally:
        if 'detector' in locals():
            detector.release()