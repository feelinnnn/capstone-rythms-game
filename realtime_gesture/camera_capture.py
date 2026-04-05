import cv2
import numpy as np
from typing import Tuple, Optional

class CameraCapture:
    def __init__(self, camera_index: int = 0):
        print(f"[*] Initializing Webcam (Index: {camera_index})...")
        self.cap = cv2.VideoCapture(camera_index)
        
        # ถ้าเปิดกล้องตัวแรกไม่ติด ให้ลองหา Index สำรอง
        if not self.cap.isOpened():
            print(f"[!] Cannot open camera index {camera_index}. Trying index 1...")
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                raise ValueError("[ERROR] Failed to connect to any Webcam.")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # บังคับให้คอมพิวเตอร์ดึง "ภาพวินาทีล่าสุด" เสมอ
        print("[SUCCESS] Camera is ready!")

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        success, frame = self.cap.read()
        if not success:
            print("[ERROR] Failed to grab frame.")
            return False, None
            
        frame = cv2.flip(frame, 1) # Mirror
        return True, frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("[*] Camera released.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

if __name__ == "__main__":
    try:
        with CameraCapture() as cam:
            print("[*] Press 'Esc' to exit.")
            while True:
                success, img = cam.get_frame()
                if not success:
                    break
                    
                cv2.imshow("Camera Capture [Press 'Esc' to close]", img)
                
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                    
    except Exception as e:
        print(f"[ERROR] Camera Test Failed: {e}")