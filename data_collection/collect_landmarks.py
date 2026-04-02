import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

if not webcam.isOpened():
    print("Error: เปิดกล้องไม่ได้")
    exit()

while (True):
    check, image = webcam.read()

    if not check:
        print("Error: ไม่เจอภาพจากล้อง")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for landmark in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, landmark, mp_hands.HAND_CONNECTIONS)

    image = cv2.flip(image, 1)
    cv2.imshow("Webcam", image)
    if cv2.waitKey(1) & 0xFF == ord("e"):
        break

webcam.release()
cv2.destroyAllWindows()