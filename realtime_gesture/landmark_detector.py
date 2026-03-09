import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

video = cv2.VideoCapture(0)

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7
) as hands_detection:

    while True:
        check, image = video.read()
        if not check:
            break
        image = cv2.flip(image, 1)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands_detection.process(rgb)

        if results.multi_hand_landmarks:

            for hand_landmarks, hand_label in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):

                hand_type = hand_label.classification[0].label
                print("Hand:", hand_type)

                landmark_list = []

                for id, lm in enumerate(hand_landmarks.landmark):

                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    landmark_list.append((cx, cy))

                print("Landmarks:", landmark_list)

                mp_draw.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        cv2.imshow("Frame", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

video.release()
cv2.destroyAllWindows()