import cv2
import mediapipe as mp
import pickle
import numpy as np

print("Loading model...")

model = pickle.load(open("sign_model.pkl", "rb"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("Starting camera...")

while True:

    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    prediction = ""

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = []

            for lm in hand_landmarks.landmark:

                landmarks.append(lm.x)
                landmarks.append(lm.y)

            landmarks = np.array(landmarks).reshape(1, -1)

            prediction = model.predict(landmarks)

    cv2.rectangle(frame,(0,0),(300,80),(0,0,0),-1)

    cv2.putText(frame,
                f"Prediction: {prediction}",
                (10,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()