import cv2
import mediapipe as mp
import csv
import os


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)

mp_draw = mp.solutions.drawing_utils

dataset_file = "dataset.csv"

label = input("Enter gesture label (example: A or Hello): ")

if not os.path.exists(dataset_file):
    with open(dataset_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = []
        for i in range(21):
            header.append(f"x{i}")
            header.append(f"y{i}")
        header.append("label")
        writer.writerow(header)

cap = cv2.VideoCapture(0)

print("Press 's' to save gesture sample")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

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

    cv2.putText(frame, f"Label: {label}", (10,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Collect Data", frame)

    key = cv2.waitKey(1)

    if key == ord('s') and result.multi_hand_landmarks:
        with open(dataset_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(landmarks + [label])

        print("Sample saved")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()