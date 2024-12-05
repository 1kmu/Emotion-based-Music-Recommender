import mediapipe as mp
import numpy as np
import cv2

# Initialize video capture
cap = cv2.VideoCapture(0)

# Verify camera initialization
if not cap.isOpened():
    print("Error: Camera not initialized. Please check your device.")
    exit()

name = input("Enter the name of the data: ")

# Mediapipe setup
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

X = []
data_size = 0

# Define the expected size of each data sample
FACE_LANDMARKS = 468 * 2  # 468 landmarks with x, y
HAND_LANDMARKS = 21 * 2   # 21 landmarks with x, y per hand
EXPECTED_SIZE = FACE_LANDMARKS + HAND_LANDMARKS * 2  # Face + both hands

while True:
    lst = []

    ret, frm = cap.read()

    # Check if the frame is successfully captured
    if not ret or frm is None:
        print("Warning: Failed to capture frame. Skipping...")
        continue

    frm = cv2.flip(frm, 1)

    # Process the frame
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    if res.face_landmarks:
        # Add face landmarks (relative to landmark[1])
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        # Add left hand landmarks (relative to landmark[8])
        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            lst.extend([0.0] * HAND_LANDMARKS)  # Placeholder for missing left hand

        # Add right hand landmarks (relative to landmark[8])
        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            lst.extend([0.0] * HAND_LANDMARKS)  # Placeholder for missing right hand

        # Ensure the list is of the expected size
        if len(lst) == EXPECTED_SIZE:
            X.append(lst)
            data_size += 1

    # Draw landmarks on the frame
    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    # Display the data size on the frame
    cv2.putText(frm, str(data_size), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Camera Window", frm)

    # Break the loop on 'Esc' key or when data size exceeds 99
    if cv2.waitKey(1) == 27 or data_size > 99:
        cv2.destroyAllWindows()
        cap.release()
        break

# Save the data as a NumPy array
try:
    X_array = np.array(X, dtype=np.float32)
    np.save(f"{name}.npy", X_array)
    print(f"Data saved successfully. Shape: {X_array.shape}")
except Exception as e:
    print(f"Error saving data: {e}")
