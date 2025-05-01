import cv2
import mediapipe as mp
import os

# Set target directory
data_dir = r"data\Help"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Directory created: {data_dir}")
else:
    print(f"Directory already exists: {data_dir}")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Image Counter
img_count = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # Mirror image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Draw landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show webcam
    cv2.imshow("Both Hands Sign Capture", img)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('s'):
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            # Collect all landmarks from both hands
            all_x = []
            all_y = []
            h, w, _ = img.shape
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    all_x.append(int(lm.x * w))
                    all_y.append(int(lm.y * h))

            # Bounding box around both hands
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)

            # Add padding
            padding = 20
            x_min = max(x_min - padding, 0)
            y_min = max(y_min - padding, 0)
            x_max = min(x_max + padding, w)
            y_max = min(y_max + padding, h)

            # Crop
            cropped_img = img[y_min:y_max, x_min:x_max]

            # Save cropped hand image
            img_count += 1
            filename = os.path.join(data_dir, f"Help{img_count}.jpg")
            cv2.imwrite(filename, cropped_img)
            print(f"Saved: {filename}")
        else:
            print("Both hands not detected, try again!")

    if key & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()