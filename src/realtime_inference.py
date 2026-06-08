import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json

# --- Load Model & Labels ---
model = tf.keras.models.load_model("model/asl_model.keras")

with open("model/label_classes.json", "r") as f:
    label_classes = json.load(f)

# --- Setup MediaPipe ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,      # video mode (not static images)
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# --- Start Webcam ---
cap = cv2.VideoCapture(0)
print("Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally (mirror effect — feels natural)
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    predicted_label = ""
    confidence = 0.0

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw hand landmarks on screen
        mp_draw.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
        )

        # Extract 63 landmark values
        row = []
        for lm in hand_landmarks.landmark:
            row += [lm.x, lm.y, lm.z]

        # Predict
        input_data = np.array([row])                        # shape: (1, 63)
        predictions = model.predict(input_data, verbose=0)  # shape: (1, 29)
        class_index = np.argmax(predictions)
        confidence = float(np.max(predictions))
        predicted_label = label_classes[class_index]

    # --- Display on Screen ---
    # Background box for text
    cv2.rectangle(frame, (0, 0), (300, 100), (0, 0, 0), -1)

    if predicted_label:
        # Show predicted letter
        cv2.putText(
            frame, f"Sign: {predicted_label}",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
            1.5, (0, 255, 0), 3
        )
        # Show confidence percentage
        cv2.putText(
            frame, f"Confidence: {confidence * 100:.1f}%",
            (10, 85), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )
    else:
        cv2.putText(
            frame, "No hand detected",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 0, 255), 2
        )

    # Show frame
    cv2.imshow("ASL Alphabet Recognizer", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
hands.close()
print("Webcam closed.")