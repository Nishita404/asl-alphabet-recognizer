import os
import csv
import cv2
import mediapipe as mp

# --- Setup MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,      # we're processing images, not video
    max_num_hands=1,             # only detect one hand per image
    min_detection_confidence=0.3 # lower = catches more hands (good for dataset)
)

# --- Paths ---
DATASET_DIR = "asl-alphabet/asl_alphabet_train/asl_alphabet_train"
OUTPUT_CSV  = "data/landmarks.csv"

os.makedirs("data", exist_ok=True)

# --- CSV Header ---
# 21 landmarks × 3 coords (x, y, z) = 63 columns + 1 label column
header = []
for i in range(21):
    header += [f"x{i}", f"y{i}", f"z{i}"]
header.append("label")

# --- Extract Landmarks ---
with open(OUTPUT_CSV, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(header)

    classes = sorted(os.listdir(DATASET_DIR))  # A, B, C ... Z, space, del, nothing

    for label in classes:
        label_dir = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(label_dir):
            continue

        images = os.listdir(label_dir)
        success_count = 0

        for img_file in images:
            img_path = os.path.join(label_dir, img_file)
            image = cv2.imread(img_path)

            if image is None:
                continue

            # MediaPipe expects RGB, OpenCV gives BGR
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]  # take first hand

                row = []
                for lm in hand.landmark:  # 21 landmarks
                    row += [lm.x, lm.y, lm.z]
                row.append(label)

                writer.writerow(row)
                success_count += 1

        print(f"[{label}] {success_count}/{len(images)} images processed")

hands.close()
print(f"\nDone! Saved to {OUTPUT_CSV}")