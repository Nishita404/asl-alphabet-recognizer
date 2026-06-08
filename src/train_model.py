import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import json

# --- Paths ---
CSV_PATH   = "data/landmarks.csv"
MODEL_PATH = "model/asl_model.keras"
LABELS_PATH = "model/label_classes.json"

os.makedirs("model", exist_ok=True)

# --- Load Data ---
print("Loading landmarks.csv...")
df = pd.read_csv(CSV_PATH)

# Separate features (63 landmark columns) and label
X = df.drop("label", axis=1).values   # shape: (N, 63)
y = df["label"].values                 # shape: (N,)

print(f"Total samples: {len(X)}")
print(f"Classes: {sorted(set(y))}")

# --- Encode Labels ---
# Convert letters like "A", "B" into numbers like 0, 1, 2...
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Save label classes so we can decode predictions later
label_classes = encoder.classes_.tolist()
with open(LABELS_PATH, "w") as f:
    json.dump(label_classes, f)
print(f"Saved label classes to {LABELS_PATH}")

# --- Train/Test Split ---
# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# --- Build Model ---
num_classes = len(label_classes)

model = keras.Sequential([
    layers.Input(shape=(63,)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),               # prevents overfitting
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation="softmax")  # output: probability per class
])

model.summary()

# --- Compile ---
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# --- Train ---
print("\nTraining...")
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,   # 10% of training data for validation
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,              # stop if no improvement for 5 epochs
            restore_best_weights=True
        )
    ]
)

# --- Evaluate ---
print("\nEvaluating on test set...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# --- Save Model ---
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")