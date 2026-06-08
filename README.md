# 🤟 ASL Alphabet Recognizer

Real-time American Sign Language (ASL) recognition using MediaPipe hand tracking, TensorFlow neural networks, OpenCV, and Streamlit.

🚀 **Live Demo:**  
https://asl-alphabet-recognizer-lk4cucarqwdtfacv8zsbs3.streamlit.app/


---

## 📌 Overview

ASL Alphabet Recognizer is a computer vision and machine learning project that identifies American Sign Language hand signs using hand landmark detection and a neural network classifier.

Instead of training directly on images, the system extracts 21 hand landmarks using MediaPipe and converts them into 63 numerical features (x, y, z coordinates). These features are then used to train a lightweight TensorFlow model capable of recognizing 29 ASL classes with over **98% test accuracy**.

This landmark-based approach significantly reduces computational requirements while maintaining high accuracy and real-time performance.

---

## ✨ Features

- Real-time ASL sign recognition
- MediaPipe hand landmark detection
- TensorFlow neural network classifier
- 98% test accuracy
- Lightweight landmark-based architecture
- Streamlit web deployment
- Supports 29 classes:
  - A–Z
  - Space
  - Delete
  - Nothing

---

## 🎯 Results

| Metric | Value |
|----------|----------|
| Test Accuracy | 98.07% |
| Classes Recognized | 29 |
| Hand Landmarks | 21 |
| Input Features | 63 |
| Framework | TensorFlow |
| Detection Engine | MediaPipe |
| Deployment | Streamlit Cloud |

---

## 🏗 System Architecture

```text
Dataset Images
       │
       ▼
MediaPipe Hand Detection
       │
       ▼
21 Hand Landmarks
       │
       ▼
63 Numerical Features
       │
       ▼
TensorFlow Neural Network
       │
       ▼
ASL Letter Prediction
       │
       ▼
Streamlit Web Application
```

---

## 🧠 Model Architecture

The classifier is trained using hand landmark coordinates rather than raw image pixels.

```text
Input Layer
63 Features
(21 Landmarks × 3 Coordinates)
        │
        ▼
Dense Layer (128)
        │
        ▼
Dense Layer (64)
        │
        ▼
Output Layer (29 Classes)
```

This design allows the model to remain lightweight while achieving high accuracy.

---

## 📂 Dataset

Dataset used:

**ASL Alphabet Dataset (Kaggle)**

https://www.kaggle.com/datasets/grassknoted/asl-alphabet

### Dataset Statistics

- 87,000+ Images
- 29 Classes
- ~3,000 Images Per Class

---

## 🛠 Tech Stack

### Machine Learning

- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-Learn

### Computer Vision

- MediaPipe
- OpenCV

### Frontend & Deployment

- Streamlit
- Streamlit Cloud

### Version Control

- Git
- GitHub

---

## 📁 Project Structure

```text
asl-alphabet-recognizer/
│
├── app.py
│
├── src/
│   ├── extract_landmarks.py
│   ├── train_model.py
│   └── realtime_inference.py
│
├── model/
│   └── label_classes.json
│
├── data/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/Nishita404/asl-alphabet-recognizer.git
cd asl-alphabet-recognizer
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running Locally

### Streamlit Application

```bash
streamlit run app.py
```

### Real-Time Webcam Inference

```bash
python src/realtime_inference.py
```

---

## 🔄 Project Workflow

### Phase 1 — Dataset Collection

Download the ASL Alphabet dataset from Kaggle.

### Phase 2 — Landmark Extraction

Convert hand images into landmark coordinates using MediaPipe.

Output:

```text
data/landmarks.csv
```

### Phase 3 — Model Training

Train a TensorFlow classifier on landmark coordinates.

Output:

```text
model/asl_model.keras
```

### Phase 4 — Real-Time Inference

Recognize ASL signs from webcam input.

### Phase 5 — Deployment

Deploy the application using Streamlit Cloud.

---

## 📈 Why Landmark-Based Learning?

Traditional image classification requires processing hundreds of thousands of pixel values per image.

This project reduces each image to:

```text
21 Hand Landmarks
×
3 Coordinates
=
63 Features
```

### Benefits

- Faster training
- Smaller model size
- Lower hardware requirements
- Better generalization
- Real-time performance
- Easy deployment

---

## 🔮 Future Improvements

- Word and sentence formation
- Continuous sign language recognition
- Text-to-speech conversion
- Multi-hand recognition
- Mobile application support
- Transformer-based gesture models
- Streamlit webcam integration

---

## 👨‍💻 Author

**Nishita**

GitHub: https://github.com/Nishita404

---

## ⭐ Support

If you found this project useful, consider giving it a star on GitHub.

It helps support future development and makes the project easier for others to discover.