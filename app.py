import streamlit as st
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# --- Page Config ---
st.set_page_config(
    page_title="ASL Alphabet Recognizer",
    page_icon="🤟",
    layout="centered"
)

# --- Load Model & Labels ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/asl_model.keras")
    with open("model/label_classes.json", "r") as f:
        label_classes = json.load(f)
    return model, label_classes

model, label_classes = load_model()

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

# --- Video Transformer ---
class ASLTransformer(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.label    = ""
        self.confidence = 0.0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

            row = []
            for lm in hand.landmark:
                row += [lm.x, lm.y, lm.z]

            input_data  = np.array([row])
            predictions = model.predict(input_data, verbose=0)
            idx         = np.argmax(predictions)
            self.confidence = float(np.max(predictions))
            self.label  = label_classes[idx]

            # Draw prediction on frame
            cv2.rectangle(img, (0, 0), (280, 90), (0, 0, 0), -1)
            cv2.putText(img, f"Sign: {self.label}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 255, 0), 3)
            cv2.putText(img, f"Confidence: {self.confidence*100:.1f}%",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
        else:
            cv2.rectangle(img, (0, 0), (280, 60), (0, 0, 0), -1)
            cv2.putText(img, "No hand detected",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI ---
st.title("🤟 ASL Alphabet Recognizer")
st.markdown("Real-time American Sign Language detection using **MediaPipe** + **TensorFlow**")

st.divider()

col1, col2, col3 = st.columns(3)
col1.metric("Model Accuracy", "98.07%")
col2.metric("Classes", "29")
col3.metric("Landmarks", "21 points")

st.divider()

st.subheader("📷 Live Webcam")
st.caption("Allow camera access, then show your hand sign to the webcam.")

ctx = webrtc_streamer(
    key="asl-recognizer",
    video_transformer_factory=ASLTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True
)

st.divider()

st.subheader("📖 ASL Alphabet Reference")
st.image(
    "https://www.lifeprint.com/asl101/fingerspelling/abc-800.jpg",
    caption="ASL Alphabet Chart",
    use_column_width=True
)

st.divider()
st.markdown(
    "Built with MediaPipe · TensorFlow · Streamlit &nbsp;|&nbsp; "
    "[GitHub](https://github.com/Nishita404/asl-alphabet-recognizer)",
    unsafe_allow_html=True
)