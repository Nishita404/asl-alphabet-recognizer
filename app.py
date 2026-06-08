import streamlit as st
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

st.set_page_config(
    page_title="ASL Recognizer",
    page_icon="🤟",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;700;900&family=Instrument+Serif:ital@0;1&display=swap');

* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main, .block-container {
    background-color: #F0E8D8 !important;
}
[data-testid="stHeader"] { background-color: #F0E8D8 !important; }
[data-testid="stToolbar"] { display: none; }
#MainMenu, footer { visibility: hidden; }

.block-container {
    padding-top: 2rem !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
    max-width: 100% !important;
}

[data-testid="stMain"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(10,10,10,0.07) 1px, transparent 1px),
        linear-gradient(90deg, rgba(10,10,10,0.07) 1px, transparent 1px);
    background-size: 80px 80px;
    pointer-events: none;
    z-index: 0;
}

.hero-wrap {
    padding: 3rem 0 2.5rem;
    border-bottom: 2.5px solid #0D0D0D;
    margin-bottom: 0;
}
.hero-eyebrow {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #0D0D0D !important;
    opacity: 0.45;
    margin-bottom: 0.75rem;
}
.hero-title {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: clamp(3.5rem, 9vw, 8rem);
    font-weight: 900;
    line-height: 0.88;
    text-transform: uppercase;
    color: #0D0D0D !important;
    letter-spacing: -0.01em;
    margin: 0;
}
.hero-title .acc {
    font-family: 'Instrument Serif', serif !important;
    font-style: italic;
    color: #0D0D0D !important;
    background: #F5C400;
    padding: 0 6px;
}
.hero-sub {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.85rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #0D0D0D !important;
    opacity: 0.5;
    margin-top: 1.75rem;
}

.stats-row {
    display: flex;
    border-bottom: 2.5px solid #0D0D0D;
    margin-bottom: 2.5rem;
}
.stat-box {
    flex: 1;
    padding: 1.25rem 1.75rem;
    border-right: 1px solid rgba(10,10,10,0.18);
}
.stat-box:last-child { border-right: none; }
.stat-num {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 2.6rem;
    font-weight: 900;
    line-height: 1;
    color: #0D0D0D !important;
}
.stat-num .acc { color: #F5C400; -webkit-text-stroke: 1.5px #0D0D0D; }
.stat-lbl {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #0D0D0D !important;
    opacity: 0.45;
    margin-top: 0.2rem;
}

.sec-label {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #0D0D0D !important;
    opacity: 0.4;
    margin-bottom: 0.75rem;
    padding-top: 1.5rem;
}
.cam-label {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #F5C400 !important;
    background: #0D0D0D;
    display: inline-block;
    padding: 0.3rem 0.8rem;
    margin-bottom: 0.75rem;
}

.tips-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0;
    border: 2px solid #0D0D0D;
    margin-top: 1.5rem;
    margin-bottom: 2.5rem;
    background: #0D0D0D;
}
.tip-cell {
    background: #F0E8D8;
    padding: 1.2rem 1.5rem;
    border-right: 1px solid #0D0D0D;
}
.tip-cell:last-child { border-right: none; }
.tip-icon { font-size: 1.4rem; margin-bottom: 0.4rem; }
.tip-txt {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #0D0D0D !important;
    opacity: 0.65;
}

.asl-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
    gap: 1px;
    background: #0D0D0D;
    border: 2px solid #0D0D0D;
    margin-bottom: 2.5rem;
}
.asl-cell {
    background: #F0E8D8;
    padding: 1rem 0.5rem;
    text-align: center;
}
.asl-letter {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1.8rem;
    font-weight: 900;
    color: #0D0D0D !important;
    line-height: 1;
}
.asl-desc {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #0D0D0D !important;
    opacity: 0.4;
    margin-top: 0.2rem;
}

.footer-bar {
    border-top: 2.5px solid #0D0D0D;
    padding: 1.25rem 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 2rem;
}
.footer-l {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #0D0D0D !important;
    opacity: 0.4;
}
.footer-badge {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.72rem;
    font-weight: 900;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #0D0D0D !important;
    background: #F5C400;
    border: 2px solid #0D0D0D;
    padding: 0.35rem 1rem;
    text-decoration: none !important;
}
[data-testid="stNotification"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# --- Load Model & Labels ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/asl_model.keras")
    with open("model/label_classes.json", "r") as f:
        label_classes = json.load(f)
    return model, label_classes

model, label_classes = load_model()

# --- MediaPipe ---
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

# --- Transformer ---
class ASLTransformer(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.frame_count = 0       # counts frames
        self.last_label  = ""      # cache last prediction
        self.last_conf   = 0.0     # cache last confidence
        self.PREDICT_EVERY = 3     # only predict every 3rd frame

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        # --- Resize to 640x480 for speed ---
        img = cv2.resize(img, (640, 480))
        h, w, _ = img.shape

        rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        self.frame_count += 1

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            # Draw landmarks
            mp_draw.draw_landmarks(
                img, hand, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(245,196,0), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(255,255,255), thickness=2)
            )

            # Only run model every Nth frame
            if self.frame_count % self.PREDICT_EVERY == 0:
                row = []
                for lm in hand.landmark:
                    row += [lm.x, lm.y, lm.z]

                input_arr      = np.array([row], dtype=np.float32)
                preds          = model(input_arr, training=False).numpy()  # faster than model.predict
                idx            = int(np.argmax(preds))
                self.last_conf  = float(np.max(preds))
                self.last_label = label_classes[idx]

            # Draw cached prediction
            if self.last_label:
                cv2.rectangle(img, (0, h-95), (w, h), (13,13,13), -1)
                cv2.putText(img, self.last_label,
                    (18, h-22), cv2.FONT_HERSHEY_DUPLEX,
                    2.4, (245,196,0), 3)
                cv2.putText(img, f"{self.last_conf*100:.0f}%  confidence",
                    (18, h-6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (200,200,200), 1)
        else:
            self.last_label = ""
            self.last_conf  = 0.0
            cv2.rectangle(img, (0, h-50), (w, h), (13,13,13), -1)
            cv2.putText(img, "Show your hand to the camera",
                (18, h-16), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (130,130,130), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- RTC Config ---
RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# ═══════ UI ═══════

st.markdown("""
<div class="hero-wrap">
  <div class="hero-eyebrow">Computer Vision · Real-Time Detection · Python</div>
  <div class="hero-title">
    ASL<br><span class="acc">Alpha</span>BET<br>RECOGNIZER
  </div>
  <div class="hero-sub">MediaPipe &nbsp;·&nbsp; TensorFlow &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; 29 Signs</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stats-row">
  <div class="stat-box"><div class="stat-num">98<span class="acc">%</span></div><div class="stat-lbl">Test Accuracy</div></div>
  <div class="stat-box"><div class="stat-num">29</div><div class="stat-lbl">Sign Classes</div></div>
  <div class="stat-box"><div class="stat-num">21</div><div class="stat-lbl">Hand Landmarks</div></div>
  <div class="stat-box"><div class="stat-num">63</div><div class="stat-lbl">Input Features</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="sec-label">— Live Webcam</div>', unsafe_allow_html=True)
st.markdown('<div class="cam-label">▶ Press START to activate camera</div>', unsafe_allow_html=True)

webrtc_streamer(
    key="asl-recognizer",
    video_transformer_factory=ASLTransformer,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={
        "video": {"width": 640, "height": 480, "frameRate": 30},
        "audio": False
    },
    async_transform=False
)

st.markdown("""
<div class="tips-grid">
  <div class="tip-cell"><div class="tip-icon">💡</div><div class="tip-txt">Good lighting on your hand</div></div>
  <div class="tip-cell"><div class="tip-icon">✋</div><div class="tip-txt">Keep hand fully in frame</div></div>
  <div class="tip-cell"><div class="tip-icon">🎯</div><div class="tip-txt">Plain background works best</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="sec-label">— ASL Alphabet Reference</div>', unsafe_allow_html=True)

asl_hints = {
    "A": "Fist, thumb side", "B": "Flat hand up",
    "C": "Curved hand", "D": "Index up, curl",
    "E": "Fingers bent", "F": "OK shape",
    "G": "Index points side", "H": "Two fingers side",
    "I": "Pinky up", "J": "Pinky + draw J",
    "K": "Index + middle up", "L": "L shape",
    "M": "Three fingers down", "N": "Two fingers down",
    "O": "O shape", "P": "K pointing down",
    "Q": "G pointing down", "R": "Fingers crossed",
    "S": "Fist, thumb front", "T": "Thumb in fist",
    "U": "Two fingers together", "V": "Peace sign",
    "W": "Three fingers up", "X": "Index hook",
    "Y": "Hang loose", "Z": "Draw Z",
    "del": "↩ gesture", "nothing": "No sign", "space": "Flat open"
}

cells = ""
for letter, hint in asl_hints.items():
    cells += f"""
    <div class="asl-cell">
        <div class="asl-letter">{letter.upper()}</div>
        <div class="asl-desc">{hint}</div>
    </div>"""

st.markdown(f'<div class="asl-grid">{cells}</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer-bar">
  <div class="footer-l">Built by Nishita · 2025</div>
  <a class="footer-badge" href="https://github.com/Nishita404/asl-alphabet-recognizer" target="_blank">GitHub ↗</a>
</div>
""", unsafe_allow_html=True)