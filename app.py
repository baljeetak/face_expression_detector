import os
import time
# Fix for TensorFlow 2.20.0
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
from deepface import DeepFace
import numpy as np
from datetime import datetime
import av

# --- CONFIG ---
SAVE_DIR = "recordings"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

st.title("Cloud Expression Recorder (20s Limit) 🎥")

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.out = None
        self.recording = False
        self.start_time = None
        self.limit = 20  # Seconds

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # 1. Emotion Analysis
        try:
            results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
            for res in results:
                x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
                emotion = res['dominant_emotion']
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except:
            pass

        # 2. Timer & Recording Logic
        if self.recording:
            if self.start_time is None:
                self.start_time = time.time()
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                path = os.path.join(SAVE_DIR, f"auto_vid_{ts}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                h, w = img.shape[:2]
                self.out = cv2.VideoWriter(path, fourcc, 20.0, (w, h))

            # Check if 20 seconds have passed
            elapsed = time.time() - self.start_time
            if elapsed >= self.limit:
                self.recording = False  # Auto-stop
            else:
                # Add a "Seconds Left" countdown on the video itself
                countdown = int(self.limit - elapsed)
                cv2.putText(img, f"Ends in: {countdown}s", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self.out.write(img)

        else:
            if self.out is not None:
                self.out.release()
                self.out = None
                self.start_time = None

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI ---
ctx = webrtc_streamer(
    key="emotion-recorder",
    video_transformer_factory=VideoTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

if ctx.video_transformer:
    if st.button("🎬 START 20s RECORDING"):
        ctx.video_transformer.recording = True
        
    if ctx.video_transformer.recording:
        st.error("🔴 RECORDING... (Will stop automatically)")
    else:
        st.info("⚪ STANDBY - Click Start to record for 20 seconds.")