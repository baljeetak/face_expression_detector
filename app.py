import os
import time
# CRITICAL: Fix for TensorFlow 2.20.0 / Keras 3 compatibility
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
from deepface import DeepFace
import numpy as np
from datetime import datetime
import av

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Emotion Recorder", layout="wide")

SAVE_DIR = "recordings"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

st.title("Face Expression Detector & Recorder 🎥")

# --- 2. VIDEO PROCESSING CLASS (20s Limit) ---
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.out = None
        self.recording = False
        self.start_time = None
        self.limit = 20  # Automatic stop after 20 seconds

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Emotion Analysis
        try:
            results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
            for res in results:
                x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
                emotion = res['dominant_emotion']
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except:
            pass

        # Timer & Recording Logic
        if self.recording:
            if self.start_time is None:
                self.start_time = time.time()
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                path = os.path.join(SAVE_DIR, f"vid_{ts}.mp4")
                # Using 'mp4v' for internal server saving
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                h, w = img.shape[:2]
                self.out = cv2.VideoWriter(path, fourcc, 20.0, (w, h))

            elapsed = time.time() - self.start_time
            if elapsed >= self.limit:
                self.recording = False  # Auto-stop after 20 seconds
            else:
                # Add countdown text to the live video feed
                countdown = int(self.limit - elapsed)
                cv2.putText(img, f"REC: {countdown}s left", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self.out.write(img)
        else:
            if self.out is not None:
                self.out.release()
                self.out = None
                self.start_time = None

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 3. MAIN INTERFACE ---
ctx = webrtc_streamer(
    key="emotion-recorder",
    video_transformer_factory=VideoTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

if ctx.video_transformer:
    if st.button("🎬 START 20s RECORDING", use_container_width=True, type="primary"):
        ctx.video_transformer.recording = True
        
    if ctx.video_transformer.recording:
        st.error("🔴 RECORDING IN PROGRESS... (Auto-stops at 20s)")
    else:
        st.info("⚪ STANDBY - Click Start to record a new session.")

# --- 4. SECRET SIDEBAR (The "Window" into the Server) ---
st.sidebar.title("📁 Server Recordings")
st.sidebar.write("Videos saved on the cloud server:")

if os.path.exists(SAVE_DIR):
    files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".mp4")]
    
    if files:
        selected_video = st.sidebar.selectbox("Select a recording:", files)
        
        if st.sidebar.button("▶️ Play Selected"):
            video_path = os.path.join(SAVE_DIR, selected_video)
            with open(video_path, 'rb') as f:
                st.sidebar.video(f.read())
                
        if st.sidebar.button("🗑️ Delete All (Clear Server)"):
            for f in files:
                os.remove(os.path.join(SAVE_DIR, f))
            st.rerun()
    else:
        st.sidebar.write("No videos found yet.")
