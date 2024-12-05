import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
import webbrowser
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model and labels with error handling
try:
    model = load_model("model.h5")
    label = np.load("labels.npy")
except Exception as e:
    st.error(f"Error loading model or labels: {e}")
    st.stop()

# Initialize Mediapipe
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

st.header("Emotion Based Music Recommender")

# Initialize session state
if "run" not in st.session_state:
    st.session_state["run"] = True

# Load previous emotion with fallback
try:
    emotion = np.load("emotion.npy")[0]
except Exception:
    emotion = ""

if not emotion:
    st.session_state["run"] = True
else:
    st.session_state["run"] = False

class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        
        lst = []
        if res.face_landmarks:
            # Extract face landmarks relative to landmark 1
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)
            
            # Left hand landmarks
            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)
            
            # Right hand landmarks
            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)
            
            # Predict emotion
            lst = np.array(lst).reshape(1, -1)
            pred = label[np.argmax(model.predict(lst))]
            cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
            
            # Save detected emotion
            np.save("emotion.npy", np.array([pred]))
        
        # Draw landmarks
        if res.face_landmarks:
            drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                                   landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=1),
                                   connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        
        if res.left_hand_landmarks:
            drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        
        if res.right_hand_landmarks:
            drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)
        
        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# User inputs
lang = st.text_input("Language")
singer = st.text_input("Singer")

# WebRTC Streamer
if lang and singer and st.session_state["run"]:
    webrtc_streamer(key="emotion-capture", 
                    desired_playing_state=True,
                    video_processor_factory=EmotionProcessor)

# Song Recommendation Button
btn = st.button("Recommend me songs")
if btn:
    try:
        if not emotion:
            st.warning("Please let me capture your emotion first.")
            st.session_state["run"] = True
        else:
            with st.spinner("Generating song recommendations..."):
                search_query = f"{lang}+{emotion}+song+{singer}"
                webbrowser.open(f"https://www.youtube.com/results?search_query={search_query}")
                
                # Reset emotion
                np.save("emotion.npy", np.array([""]))
                st.session_state["run"] = False
    except Exception as e:
        st.error(f"Error in recommendation: {e}")
        logger.error(f"Recommendation error: {e}")
