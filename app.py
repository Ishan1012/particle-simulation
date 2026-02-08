import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from hand_tracker import HandTracker
from particle import ParticleManager

st.set_page_config(page_title="Hand Particle Sim", layout="wide")

class ParticleProcessor(VideoTransformerBase):
    def __init__(self, spawn_rate, bg_gray):
        self.hand_tracker = HandTracker()
        self.particle_manager = ParticleManager()
        self.spawn_rate = spawn_rate
        self.bg_gray = bg_gray

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        hand_pos, gesture, frame_proc = self.hand_tracker.get_hand_info(img)
        canvas = np.full(img.shape, (self.bg_gray, self.bg_gray, self.bg_gray), dtype=np.uint8)

        if hand_pos is not None:
            self.particle_manager.spawn(hand_pos[0], hand_pos[1], count=self.spawn_rate)
        
        self.particle_manager.update_and_draw(canvas, hand_pos, gesture)

        ph, pw = img.shape[0] // 4, img.shape[1] // 4
        pip = cv2.resize(frame_proc, (pw, ph))
        cv2.rectangle(pip, (0, 0), (pw - 1, ph - 1), (255, 255, 255), 2)
        
        y, x = canvas.shape[0] - ph - 20, canvas.shape[1] - pw - 20
        canvas[y:y+ph, x:x+pw] = pip

        return canvas

def main():
    st.title("Interactive Hand Particle Simulation")
    
    st.sidebar.header("Settings")
    spawn_rate = st.sidebar.slider("Spawn Rate", 1, 30, 15)
    bg_gray = st.sidebar.slider("Background", 0, 255, 20)
    
    webrtc_streamer(
        key="hand-particle-sim",
        mode="sendrecv",
        video_processor_factory=lambda: ParticleProcessor(spawn_rate, bg_gray),
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        video_html_attrs={
            "style": {"width": "100%"},
            "controls": False,
            "autoPlay": True,
        },
    )

if __name__ == "__main__":
    main()