import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from hand_tracker import HandTracker
from particle import ParticleManager

st.set_page_config(page_title="Hand Particle Sim", layout="wide")

class ParticleProcessor(VideoProcessorBase):
    def __init__(self, spawn_rate, bg_gray):
        self.hand_tracker = HandTracker()
        self.particle_manager = ParticleManager()
        self.spawn_rate = spawn_rate
        self.bg_gray = bg_gray
        self.zoom_factor = 1.5

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        canvas_h, canvas_w = int(h * self.zoom_factor), int(w * self.zoom_factor)
        
        hand_pos, gesture, frame_proc = self.hand_tracker.get_hand_info(img)
        canvas = np.full((canvas_h, canvas_w, 3), (self.bg_gray, self.bg_gray, self.bg_gray), dtype=np.uint8)

        offset_x = (canvas_w - w) // 2
        offset_y = (canvas_h - h) // 2

        if hand_pos is not None:
            adjusted_x = hand_pos[0] + offset_x
            adjusted_y = hand_pos[1] + offset_y
            self.particle_manager.spawn(adjusted_x, adjusted_y, count=self.spawn_rate)
            scaled_hand = [adjusted_x, adjusted_y]
        else:
            scaled_hand = None
        
        self.particle_manager.update_and_draw(canvas, scaled_hand, gesture)

        ph, pw = h // 4, w // 4
        pip = cv2.resize(frame_proc, (pw, ph))
        cv2.rectangle(pip, (0, 0), (pw - 1, ph - 1), (255, 255, 255), 2)
        
        canvas[canvas_h - ph - 20 : canvas_h - 20, canvas_w - pw - 20 : canvas_w - 20] = pip

        return av.VideoFrame.from_ndarray(canvas, format="bgr24")

def main():
    if 'run' not in st.session_state:
        st.session_state.run = True

    st.title("Interactive Hand Particle Simulation")
    
    st.sidebar.header("Settings")
    spawn_rate = st.sidebar.slider("Spawn Rate", 1, 30, 15)
    bg_gray = st.sidebar.slider("Background", 0, 255, 20)
    
    if st.session_state.run:
        webrtc_streamer(
            key="hand-particle-sim",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: ParticleProcessor(spawn_rate, bg_gray),
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True,
        )
    else:
        st.info("Simulation is currently stopped.")

if __name__ == "__main__":
    main()