import streamlit as st
import cv2
import numpy as np
from hand_tracker import HandTracker
from particle import ParticleManager

st.set_page_config(page_title="Hand Particle Sim", layout="wide")

def main():
    st.title("Interactive Hand Particle Simulation")
    
    st.sidebar.header("Settings")
    spawn_rate = st.sidebar.slider("Spawn Rate", 1, 30, 15)
    bg_gray = st.sidebar.slider("Background", 0, 255, 20)
    
    if 'hand_tracker' not in st.session_state:
        st.session_state.hand_tracker = HandTracker()
    if 'particle_manager' not in st.session_state:
        st.session_state.particle_manager = ParticleManager()
    if 'run' not in st.session_state:
        st.session_state.run = True

    def toggle():
        st.session_state.run = not st.session_state.run

    st.sidebar.button("Stop" if st.session_state.run else "Start", on_click=toggle)
    view = st.empty()

    if st.session_state.run:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while cap.isOpened() and st.session_state.run:
            ret, frame = cap.read()
            if not ret: break

            hand_pos, gesture, frame_proc = st.session_state.hand_tracker.get_hand_info(frame)
            canvas = np.full(frame.shape, (bg_gray, bg_gray, bg_gray), dtype=np.uint8)

            if hand_pos is not None:
                st.session_state.particle_manager.spawn(hand_pos[0], hand_pos[1], count=spawn_rate)
            
            st.session_state.particle_manager.update_and_draw(canvas, hand_pos, gesture)

            ph, pw = frame.shape[0] // 4, frame.shape[1] // 4
            pip = cv2.resize(frame_proc, (pw, ph))
            cv2.rectangle(pip, (0,0), (pw-1, ph-1), (255,255,255), 2)
            
            y, x = canvas.shape[0] - ph - 20, canvas.shape[1] - pw - 20
            canvas[y:y+ph, x:x+pw] = pip

            view.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), use_container_width=True)

        cap.release()
    else:
        view.info("Paused.")

if __name__ == "__main__":
    main()