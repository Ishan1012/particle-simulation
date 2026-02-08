import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker:
    def __init__(self, model_path='hand_landmarker.task'):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.smoothed_pos = None
        self.alpha = 0.15
        self.timestamp = 0

    def get_hand_info(self, frame):
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        self.timestamp += 33
        detection_result = self.detector.detect_for_video(mp_image, self.timestamp)

        if detection_result.hand_landmarks:
            landmarks = detection_result.hand_landmarks[0]
            palm_idx = [0, 5, 17]
            palm_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in palm_idx])
            
            raw_cx = np.mean(palm_pts[:, 0])
            raw_cy = np.mean(palm_pts[:, 1])

            if self.smoothed_pos is None:
                self.smoothed_pos = np.array([raw_cx, raw_cy], dtype=float)
            else:
                self.smoothed_pos = (self.alpha * np.array([raw_cx, raw_cy])) + ((1 - self.alpha) * self.smoothed_pos)
            
            final_pos = self.smoothed_pos.astype(int)
            
            fingertips = [8, 12, 16, 20]
            knuckles = [6, 10, 14, 18]
            is_fist = all(landmarks[tip].y > landmarks[knuck].y for tip, knuck in zip(fingertips, knuckles))
            gesture = "Fist" if is_fist else "Open Palm"
            
            for t_idx in [4, 8, 12, 16, 20]:
                tx, ty = int(landmarks[t_idx].x * w), int(landmarks[t_idx].y * h)
                cv2.line(frame, (final_pos[0], final_pos[1]), (tx, ty), (0, 255, 0), 2)
            
            return final_pos, gesture, frame
            
        self.smoothed_pos = None
        return None, None, frame