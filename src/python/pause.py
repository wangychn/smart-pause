import math
import time
from queue import Queue
from typing import Optional, Sequence, List, Union  # (kept if you use them)
import sys
import cv2 as cv
import numpy as np
from PIL import Image
from pynput.keyboard import Key, Controller

from fdlite import FaceDetectionMXA, FaceLandmarkMXA
from fdlite.render import Colors, landmarks_to_render_data, render_to_image
from memryx import AsyncAccl

import threading
import signal
import os, traceback

stop_event = threading.Event()
accl_ref = None  # so we can stop it later if API allows

# --- Part 1: The "Eyes" - Head Pose Estimation ---
def get_head_pose(landmarks, frame_shape):
    """
    Calculates the head's orientation (Pitch, Yaw, Roll) from facial landmarks.
    We only need the Yaw for this project.
    """
    face_3d_model = np.array([
        [0.0, 0.0, 0.0],        # Nose tip
        [0.0, -330.0, -65.0],   # Chin
        [-225.0, 170.0, -135.0],  # Left eye left corner
        [225.0, 170.0, -135.0],   # Right eye right corner
        [-150.0, -150.0, -125.0],  # Left Mouth corner
        [150.0, -150.0, -125.0]    # Right mouth corner
    ], dtype=np.float64)

    # These landmark indices are specific to the MediaPipe Face Mesh model
    # FIX: Extract .x and .y coordinates from each Landmark object
    face_2d_points = np.array([
        (landmarks[1].x, landmarks[1].y),     # Nose tip
        (landmarks[152].x, landmarks[152].y), # Chin
        (landmarks[263].x, landmarks[263].y), # Left eye left corner
        (landmarks[33].x, landmarks[33].y),   # Right eye right corner
        (landmarks[287].x, landmarks[287].y), # Left Mouth corner
        (landmarks[57].x, landmarks[57].y)    # Right mouth corner
    ], dtype=np.float64)

    focal_length = frame_shape[1]
    center = (frame_shape[1] / 2, frame_shape[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))
    (_, rotation_vector, _) = cv.solvePnP(
        face_3d_model,
        face_2d_points,
        camera_matrix,
        dist_coeffs,
        flags=cv.SOLVEPNP_ITERATIVE
    )

    rotation_matrix, _ = cv.Rodrigues(rotation_vector)
    # sy not used for yaw here, but left as a reference to full decomposition
    # sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    yaw = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

    # Convert yaw to degrees
    yaw_degrees = yaw * (180.0 / np.pi)
    return yaw_degrees


# --- Pipeline Component Classes ---
class FaceApp(FaceDetectionMXA):
    def __init__(self, cam):
        super().__init__()
        self.cam = cam
        self.input_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.input_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
        self.capture_queue = Queue(maxsize=4)

    def generate_frame(self):
        while True:
            ok, frame = self.cam.read()
            if not ok:
                return None
            if not self.capture_queue.full():
                out, padding = self._preprocess(frame)
                self.capture_queue.put((frame, padding))
                return out

    def process_face(self, *ofmaps):
        (original_frame, padding) = self.capture_queue.get()
        dets, face_roi = self._postprocess(ofmaps, padding, (self.input_width, self.input_height))
 
        # Pass the original frame along with the ROI
        return (original_frame, face_roi) if dets else None


class LandmarkApp(FaceLandmarkMXA):
    def __init__(self, cam_size):
        super().__init__()
        self.capture_queue = Queue(maxsize=4)
        self.canvas_size = cam_size

    def process_landmark(self, *ofmaps):
        (landmark_frame, padding, roi) = self.capture_queue.get()
        landmarks = self._postprocess(ofmaps, padding, self.canvas_size, roi)
        # Pass the original frame along with the landmarks
        return landmark_frame, landmarks


# --- Main Application Controller ---
class App:
    def __init__(self, cam, center_yaw):
        self.face_app = FaceApp(cam)
        cam_size = (
            int(cam.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
        )
        self.landmark_app = LandmarkApp(cam_size=cam_size)

        self.video_state = 'PLAYING'
        self.last_seen_paying_attention = time.time()
        self.YAW_THRESHOLD = 20 
        self.yaw_lower_bound = center_yaw - self.YAW_THRESHOLD
        self.yaw_upper_bound = center_yaw + self.YAW_THRESHOLD
        print("lower: ", self.yaw_lower_bound)  
        print("upper: ", self.yaw_upper_bound)
        self.ATTENTION_GRACE_PERIOD = 2.0
        self.keyboard = Controller()
        self.PLAYBACK_KEY = Key.space
        self.start_time = time.time()
        self.WARMUP_PERIOD = 3.0

    def generate_frame_face(self):
        return self.face_app.generate_frame()

    def process_face(self, *ofmaps):
        # This callback now only decides what the next step is.
        result = self.face_app.process_face(*ofmaps)
          
        if result:
            original_frame, face_roi = result
            # If a face is found, queue it for the landmark model
            self.landmark_app.capture_queue.put((original_frame, None, face_roi))
            return face_roi
        else:
            # If NO face, call the final controller directly with no landmark data
            # Note: no original_frame available here; skip UI until next frame
            self.control_playback(self.yaw_lower_bound-100)          
            return None

    def generate_frame_landmark(self):
        # This is only called by the pipeline if process_face returns a valid ROI
        frame, _, roi = self.landmark_app.capture_queue.get()
        landmark_input, landmark_pad = self.landmark_app._preprocess(frame, roi)
        # Put back for the landmark output stage
        self.landmark_app.capture_queue.put((frame, landmark_pad, roi))
        return landmark_input

    def process_landmark(self, *ofmaps):
        # This is only called if the landmark model successfully runs
        original_frame, landmarks = self.landmark_app.process_landmark(*ofmaps)
        # Call the final controller with the landmark data
        self.final_controller(original_frame, landmarks)

    def final_controller(self, frame, landmarks):
        """
        This is the single, unified function for all UI and keyboard actions.
        """
        yaw = 999
        if landmarks is not None and frame is not None:
            yaw = get_head_pose(landmarks, frame.shape)

        # 1. Control Playback Logic
        is_warming_up = (time.time() - self.start_time) < self.WARMUP_PERIOD
        if not is_warming_up:
            self.control_playback(yaw)

        # 2. Display Logic
        # Note: supply a connection color to landmarks_to_render_data
        render_data = landmarks_to_render_data(
            landmarks if landmarks else [],
            [],  # no connections (points only); or pass your FACE_LANDMARK_CONNECTIONS here
            landmark_color=Colors.PINK,
            connection_color=Colors.GREEN,
            thickness=3
        )
        output_image = render_to_image(render_data, Image.fromarray(frame))
        cv_image = np.array(output_image)      
        cv.putText(cv_image, f"Yaw: {round(yaw, 2)}", (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.imshow('Smart Pauser', cv_image)
        k = cv.waitKey(1) & 0xff  # Use waitKey(1) for smoothness
        if k == 27:
            cv.destroyAllWindows()
            exit(0)

    def control_playback(self, yaw):
        current_time = time.time()
        is_paying_attention = yaw < self.yaw_upper_bound and yaw > self.yaw_lower_bound
        # print(is_paying_attention)
        if is_paying_attention:
            self.last_seen_paying_attention = current_time
            if self.video_state == 'PAUSED':
                print("Resuming video...")
                self.keyboard.press(self.PLAYBACK_KEY)
                self.keyboard.release(self.PLAYBACK_KEY)
                self.video_state = 'PLAYING'
        else:
            if self.video_state == 'PLAYING' and \
               (current_time - self.last_seen_paying_attention) > self.ATTENTION_GRACE_PERIOD:
                print(f"Pausing video (Yaw: {round(yaw, 2)})...")
                self.keyboard.press(self.PLAYBACK_KEY)
                self.keyboard.release(self.PLAYBACK_KEY)
                self.video_state = 'PAUSED'

    
def run_mxa(dfp):
    global accl_ref
    try:
        print(f"[run_mxa] loading DFP: {dfp}  exists={os.path.exists(dfp)}")
        accl = AsyncAccl(dfp)
        accl_ref = accl

        # connect IO
        accl.connect_input(app.generate_frame_face)              # input 0
        accl.connect_input(app.generate_frame_landmark, 1)       # input 1
        accl.connect_output(app.process_face)                    # output 0
        accl.connect_output(app.process_landmark, 1)             # output 1

        print("[run_mxa] entering wait()")
        accl.wait()  # blocks inside the worker thread
        print("[run_mxa] wait() returned (pipeline ended)")

    except Exception as e:
        print("[run_mxa] EXCEPTION:", e)
        traceback.print_exc()
    finally:
        print("[run_mxa] cleanup")
        cv.destroyAllWindows()

def start_pipeline(dfp_path):
    t = threading.Thread(target=run_mxa, args=(dfp_path,), daemon=True)
    t.start()
    return t

def stop_pipeline():
    # best-effort shutdown; use the right API if Memryx exposes one
    # e.g., accl_ref.stop() / accl_ref.close() / accl_ref.shutdown()
    # fallback: release camera & exit process
    try:
        if accl_ref:
            pass  # call the real stop if available
    finally:
        cv.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Error: Calibrated center yaw value must be provided.", file=sys.stderr)
        exit(1)
    
    try:
        calibrated_yaw = float(sys.argv[1])
    except ValueError:
        print(f"Error: Invalid yaw value provided '{sys.argv[1]}'. Must be a number.", file=sys.stderr)
        exit(1)
    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Cannot open camera.")
        exit()

    app = App(cam, calibrated_yaw )
    dfp_path = "models/models.dfp"

    print("Starting Smart Pauser. Press 'ESC' in the display window to quit.")
    print("Switching windows momentarily...")
    time.sleep(1)
  
    print("Starting Smart Pauser…")
    thread = start_pipeline(dfp_path)

    try:
        while thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_pipeline()

    cam.release()
    cv.destroyAllWindows()
