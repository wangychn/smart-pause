import math
import time
from queue import Queue
from typing import Optional, Sequence, List, Union
import sys
import cv2 as cv
import numpy as np
from PIL import Image
from pynput.keyboard import Key, Controller

from fdlite import FaceDetectionMXA, FaceLandmarkMXA
from fdlite.render import Colors, landmarks_to_render_data, render_to_image
from memryx import AsyncAccl

import threading
import traceback
import socket
import requests

# --- Global Configuration ---
stop_event = threading.Event()
accl_ref = None

SERVER_A_IP = '127.0.0.1'
SOCKET_A_PORT = 65432
SERVER_B_URL = 'https://sicklily-legible-marline.ngrok-free.dev/get_data' # <-- IMPORTANT: Replace with your actual ngrok URL

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
    yaw = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    return yaw * (180.0 / np.pi)


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
        return (original_frame, face_roi) if dets else None


class LandmarkApp(FaceLandmarkMXA):
    def __init__(self, cam_size):
        super().__init__()
        self.capture_queue = Queue(maxsize=4)
        self.canvas_size = cam_size

    def process_landmark(self, *ofmaps):
        (landmark_frame, padding, roi) = self.capture_queue.get()
        landmarks = self._postprocess(ofmaps, padding, self.canvas_size, roi)
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
        self.ATTENTION_GRACE_PERIOD = 2.0
        self.keyboard = Controller()
        self.PLAYBACK_KEY = Key.space
        self.start_time = time.time()
        self.WARMUP_PERIOD = 3.0
        self.time_geeked = 0.0
        self.time_locked = 0.0
        self.num_geeked = 0
        self.max_geek = 0.0
        self.pause_start = time.time()
        
        # Connect to Server A using the original socket method
        self.socket_server_a = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_server_a.connect((SERVER_A_IP, SOCKET_A_PORT))
        print("Connected to Server A (socket)")

        # Set up for the background thread to handle Server B
        self.received_number = 0
        self.update_thread = threading.Thread(target=self._update_data_from_server_b, daemon=True)
        self.update_thread.start()
        print("Started background thread to fetch data from Server B.")

    def _update_data_from_server_b(self):
        """
        This function runs in a separate thread and continuously fetches data
        from Server B every 2 seconds.
        """
        while not stop_event.is_set():
            try:
                response = requests.get(SERVER_B_URL, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    self.received_number = data.get('number')
                else:
                    self.received_number = 0
            except requests.exceptions.RequestException:
                self.received_number = 0
            
            time.sleep(2)

    def send_data(self, data):
        """Helper method to send data over the socket to Server A."""
        try:
            self.socket_server_a.sendall(data.encode('utf-8'))
        except Exception as e:
            print(f"Error sending data to Server A: {e}")

    def generate_frame_face(self):
        return self.face_app.generate_frame()

    def process_face(self, *ofmaps):
        result = self.face_app.process_face(*ofmaps)
        if result:
            original_frame, face_roi = result
            self.landmark_app.capture_queue.put((original_frame, None, face_roi))
            return face_roi
        else:
            self.control_playback(self.yaw_lower_bound - 100)
            return None

    def generate_frame_landmark(self):
        frame, _, roi = self.landmark_app.capture_queue.get()
        landmark_input, landmark_pad = self.landmark_app._preprocess(frame, roi)
        self.landmark_app.capture_queue.put((frame, landmark_pad, roi))
        return landmark_input

    def process_landmark(self, *ofmaps):
        original_frame, landmarks = self.landmark_app.process_landmark(*ofmaps)
        self.final_controller(original_frame, landmarks)

    def final_controller(self, frame, landmarks):
        yaw = 999
        if landmarks is not None and frame is not None:
            yaw = get_head_pose(landmarks, frame.shape)
        
        is_warming_up = (time.time() - self.start_time) < self.WARMUP_PERIOD
        if not is_warming_up:
            self.control_playback(yaw)
        
        render_data = landmarks_to_render_data(
            landmarks if landmarks else [], [], landmark_color=Colors.PINK,
            connection_color=Colors.GREEN, thickness=3
        )
        output_image = render_to_image(render_data, Image.fromarray(frame))
        cv_image = np.array(output_image)
        
        cv.putText(cv_image, f"Yaw: {round(yaw, 2)}", (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if self.received_number is not None:
            cv.putText(cv_image , f"Ipad Data: {self.received_number}", (20, 80),
            cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv.imshow('Smart Pauser', cv_image)
        k = cv.waitKey(1) & 0xff
        if k == 27:
            stop_event.set()
            cv.destroyAllWindows()
            exit(0)

    def control_playback(self, yaw):
        is_paying_attention = self.yaw_lower_bound < yaw < self.yaw_upper_bound or self.received_number
        if is_paying_attention:
            if self.video_state == 'PAUSED':
                self.send_data(f"Time Locked {self.time_locked}, Time Geeked: {self.time_geeked}, Max Geek: {self.max_geek}, Num Geeked: {self.num_geeked}" + "\n")
                self.last_seen_paying_attention = time.time()
                self.time_geeked += time.time() - self.pause_start
                self.max_geek = max(self.max_geek, time.time() - self.pause_start)
                self.keyboard.press(self.PLAYBACK_KEY)
                self.keyboard.release(self.PLAYBACK_KEY)
                self.video_state = 'PLAYING'
        else:
            if self.video_state == 'PLAYING' and \
               (time.time() - self.last_seen_paying_attention) > self.ATTENTION_GRACE_PERIOD:
                self.time_locked += time.time() - self.last_seen_paying_attention
                self.pause_start = time.time()
                self.keyboard.press(self.PLAYBACK_KEY)
                self.keyboard.release(self.PLAYBACK_KEY)
                self.num_geeked += 1  
                self.video_state = 'PAUSED'

def run_mxa(app, dfp):
    global accl_ref
    try:
        accl = AsyncAccl(dfp)
        accl_ref = accl
        accl.connect_input(app.generate_frame_face)
        accl.connect_input(app.generate_frame_landmark, 1)
        accl.connect_output(app.process_face)
        accl.connect_output(app.process_landmark, 1)
        accl.wait()
    except Exception:
        traceback.print_exc()
    finally:
        cv.destroyAllWindows()

def start_pipeline(app, dfp_path):
    t = threading.Thread(target=run_mxa, args=(app, dfp_path,), daemon=True)
    t.start()
    return t

def stop_pipeline():
    try:
        if accl_ref:
            pass
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

    app = App(cam, calibrated_yaw)
    dfp_path = "models/models.dfp"

    print("Starting Smart Pauser. Press 'ESC' in the display window to quit.")
    time.sleep(1)
   
    pipeline_thread = start_pipeline(app, dfp_path)

    try:
        while pipeline_thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping background threads...")
        stop_event.set()
        stop_pipeline()

    cam.release()
    cv.destroyAllWindows()
    print("Application closed.")