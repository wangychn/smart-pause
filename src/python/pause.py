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
import signal
import os, traceback
import socket
import requests  # <-- CHANGE: Added requests library

stop_event = threading.Event()
accl_ref = None
SERVER_A_IP = '127.0.0.1'
# --- CHANGE: Replaced SERVER_B_IP and PORT with a full URL ---
SERVER_B_URL = 'http://172.29.112.216:8000/get_data'
SOCKET_A_PORT = 65432

# --- Part 1: The "Eyes" - Head Pose Estimation (Unchanged) ---
def get_head_pose(landmarks, frame_shape):
    face_3d_model = np.array([
        [0.0, 0.0, 0.0], [0.0, -330.0, -65.0], [-225.0, 170.0, -135.0],
        [225.0, 170.0, -135.0], [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]
    ], dtype=np.float64)
    face_2d_points = np.array([
        (landmarks[1].x, landmarks[1].y), (landmarks[152].x, landmarks[152].y),
        (landmarks[263].x, landmarks[263].y), (landmarks[33].x, landmarks[33].y),
        (landmarks[287].x, landmarks[287].y), (landmarks[57].x, landmarks[57].y)
    ], dtype=np.float64)
    focal_length = frame_shape[1]
    center = (frame_shape[1] / 2, frame_shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float64
    )
    dist_coeffs = np.zeros((4, 1))
    (_, rotation_vector, _) = cv.solvePnP(
        face_3d_model, face_2d_points, camera_matrix, dist_coeffs,
        flags=cv.SOLVEPNP_ITERATIVE
    )
    rotation_matrix, _ = cv.Rodrigues(rotation_vector)
    yaw = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    return yaw * (180.0 / np.pi)

# --- Pipeline Component Classes (Unchanged) ---
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
            if not ok: return None
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

        # --- CHANGE: Removed all setup for socket_server_b ---
        # Still connect to server A using the original socket method
        self.socket_server_a = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_server_a.connect((SERVER_A_IP, SOCKET_A_PORT))
        print("Connected to Server A (socket)")

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
        cv.imshow('Smart Pauser', cv_image)
        k = cv.waitKey(1) & 0xff
        if k == 27:
            cv.destroyAllWindows()
            exit(0)

    def control_playback(self, yaw):
        # --- CHANGE: Replaced socket.recv() with requests.get() ---
        try:
            # Make a web request to Server B to get data.
            # The timeout prevents the app from freezing if the server is unresponsive.
            response = requests.get(SERVER_B_URL, timeout=0.5)
            if response.status_code == 200:
                data = response.json()  # Parse the JSON response
                received_number = data.get('number')
                print(f"Received from Server B: {received_number}")
            else:
                print(f"Server B returned status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            # This will catch connection errors, timeouts, etc.
            print(f"Could not connect to Server B: {e}")

        is_paying_attention = self.yaw_lower_bound < yaw < self.yaw_upper_bound
        if is_paying_attention:
            if self.video_state == 'PAUSED':
                self.send_data(f"Resuming video... Time Locked {self.time_locked}, Time Geeked: {self.time_geeked}, Max Geek: {self.max_geek}, Num Geeked: {self.num_geeked}")
                self.last_seen_paying_attention = time.time()
                self.time_geeked += time.time() - self.pause_start
                self.max_geek = max(self.max_geek, time.time() - self.pause_start)
                self.keyboard.press(self.PLAYBACK_KEY)
                self.keyboard.release(self.PLAYBACK_KEY)
                self.video_state = 'PLAYING'
        else:
            if self.video_state == 'PLAYING' and \
               (time.time() - self.last_seen_paying_attention) > self.ATTENTION_GRACE_PERIOD:
                self.num_geeked += 1
                self.time_locked += time.time() - self.last_seen_paying_attention
                self.pause_start = time.time()
                self.keyboard.press(self.PLAYBACK_KEY)
                self.keyboard.release(self.PLAYBACK_KEY)
                self.video_state = 'PAUSED'

# --- Remaining code is unchanged ---
def run_mxa(dfp):
    global accl_ref
    try:
        accl = AsyncAccl(dfp)
        accl_ref = accl
        accl.connect_input(app.generate_frame_face)
        accl.connect_input(app.generate_frame_landmark, 1)
        accl.connect_output(app.process_face)
        accl.connect_output(app.process_landmark, 1)
        accl.wait()
    except Exception as e:
        traceback.print_exc()
    finally:
        cv.destroyAllWindows()

def start_pipeline(dfp_path):
    t = threading.Thread(target=run_mxa, args=(dfp_path,), daemon=True)
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
        exit(1)
    try:
        calibrated_yaw = float(sys.argv[1])
    except ValueError:
        exit(1)
    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        exit()

    app = App(cam, calibrated_yaw)
    dfp_path = "models/models.dfp"
    time.sleep(1)
    thread = start_pipeline(dfp_path)
    try:
        while thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_pipeline()
    cam.release()
    cv.destroyAllWindows()