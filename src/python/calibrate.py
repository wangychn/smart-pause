import cv2 as cv
import numpy as np
import time
import math
import sys
import os
from PIL import Image
from queue import Queue
from fdlite import FaceDetectionMXA, FaceLandmarkMXA
from fdlite.render import Colors, landmarks_to_render_data, render_to_image
from memryx import AsyncAccl

# This file is based on the original, working pause.py logic to ensure stability.

# --- Head Pose Estimation (Same as original) ---
def get_head_pose(landmarks, frame_shape):
    """Calculates head yaw from facial landmarks."""
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
        face_3d_model, face_2d_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE
    )
    rotation_matrix, _ = cv.Rodrigues(rotation_vector)
    yaw = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    return yaw * (180.0 / np.pi)

# --- Pipeline Component Classes (Same as original) ---
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
        return landmark_frame, landmarks

# --- Main Calibration Application Controller ---
class CalibrationApp:
    def __init__(self, cam):
        self.cam = cam
        self.face_app = FaceApp(cam)
        cam_size = (int(cam.get(cv.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv.CAP_PROP_FRAME_HEIGHT)))
        self.landmark_app = LandmarkApp(cam_size=cam_size)
        self.current_yaw = 0.0

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
        yaw = 999
        if landmarks is not None and frame is not None:
            yaw = get_head_pose(landmarks, frame.shape)
            self.current_yaw = yaw

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
        cv.putText(cv_image, "Position head and click button window", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.imshow('Calibration', cv_image)
        
        button_img = np.zeros((80, 400, 3), dtype=np.uint8) + 220
        cv.putText(button_img, "Set Center Position", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv.imshow('Controls', button_img)
        cv.setMouseCallback('Controls', self.on_mouse_click)

        if cv.waitKey(1) & 0xFF == 27:
            self.cleanup_and_exit(success=False)

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            print(f"{self.current_yaw}", flush=True)
            self.cleanup_and_exit(success=True)

    def cleanup_and_exit(self, success=True):
        self.cam.release()
        cv.destroyAllWindows()
        exit(0 if success else 1)

def run_mxa(app, dfp):
    accl = AsyncAccl(dfp)
    accl.connect_input(app.generate_frame_face)
    accl.connect_input(app.generate_frame_landmark, 1)
    accl.connect_output(app.process_face)
    accl.connect_output(app.process_landmark, 1)
    accl.wait()

if __name__ == '__main__':
    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Cannot open camera.", file=sys.stderr)
        exit(1)

    dfp_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "models.dfp")
    app = CalibrationApp(cam)
    run_mxa(app, dfp_path)

