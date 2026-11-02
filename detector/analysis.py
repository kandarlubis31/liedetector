import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

class LieDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.blink_counter = 0
        self.ear_threshold = 0.25
        
    def eye_aspect_ratio(self, eye):
        # Hitung rasio aspek mata
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    def analyze_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        results = {
            'blink_count': 0,
            'smile_intensity': 0,
            'gaze_direction': None,
            'lie_probability': 0
        }
        
        for face in faces:
            landmarks = self.predictor(gray, face)
            
            # Eye tracking
            left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
            right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]
            
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            
            ear = (left_ear + right_ear) / 2.0
            
            if ear < self.ear_threshold:
                self.blink_counter += 1
            
            # Simpan hasil
            results['blink_count'] = self.blink_counter
            results['lie_probability'] = min(0.99, self.blink_counter * 0.1)
            
        return result