# detector/calibrate.py
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import defaultdict
from typing import Dict, List

class FaceCalibrator:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.calibration_data = defaultdict(list)
        self.thresholds = {}

    def collect_samples(self, duration: int = 30, show_feed: bool = True) -> Dict[str, float]:
        """
        Collect facial samples for calibration
        
        Args:
            duration: Calibration duration in seconds
            show_feed: Whether to show live calibration feed
            
        Returns:
            Dictionary of optimal thresholds for detection
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open video capture device")

        print("Calibration starting... Maintain neutral expression")
        start_time = time.time()
        last_update = time.time()
        frame_count = 0

        try:
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame_count += 1
                current_time = time.time()
                elapsed = current_time - start_time
                remaining = max(0, duration - elapsed)

                # Process frame
                results = self._process_frame(frame)
                if results:
                    self._extract_features(results)

                # Show calibration progress
                if show_feed:
                    self._display_calibration_frame(frame, elapsed, remaining, frame_count)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Periodic status update
                if current_time - last_update > 5:
                    print(f"Calibrating... {int(elapsed)}s/{duration}s completed")
                    last_update = current_time

            self._calculate_thresholds()
            print("\nCalibration complete!")
            print(f"Processed {frame_count} frames")
            print("Recommended thresholds:", self.thresholds)
            
            return self.thresholds

        finally:
            cap.release()
            if show_feed:
                cv2.destroyAllWindows()
            self.face_mesh.close()

    def _process_frame(self, frame):
        """Process a single frame for facial landmarks"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.face_mesh.process(rgb_frame)

    def _extract_features(self, results):
        """Extract facial features from detection results"""
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Eye openness (average of both eyes)
        left_eye = abs(landmarks[145].y - landmarks[159].y)
        right_eye = abs(landmarks[374].y - landmarks[386].y)
        self.calibration_data['eye_openness'].append((left_eye + right_eye) / 2)
        
        # Lip distance
        upper_lip = landmarks[13].y
        lower_lip = landmarks[14].y
        self.calibration_data['lip_distance'].append(abs(upper_lip - lower_lip))
        
        # Brow furrow (forehead wrinkles)
        brow_left = landmarks[105].y
        brow_right = landmarks[334].y
        self.calibration_data['brow_furrow'].append(abs(brow_left - brow_right))

    def _calculate_thresholds(self):
        """Calculate optimal thresholds from collected data"""
        for feature, values in self.calibration_data.items():
            if not values:
                continue
                
            mean = np.mean(values)
            std = np.std(values)
            
            # Dynamic threshold calculation
            if feature == 'eye_openness':
                self.thresholds['eye_blink'] = mean - 2*std  # Eyes more closed than normal
            elif feature == 'lip_distance':
                self.thresholds['lip_tightness'] = mean + 2*std  # Lips more tense than normal
            elif feature == 'brow_furrow':
                self.thresholds['brow_furrow'] = mean + 1.5*std  # More furrowed than normal

    def _display_calibration_frame(self, frame, elapsed, remaining, frame_count):
        """Display calibration progress on frame"""
        # Add calibration info
        cv2.putText(frame, f"Calibrating: {int(elapsed)}s / {int(remaining)}s remaining", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Frames processed: {frame_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add instructions
        cv2.putText(frame, "Maintain neutral expression", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Calibration Feed', frame)

    def save_calibration(self, filepath: str = 'calibration.npy'):
        """Save calibration data to file"""
        np.save(filepath, {
            'data': dict(self.calibration_data),
            'thresholds': self.thresholds
        })

    def load_calibration(self, filepath: str = 'calibration.npy'):
        """Load calibration data from file"""
        data = np.load(filepath, allow_pickle=True).item()
        self.calibration_data = defaultdict(list, data['data'])
        self.thresholds = data['thresholds']

