from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import mediapipe as mp
import time
from collections import deque
import threading
import numpy as np  # Added missing import

# Initialize global variables at module level
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,  # Fixed typo: detection -> detection
    min_tracking_confidence=0.5)

blink_history = deque(maxlen=10)
blink_threshold = 0.04
calibration_time = 10  # seconds
calibration_start = None
min_analysis_duration = 15  

analysis_results = {
    'is_lying': False,
    'confidence': 0,
    'metrics': {
        'eye_blink': {'is_lying': False, 'confidence': 0, 'blink_rate': 'normal'},
        'lip_movement': {'is_lying': False, 'confidence': 0, 'movement': 'normal'},  # Fixed typo
    },
    'calibrated': False,
    'calibration_data': {
        'eye_open': [],
        'lip_distance': [],
        'eye_threshold': 0.04,
        'lip_threshold': 0.05
    }
}

def detector_view(request):
    return render(request, 'detector/detector.html')

def video_feed(request):
    try:
        return StreamingHttpResponse(gen_frames(),
                                  content_type='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Video feed error: {str(e)}")
        error_img = cv2.imencode('.jpg', np.zeros((480,640,3), dtype=np.uint8))[1].tobytes()
        return StreamingHttpResponse(
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + error_img + b'\r\n',
            content_type='multipart/x-mixed-replace; boundary=frame'
        )

@csrf_exempt
def start_analysis(request):
    global analysis_start_time  
    global analysis_results, calibration_start
    analysis_start_time = time.time()

    if request.method == 'POST':
        analysis_results = {
            'is_lying': False,
            'confidence': 0,
            'metrics': {
                'eye_blink': {'is_lying': False, 'confidence': 0, 'blink_rate': 'normal'},
                'lip_movement': {'is_lying': False, 'confidence': 0, 'movement': 'normal'},
            },
            'calibrated': False,
            'calibration_data': {
                'eye_open': [],
                'lip_distance': [],
                'eye_threshold': 0.04,
                'lip_threshold': 0.05
            }
        }
        calibration_start = time.time()
        return JsonResponse({'status': 'started', 'calibrating': True})
    return JsonResponse({'error': 'Invalid request'}, status=400)

def get_results(request):
    return JsonResponse(analysis_results)

def gen_frames():
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not camera.isOpened():
        print("Error: Could not open video source")
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + 
               cv2.imencode('.jpg', blank_frame)[1].tobytes() + 
               b'\r\n')
        return
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Failed to capture frame")
                break
                
            # Perform analysis
            results = analyze_face(frame)
            
            # Update global results
            global analysis_results
            analysis_results = results
            
            # Add overlay to frame
            label = "BERBOHONG!" if results['is_lying'] else "JUJUR"
            color = (0, 0, 255) if results['is_lying'] else (0, 255, 0)
            cv2.putText(frame, label, (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Add calibration status
            if not results['calibrated'] and calibration_start:
                calibration_progress = min(1.0, (time.time() - calibration_start) / calibration_time)
                cv2.putText(frame, f"Calibrating: {int(calibration_progress*100)}%", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode frame")
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        camera.release()
        print("Camera released")

def analyze_face(frame):
    global analysis_start_time
    
    if time.time() - analysis_start_time < min_analysis_duration:
        return {
            'is_lying': False,
            'confidence': 0,
            'metrics': {
                'eye_blink': {'is_lying': False, 'confidence': 0, 'blink_rate': 'calibrating'},
                'lip_movement': {'is_lying': False, 'confidence': 0, 'movement': 'calibrating'}
            },
            'calibrated': False
        }

    # Convert to RGB for mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    # Initialize default values
    metrics = {
        'eye_blink': {'is_lying': False, 'confidence': 0, 'blink_rate': 'normal'},
        'lip_movement': {'is_lying': False, 'confidence': 0, 'movement': 'normal'},
        'calibrated': analysis_results.get('calibrated', False),
        'calibration_data': analysis_results.get('calibration_data', {
            'eye_threshold': 0.04,
            'lip_threshold': 0.05
        })
    }

    lying = False
    confidence = 0
    
    if results.multi_face_landmarks:
        # Calibration phase
        if not metrics['calibrated'] and calibration_start:
            elapsed = time.time() - calibration_start
            if elapsed < calibration_time:
                update_calibration_data(results)
                return {
                    'is_lying': False,
                    'confidence': 0,
                    'metrics': metrics,
                    'calibrated': False
                }
            else:
                metrics['calibrated'] = True
                metrics['calibration_data'] = finalize_calibration()
                analysis_results['calibrated'] = True
                analysis_results['calibration_data'] = metrics['calibration_data']
        
        # Perform analysis
        eye_results = detect_eye_blink(results, metrics['calibration_data'])
        lip_results = detect_lip_movement(results, metrics['calibration_data'])
        
        lying = eye_results['is_lying'] or lip_results['is_lying']
        confidence = max(eye_results['confidence'], lip_results['confidence'])
        
        metrics.update({
            'eye_blink': eye_results,
            'lip_movement': lip_results
        })
        
    return {
        'is_lying': lying,
        'confidence': confidence,
        'metrics': metrics,
        'calibrated': metrics['calibrated']
    }

def update_calibration_data(face_results):
    global analysis_results
    
    landmarks = face_results.multi_face_landmarks[0].landmark
    
    left_eye_open = abs(landmarks[159].y - landmarks[145].y)
    right_eye_open = abs(landmarks[386].y - landmarks[374].y)
    lip_distance = abs(landmarks[13].y - landmarks[14].y)
    
    if 'calibration_data' not in analysis_results:
        analysis_results['calibration_data'] = {
            'eye_open': [],
            'lip_distance': []
        }
    
    analysis_results['calibration_data']['eye_open'].append((left_eye_open + right_eye_open) / 2)
    analysis_results['calibration_data']['lip_distance'].append(lip_distance)

def finalize_calibration():
    global analysis_results
    cal_data = analysis_results['calibration_data']
    
    if not cal_data.get('eye_open') or not cal_data.get('lip_distance'):
        return {
            'eye_threshold': 0.04,
            'lip_threshold': 0.05
        }
    
    avg_eye = sum(cal_data['eye_open']) / len(cal_data['eye_open'])
    avg_lip = sum(cal_data['lip_distance']) / len(cal_data['lip_distance'])
    
    return {
        'eye_threshold': avg_eye * 0.7,
        'lip_threshold': avg_lip * 1.3
    }

def detect_eye_blink(face_results, calibration):
    global blink_history
    
    try:
        landmarks = face_results.multi_face_landmarks[0].landmark
        
        # Calculate eye aspect ratio (more accurate than simple distance)
        left_eye_vertical = abs(landmarks[159].y - landmarks[145].y)
        left_eye_horizontal = abs(landmarks[263].x - landmarks[362].x)
        left_eye_ratio = left_eye_vertical / left_eye_horizontal
        
        right_eye_vertical = abs(landmarks[386].y - landmarks[374].y)
        right_eye_horizontal = abs(landmarks[133].x - landmarks[33].x)
        right_eye_ratio = right_eye_vertical / right_eye_horizontal
        
        eye_ratio = (left_eye_ratio + right_eye_ratio) / 2
        
        threshold = calibration.get('eye_threshold', 0.25)  # Typical EAR threshold
        
        if eye_ratio < threshold:
            blink_history.append(1)
            blink_rate = sum(blink_history) / len(blink_history)
            
            # Scale confidence from 0-100% based on deviation
            confidence = min(100, (threshold - eye_ratio) / threshold * 150)
            
            return {
                'is_lying': blink_rate > 0.5,  # If blinking more than 50% of frames
                'confidence': confidence,
                'blink_rate': 'high' if blink_rate > 0.5 else 'normal'
            }
        
        return {
            'is_lying': False,
            'confidence': 0,
            'blink_rate': 'normal'
        }
    except Exception as e:
        print(f"Eye detection error: {str(e)}")
        return {
            'is_lying': False,
            'confidence': 0,
            'blink_rate': 'error'
        }

def detect_lip_movement(face_results, calibration):
    try:
        landmarks = face_results.multi_face_landmarks[0].landmark
        
        lip_distance = abs(landmarks[13].y - landmarks[14].y)
        threshold = calibration.get('lip_threshold', 0.05)
        
        if lip_distance > threshold:
            return {
                'is_lying': True,
                'confidence': min(0.8, (lip_distance - threshold)/threshold * 2),
                'movement': 'abnormal'  # Note: keeping original spelling for consistency
            }
            
        return {
            'is_lying': False,
            'confidence': 0.1,
            'movement': 'normal'
        }
    except Exception as e:
        print(f"Lip detection error: {str(e)}")
        return {
            'is_lying': False,
            'confidence': 0,
            'movement': 'error'
        }