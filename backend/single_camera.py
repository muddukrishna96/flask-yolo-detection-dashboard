"""
Single webcam processing for real-time YOLO object detection.
Captures frames from one camera and yields MJPEG stream.
"""

import time
import cv2
from backend.model_manager import get_model


def get_webcam_frame(model_name='yolov8n.pt'):
    """
    Capture frames from default webcam, run YOLO inference and yield MJPEG frames.
    
    Args:
        model_name: YOLO model to use for inference
        
    Yields:
        bytes: MJPEG frame data
    """
    # Preload model before opening camera to reduce startup time
    try:
        model = get_model(model_name)
    except Exception:
        model = None
    
    # Open camera with optimized settings
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow on Windows for faster init
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
    
    if not cap.isOpened():
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference if model is loaded
            if model:
                try:
                    results = model(frame)
                    res_plotted = results[0].plot()
                except Exception:
                    res_plotted = frame
            else:
                res_plotted = frame

            ret2, jpeg = cv2.imencode('.jpg', res_plotted)
            if not ret2:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            time.sleep(0.03)
    finally:
        cap.release()
