"""
Dual webcam processing for simultaneous multi-camera YOLO object detection.
Captures frames from two cameras and yields combined MJPEG stream.
"""

import time
import cv2
import numpy as np
from backend.model_manager import get_model
from backend.image_video_processor import add_model_overlay


def get_dual_webcam_frame(model0='yolov8n.pt', model1='yolov8n.pt'):
    """
    Capture frames from two webcams simultaneously and run YOLO inference in parallel.
    Returns a vertically stacked split-screen view with model overlays.
    
    Args:
        model0: YOLO model for camera 0
        model1: YOLO model for camera 1
        
    Yields:
        bytes: MJPEG frame data with combined camera views
    """
    # Preload both models before opening cameras
    try:
        m0 = get_model(model0)
    except Exception:
        m0 = None
    
    try:
        m1 = get_model(model1)
    except Exception:
        m1 = None
    
    # Open cameras with optimized settings
    cap0 = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow on Windows
    cap0.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap0.isOpened() or not cap1.isOpened():
        cap0.release()
        cap1.release()
        return

    try:
        while True:
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()
            
            if not ret0 or not ret1:
                break

            # Process camera 0
            if m0:
                try:
                    results0 = m0(frame0)
                    res_plotted0 = results0[0].plot()
                    # Add stylish model overlay
                    res_plotted0 = add_model_overlay(res_plotted0, model0)
                except Exception:
                    res_plotted0 = frame0
            else:
                res_plotted0 = frame0
            
            # Process camera 1
            if m1:
                try:
                    results1 = m1(frame1)
                    res_plotted1 = results1[0].plot()
                    # Add stylish model overlay
                    res_plotted1 = add_model_overlay(res_plotted1, model1)
                except Exception:
                    res_plotted1 = frame1
            else:
                res_plotted1 = frame1
            
            # Resize both frames to same width for stacking
            height0, width0 = res_plotted0.shape[:2]
            height1, width1 = res_plotted1.shape[:2]
            target_width = max(width0, width1)
            
            if width0 != target_width:
                aspect0 = height0 / width0
                res_plotted0 = cv2.resize(res_plotted0, (target_width, int(target_width * aspect0)))
            if width1 != target_width:
                aspect1 = height1 / width1
                res_plotted1 = cv2.resize(res_plotted1, (target_width, int(target_width * aspect1)))
            
            # Add camera labels at top left
            cv2.putText(res_plotted0, "Camera 0", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(res_plotted1, "Camera 1", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Stack frames vertically
            combined_frame = np.vstack([res_plotted0, res_plotted1])
            
            ret2, jpeg = cv2.imencode('.jpg', combined_frame)
            if not ret2:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            time.sleep(0.03)
    finally:
        cap0.release()
        cap1.release()
