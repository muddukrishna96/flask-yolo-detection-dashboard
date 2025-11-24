"""
Single webcam processing for real-time YOLO object detection.
Captures frames from one camera and yields MJPEG stream.
"""

import time
import cv2
import numpy as np
from backend.model_manager import get_model
from backend.image_video_processor import add_model_overlay, annotate_detections


def get_webcam_frame(model_name='yolov8n.pt', camera_index=0):
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
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Use DirectShow on Windows for faster init
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
                    res_plotted = annotate_detections(frame, results[0])
                    res_plotted = add_model_overlay(res_plotted, model_name)
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


def get_split_webcam_frame(model0='yolov8n.pt', model1='yolov8n.pt', camera_index=0):
    """
    Capture frames from a single webcam, run two different YOLO models on the same frame,
    and yield a vertically stacked MJPEG stream showing both model outputs.

    Args:
        model0: YOLO model for the first view
        model1: YOLO model for the second view

    Yields:
        bytes: MJPEG frame data with combined model outputs
    """
    # Preload both models
    try:
        m0 = get_model(model0)
    except Exception:
        m0 = None

    try:
        m1 = get_model(model1)
    except Exception:
        m1 = None

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process with model0
            if m0:
                try:
                    r0 = m0(frame)
                    plotted0 = annotate_detections(frame, r0[0])
                    plotted0 = add_model_overlay(plotted0, model0)
                except Exception:
                    plotted0 = frame
            else:
                plotted0 = frame

            # Process with model1
            if m1:
                try:
                    r1 = m1(frame)
                    plotted1 = annotate_detections(frame, r1[0])
                    plotted1 = add_model_overlay(plotted1, model1)
                except Exception:
                    plotted1 = frame
            else:
                plotted1 = frame

            # Resize to same width
            h0, w0 = plotted0.shape[:2]
            h1, w1 = plotted1.shape[:2]
            target_w = max(w0, w1)

            if w0 != target_w:
                plotted0 = cv2.resize(plotted0, (target_w, int(target_w * (h0 / w0))))
            if w1 != target_w:
                plotted1 = cv2.resize(plotted1, (target_w, int(target_w * (h1 / w1))))

            # Add labels
            try:
                cv2.putText(plotted0, "Model A", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (25, 255, 255), 2)
                cv2.putText(plotted1, "Model B", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (25, 255, 255), 2)
            except Exception:
                pass

            # Stack with separator
            gap = 10
            separator = (np.ones((gap, target_w, 3), dtype=np.uint8) * 245)
            combined = np.vstack([plotted0, separator, plotted1])

            ret2, jpeg = cv2.imencode('.jpg', combined)
            if not ret2:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            time.sleep(0.03)
    finally:
        cap.release()
