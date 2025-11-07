"""
Video streaming utilities for processed videos.
Handles MJPEG streaming from saved video files.
"""

import os
import time
import cv2


def get_frame():
    """
    Generator: Read frames from output.mp4 and yield as MJPEG stream.
    Default fallback for video streaming.
    
    Yields:
        bytes: MJPEG frame data
    """
    mp4_files = 'output.mp4'
    video = cv2.VideoCapture(mp4_files)
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image) 
      
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')   
        time.sleep(0.1)


def stream_file_mp4_as_mjpeg(mp4_path):
    """
    Generator: Read an mp4 file and yield MJPEG frames.
    
    Args:
        mp4_path: Path to the mp4 file to stream
        
    Yields:
        bytes: MJPEG frame data
    """
    video = cv2.VideoCapture(mp4_path)
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.03)


def find_latest_video_folder():
    """
    Find the most recently created folder in runs/detect.
    
    Returns:
        str: Name of the latest subfolder
    """
    folder_path = os.path.join(os.getcwd(), 'runs', 'detect')
    subfolders = [f for f in os.listdir(folder_path) 
                  if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    return latest_subfolder


def get_video_from_folder(folder):
    """
    Get the first MP4 file from a specific runs/detect subfolder.
    
    Args:
        folder: Subfolder name in runs/detect
        
    Returns:
        str: Full path to the mp4 file, or None if not found
    """
    folder_dir = os.path.join(os.getcwd(), 'runs', 'detect', folder)
    if os.path.isdir(folder_dir):
        mp4s = [f for f in os.listdir(folder_dir) if f.lower().endswith('.mp4')]
        if mp4s:
            return os.path.join(folder_dir, mp4s[0])
    return None
