"""
Image and video processing functions for YOLO object detection.
Handles file upload, detection, and result generation.
"""

import os
import cv2
from backend.model_manager import get_model


def process_image(filepath, model_name='yolov8n.pt'):
    """
    Process an image file with YOLO detection.
    
    Args:
        filepath: Path to the image file
        model_name: YOLO model to use for detection
        
    Returns:
        tuple: (latest_subfolder, processed_filename) for accessing results
        
    Raises:
        Exception: If model loading or detection fails
    """
    img = cv2.imread(filepath)
    
    # Load model and run detection
    model = get_model(model_name)
    detections = model(img, save=True)
    
    # Locate the latest runs/detect folder
    folder_path = os.path.join(os.getcwd(), 'runs', 'detect')
    subfolders = [d for d in os.listdir(folder_path) 
                  if os.path.isdir(os.path.join(folder_path, d))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    
    # Find processed image filename
    filename = os.path.basename(filepath)
    processed_filename = None
    
    for fname in os.listdir(os.path.join(folder_path, latest_subfolder)):
        if os.path.splitext(fname)[0] in filename:
            processed_filename = fname
            break
    
    if processed_filename is None:
        # Fallback to any jpg in folder
        files = [fn for fn in os.listdir(os.path.join(folder_path, latest_subfolder)) 
                if fn.lower().endswith('.jpg')]
        processed_filename = files[0] if files else ''
    
    return latest_subfolder, processed_filename


def process_video(filepath, model_name='yolov8n.pt'):
    """
    Process a video file with YOLO detection frame by frame.
    
    Args:
        filepath: Path to the video file
        model_name: YOLO model to use for detection
        
    Returns:
        str: latest_subfolder name where results are stored
        
    Raises:
        Exception: If model loading or video processing fails
    """
    cap = cv2.VideoCapture(filepath)
    
    # Get video dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))
    
    # Load model
    model = get_model(model_name)
    
    # Process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO detection on frame
        results = model(frame, save=True)
        res_plotted = results[0].plot()
        
        # Write frame to output video
        out.write(res_plotted)
    
    # Release resources
    out.release()
    cap.release()
    
    # Find the latest runs/detect folder
    folder_path = os.path.join(os.getcwd(), 'runs', 'detect')
    subfolders = [d for d in os.listdir(folder_path) 
                  if os.path.isdir(os.path.join(folder_path, d))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    
    return latest_subfolder


def get_file_extension(filename):
    """
    Extract file extension from filename.
    
    Args:
        filename: Name of the file
        
    Returns:
        str: Lowercase file extension without dot
    """
    return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
