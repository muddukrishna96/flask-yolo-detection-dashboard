"""
Image and video processing functions for YOLO object detection.
Handles file upload, detection, and result generation.
"""

import os
import cv2
import numpy as np
from backend.model_manager import get_model


def add_model_overlay(image, model_name):
    """
    Add a stylish model name overlay to the bottom right corner of an image.
    
    Args:
        image: OpenCV image (numpy array)
        model_name: Name of the YOLO model used
        
    Returns:
        Image with overlay added
    """
    height, width = image.shape[:2]
    
    # Clean model name (remove .pt extension)
    display_name = model_name.replace('.pt', '').upper()
    
    # Text properties - increased by 40% for better visibility
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.0  # Increased from 0.7
    font_thickness = 3  # Increased from 2
    text = f"Model: {display_name}"
    
    # Get text size for background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    # Position at bottom right with padding
    padding = 20  # Increased from 15
    x = width - text_width - padding - 14  # Increased margin
    y = height - padding
    
    # Create semi-transparent overlay
    overlay = image.copy()
    
    # Draw rounded rectangle background (stylish dark background)
    rect_x1 = x - 14  # Increased padding
    rect_y1 = y - text_height - 14
    rect_x2 = x + text_width + 14
    rect_y2 = y + baseline + 7
    
    # Draw background with slight transparency
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
    
    # Add colored accent border (gradient-like effect with cyan/blue) - thicker border
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 200, 0), 4)  # Increased from 3
    
    # Blend overlay with original image for transparency
    alpha = 0.75
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    # Add text in bright cyan color
    cv2.putText(image, text, (x, y), font, font_scale, (255, 255, 0), font_thickness, cv2.LINE_AA)
    
    return image


# Supported file extensions
SUPPORTED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp'}
SUPPORTED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm', 'mpeg', 'mpg'}


def is_image_file(filename):
    """
    Check if file is a supported image format.
    
    Args:
        filename: Name of the file
        
    Returns:
        bool: True if supported image format
    """
    ext = get_file_extension(filename)
    return ext in SUPPORTED_IMAGE_EXTENSIONS


def is_video_file(filename):
    """
    Check if file is a supported video format.
    
    Args:
        filename: Name of the file
        
    Returns:
        bool: True if supported video format
    """
    ext = get_file_extension(filename)
    return ext in SUPPORTED_VIDEO_EXTENSIONS


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
    
    # Add model overlay to the processed image
    processed_path = os.path.join(folder_path, latest_subfolder, processed_filename)
    if os.path.exists(processed_path):
        result_img = cv2.imread(processed_path)
        result_img = add_model_overlay(result_img, model_name)
        cv2.imwrite(processed_path, result_img)
    
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
        
        # Add model overlay to each frame
        res_plotted = add_model_overlay(res_plotted, model_name)
        
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
