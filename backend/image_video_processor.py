"""
Image and video processing functions for YOLO object detection.
Handles file upload, detection, and result generation.
"""

import colorsys
import os
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from backend.model_manager import get_model


def draw_neon_corner_box(frame, x1, y1, x2, y2, color=(0, 255, 255), thickness=3, corner_len=25, glow_intensity=0.1):
    """Draw a glowing neon-style corner box around the object."""

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, glow_intensity, frame, 1 - glow_intensity, 0, frame)

    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

    return frame


_CLASS_COLOR_HEX = (
    0xFF6B6B,
    0xFFD166,
    0x06D6A0,
    0x118AB2,
    0xEF476F,
    0x9B5DE5,
    0xF15BB5,
    0x00BBF9,
    0x00F5D4,
    0xFEE440,
)


def _hex_to_bgr(value: int) -> tuple[int, int, int]:
    r = (value >> 16) & 0xFF
    g = (value >> 8) & 0xFF
    b = value & 0xFF
    return (b, g, r)


_CLASS_COLOR_PALETTE = tuple(_hex_to_bgr(v) for v in _CLASS_COLOR_HEX)


def _get_color_for_class(class_id: int | None, default_color=(0, 255, 255)) -> tuple[int, int, int]:
    if class_id is None or class_id < 0:
        return default_color

    if class_id < len(_CLASS_COLOR_PALETTE):
        return _CLASS_COLOR_PALETTE[class_id]

    hue = (class_id * 0.161803) % 1.0  # fractional golden ratio for good separation
    saturation = 0.78
    value = 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (int(b * 255), int(g * 255), int(r * 255))


def annotate_detections(image, prediction, color=(0, 255, 255)):
    """Apply custom neon-style visualization for YOLO bounding boxes."""

    if image is None or prediction is None:
        return image

    annotated = image.copy()
    boxes = getattr(prediction, 'boxes', None)
    if boxes is None or len(boxes) == 0:
        return annotated

    xyxy = boxes.xyxy
    if hasattr(xyxy, 'cpu'):
        xyxy = xyxy.cpu()
    xyxy = np.asarray(xyxy)

    confs = getattr(boxes, 'conf', None)
    if confs is not None and hasattr(confs, 'cpu'):
        confs = confs.cpu().numpy()
    elif confs is not None:
        confs = np.asarray(confs)

    classes = getattr(boxes, 'cls', None)
    if classes is not None and hasattr(classes, 'cpu'):
        classes = classes.cpu().numpy().astype(int)
    elif classes is not None:
        classes = np.asarray(classes).astype(int)

    names = getattr(prediction, 'names', None)
    if names is None:
        model_ref = getattr(prediction, 'model', None)
        names = getattr(model_ref, 'names', None)

    height, width = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    base_scale = 0.6
    base_thickness = 2

    for idx, coords in enumerate(xyxy):
        x1, y1, x2, y2 = coords
        x1 = int(max(0, min(round(x1), width - 1)))
        y1 = int(max(0, min(round(y1), height - 1)))
        x2 = int(max(0, min(round(x2), width - 1)))
        y2 = int(max(0, min(round(y2), height - 1)))

        if x2 <= x1 or y2 <= y1:
            continue

        class_id = None
        if classes is not None and idx < len(classes):
            class_id = int(classes[idx])

        box_color = _get_color_for_class(class_id, default_color=color)

        draw_neon_corner_box(annotated, x1, y1, x2, y2, color=box_color)

        label_parts = []
        if class_id is not None:
            if isinstance(names, dict):
                label_parts.append(str(names.get(class_id, class_id)))
            elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
                label_parts.append(str(names[class_id]))
            else:
                label_parts.append(str(class_id))

        if confs is not None and idx < len(confs):
            label_parts.append(f"{confs[idx]:.2f}")

        if label_parts:
            label_text = " ".join(label_parts)
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, base_scale, base_thickness)

            text_x = x1 + 8
            text_y = y1 - 12
            if text_y - text_h - baseline < 0:
                text_y = y1 + text_h + 12

            rect_start = (max(text_x - 6, 0), max(text_y - text_h - 6, 0))
            rect_end = (min(text_x + text_w + 6, width - 1), min(text_y + 6, height - 1))

            accent_color = box_color if sum(box_color) > 255 else (255, 255, 255)
            cv2.rectangle(annotated, rect_start, rect_end, (0, 0, 0), -1)
            cv2.putText(annotated, label_text, (text_x, text_y), font, base_scale, accent_color, base_thickness, cv2.LINE_AA)

    return annotated


def annotate_segmentation(image, prediction, mask_alpha: float = 0.45):
    """Apply segmentation masks with neon styling."""

    if image is None or prediction is None:
        return image

    annotated = image.copy()
    boxes = getattr(prediction, 'boxes', None)
    masks = getattr(prediction, 'masks', None)

    xyxy = None
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy
        if hasattr(xyxy, 'cpu'):
            xyxy = xyxy.cpu()
        xyxy = np.asarray(xyxy)
    else:
        xyxy = np.zeros((0, 4), dtype=np.float32)

    confs = getattr(boxes, 'conf', None)
    if confs is not None and hasattr(confs, 'cpu'):
        confs = confs.cpu().numpy()
    elif confs is not None:
        confs = np.asarray(confs)

    classes = getattr(boxes, 'cls', None)
    if classes is not None and hasattr(classes, 'cpu'):
        classes = classes.cpu().numpy().astype(int)
    elif classes is not None:
        classes = np.asarray(classes).astype(int)

    names = getattr(prediction, 'names', None)
    if names is None:
        model_ref = getattr(prediction, 'model', None)
        names = getattr(model_ref, 'names', None)

    mask_data = None
    if masks is not None and getattr(masks, 'data', None) is not None:
        mask_data = masks.data
        if hasattr(mask_data, 'cpu'):
            mask_data = mask_data.cpu().numpy()
        else:
            mask_data = np.asarray(mask_data)
    else:
        mask_data = np.zeros((0, image.shape[0], image.shape[1]), dtype=np.uint8)

    height, width = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    base_scale = 0.6
    base_thickness = 2

    prepared_masks = []
    if mask_data is not None and len(mask_data):
        for mask in mask_data:
            if mask.shape[-2:] != (height, width):
                resized = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
            else:
                resized = mask
            prepared_masks.append((resized > 0.5).astype(np.uint8))

    for idx, coords in enumerate(xyxy):
        x1, y1, x2, y2 = coords
        x1 = int(max(0, min(round(x1), width - 1)))
        y1 = int(max(0, min(round(y1), height - 1)))
        x2 = int(max(0, min(round(x2), width - 1)))
        y2 = int(max(0, min(round(y2), height - 1)))

        if x2 <= x1 or y2 <= y1:
            continue

        class_id = None
        if classes is not None and idx < len(classes):
            class_id = int(classes[idx])

        box_color = _get_color_for_class(class_id)

        if idx < len(prepared_masks):
            mask = prepared_masks[idx]
            if mask is not None:
                color_arr = np.zeros_like(annotated, dtype=np.uint8)
                color_arr[:] = box_color
                mask_bool = mask.astype(bool)
                annotated = np.where(mask_bool[..., None],
                                      (annotated * (1 - mask_alpha) + color_arr * mask_alpha).astype(np.uint8),
                                      annotated)

                try:
                    contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(annotated, contours, -1, box_color, 2)
                except Exception:
                    pass

        draw_neon_corner_box(annotated, x1, y1, x2, y2, color=box_color)

        label_parts = []
        if class_id is not None:
            if isinstance(names, dict):
                label_parts.append(str(names.get(class_id, class_id)))
            elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
                label_parts.append(str(names[class_id]))
            else:
                label_parts.append(str(class_id))

        if confs is not None and idx < len(confs):
            label_parts.append(f"{confs[idx]:.2f}")

        if label_parts:
            label_text = " ".join(label_parts)
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, base_scale, base_thickness)

            text_x = x1 + 8
            text_y = y1 - 12
            if text_y - text_h - baseline < 0:
                text_y = y1 + text_h + 12

            rect_start = (max(text_x - 6, 0), max(text_y - text_h - 6, 0))
            rect_end = (min(text_x + text_w + 6, width - 1), min(text_y + 6, height - 1))

            accent_color = box_color if sum(box_color) > 255 else (255, 255, 255)
            cv2.rectangle(annotated, rect_start, rect_end, (0, 0, 0), -1)
            cv2.putText(annotated, label_text, (text_x, text_y), font, base_scale, accent_color, base_thickness, cv2.LINE_AA)

    return annotated


def annotate_frame(image, prediction, task: str = 'detection'):
    if task == 'segmentation':
        return annotate_segmentation(image, prediction)
    return annotate_detections(image, prediction)


def _create_run_folder(task_dir: str, prefix: str):
    base_dir = Path(os.getcwd()) / 'runs' / task_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
    folder_name = f"{prefix}_{timestamp}"
    folder_path = base_dir / folder_name
    folder_path.mkdir(parents=True, exist_ok=False)
    return folder_name, folder_path


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


def process_image(filepath, model_name='yolov8n.pt', task: str = 'detection'):
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
    if img is None:
        raise ValueError(f"Unable to load image from {filepath}")

    model = get_model(model_name)
    results = model(img, verbose=False)
    prediction = results[0]

    annotated = annotate_frame(img, prediction, task=task)
    annotated = add_model_overlay(annotated, model_name)

    task_dir = 'detect' if task == 'detection' else 'segment'
    folder_name, folder_path = _create_run_folder(task_dir, 'image')
    original_name = Path(filepath).stem
    extension = Path(filepath).suffix or '.jpg'
    suffix = 'detected' if task == 'detection' else 'segmented'
    processed_filename = f"{original_name}_{suffix}{extension}"
    processed_path = folder_path / processed_filename

    cv2.imwrite(str(processed_path), annotated)

    return folder_name, processed_filename


def process_video(filepath, model_name='yolov8n.pt', task: str = 'detection'):
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
    if not cap.isOpened():
        raise ValueError(f"Unable to open video source {filepath}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    model = get_model(model_name)

    task_dir = 'detect' if task == 'detection' else 'segment'
    folder_name, folder_path = _create_run_folder(task_dir, 'video')
    video_name = Path(filepath).stem or 'processed'
    suffix = 'detected' if task == 'detection' else 'segmented'
    output_path = folder_path / f"{video_name}_{suffix}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)
            annotated = annotate_frame(frame, results[0], task=task)
            annotated = add_model_overlay(annotated, model_name)

            writer.write(annotated)
    finally:
        writer.release()
        cap.release()

    try:
        shutil.copyfile(output_path, 'output.mp4')
    except Exception:
        pass

    return folder_name


def get_file_extension(filename):
    """
    Extract file extension from filename.
    
    Args:
        filename: Name of the file
        
    Returns:
        str: Lowercase file extension without dot
    """
    return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
