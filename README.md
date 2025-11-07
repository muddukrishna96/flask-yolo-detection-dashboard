# YOLO Object Detection Web Application

A real-time object detection web application built with Flask and YOLOv8/v9 models, featuring a modern dashboard interface for image, video, and multi-camera webcam processing.

## 🚀 Tech Stack

**Backend:**
- Python 3.x
- Flask - Web framework
- Ultralytics YOLO (v8/v9) - Object detection models
- OpenCV (cv2) - Computer vision processing
- PyTorch - Deep learning framework

**Frontend:**
- HTML5/CSS3/JavaScript
- Jinja2 - Template engine
- AdminLTE 3 - Dashboard UI framework
- Bootstrap 4 - Responsive layout
- Font Awesome - Icons
- jQuery - DOM manipulation

## ✨ Key Features

### Detection Capabilities
- **Image Detection** - Upload and process single images
- **Video Detection** - Process video files with frame-by-frame inference
- **Live Webcam** - Real-time single camera object detection
- **Dual Webcam** - Simultaneous two-camera processing with independent models

### Model Support
- YOLOv8n, YOLOv8s, YOLOv8m (Nano, Small, Medium)
- YOLOv9s (Small)
- YOLOv8n-OIV7 (OpenImages V7 dataset)
- Auto-download of missing model weights
- Model caching for faster inference

### User Interface
- Clean, responsive dashboard powered by AdminLTE
- Inline results display (no external windows)
- Real-time MJPEG video streaming
- Model selection dropdown
- Camera configuration modal for multi-camera setup
- Loading spinner with visual feedback
- One-click start/stop controls

## 📁 Project Structure

```
├── webapp.py                 # Main Flask application & routes
├── backend/                  # Modular backend processing
│   ├── __init__.py          # Package initialization
│   ├── model_manager.py     # Model loading and caching
│   ├── image_video_processor.py  # Image & video processing
│   ├── single_camera.py     # Single webcam processing
│   ├── dual_camera.py       # Dual webcam processing
│   └── video_stream.py      # Video streaming utilities
├── templates/
│   ├── base.html            # Base template with layout
│   ├── index.html           # Main dashboard page
│   ├── navigation.html      # Top navbar
│   ├── sidebar.html         # Sidebar menu with logo
│   ├── footer.html          # Footer section
│   └── scripts.html         # JS includes
├── static/
│   └── assets/
│       ├── css/             # AdminLTE & Bootstrap styles
│       ├── js/              # AdminLTE & plugins
│       ├── plugins/         # jQuery, DataTables, etc.
│       ├── logo.png         # Application logo
│       └── img/             # Images & icons
├── uploads/                 # Temporary uploaded files
├── runs/
│   └── detect/             # YOLO detection outputs
├── models/                  # Local YOLO model weights (optional)
├── yolov9c.pt              # YOLOv9 model weights
└── yolov9s.pt              # YOLOv9 small model weights
```

## 🔧 Installation & Setup

**1. Install Dependencies:**
```bash
pip install ultralytics flask opencv-python torch torchvision
```

**2. Navigate to Project Directory:**
```bash
cd path/to/your/code/directory
```

**3. (Optional) Download YOLO Weights:**
Place model weights (yolov9s.pt, yolov9c.pt) in the project root, or let the app auto-download them.

**4. Run the Application:**
```bash
python webapp.py
```

**5. Access the Dashboard:**
Open your browser and go to: `http://localhost:5000`

## 🎯 Usage

### Image/Video Detection
1. Click "Choose File" and select an image or video
2. Select your preferred YOLO model from the dropdown
3. Click "Predict" to process
4. View results inline with bounding boxes and labels

### Single Webcam
1. Click "Webcam" button
2. Select "Single Camera" mode
3. Choose detection model
4. Click "Start Webcam"
5. Live stream appears with real-time detections

### Dual Webcam
1. Click "Webcam" button
2. Select "Multiple Cameras (2)" mode
3. Choose models for Camera 0 and Camera 1
4. Click "Start Webcam"
5. Split-screen view with parallel processing

## 🔑 Key Routes

- `/` - Main dashboard
- `/predict` (POST) - Upload & process files
- `/display/<folder>/<filename>` - Serve processed outputs
- `/video_feed` - MJPEG stream for processed videos
- `/webcam_feed?model=<model>` - Single webcam stream
- `/webcam_feed_dual?model0=<m0>&model1=<m1>` - Dual webcam stream

## 📊 Performance Tips

- Use smaller models (YOLOv8n) for real-time webcam processing
- Enable GPU acceleration with CUDA-enabled PyTorch
- Adjust image size for speed/accuracy trade-off
- Avoid saving frames to disk for video/webcam streams

## 🎨 Customization

- **Logo:** Replace `/static/assets/logo.png` with your brand logo
- **Colors:** Modify AdminLTE theme in `/static/assets/css/`
- **Models:** Add custom YOLO models to `models/` directory

## 🙏 Credits

This project was built upon the base template created by Ms. Aarohi Singla. Special thanks for the excellent tutorial and starter code that made this enhanced dashboard possible.

