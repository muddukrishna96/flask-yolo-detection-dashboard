"""
YOLO Object Detection Web Application
Flask-based web interface for real-time object detection using YOLOv8/v9 models.
Supports image, video, single webcam, and dual webcam processing.
"""

import argparse
import os
from flask import Flask, render_template, request, url_for, Response, send_from_directory

# Import backend processing modules
from backend.model_manager import get_model, preload_default_model
from backend.image_video_processor import (
    process_image, process_video, is_image_file, is_video_file
)
from backend.single_camera import get_webcam_frame
from backend.dual_camera import get_dual_webcam_frame
from backend.video_stream import get_frame, get_video_from_folder, stream_file_mp4_as_mjpeg


# Flask app initialization
app = Flask(__name__)

# Ensure uploads folder exists
os.makedirs(os.path.join(os.path.dirname(__file__), 'uploads'), exist_ok=True)


@app.route("/")
def hello_world():
    """Render main dashboard page."""
    return render_template('index.html')

    
@app.route("/", methods=["GET", "POST"])
def predict_img():
    """
    Handle image and video upload, process with selected YOLO model.
    
    Returns:
        Rendered template with detection results
    """
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            
            # Validate file was selected
            if f.filename == '':
                message = "No file selected. Please choose an image or video file."
                return render_template('index.html', image_url='', video_present=False, 
                                     selected_model='yolov8n.pt', message=message)
            
            # Validate file format BEFORE saving
            if not is_image_file(f.filename) and not is_video_file(f.filename):
                message = ("Unsupported file format! Only these formats are accepted:<br>"
                          "<strong>Images:</strong> PNG, JPG, JPEG, BMP, GIF, WebP<br>"
                          "<strong>Videos:</strong> MP4, AVI, MOV, MKV, FLV, WMV")
                return render_template('index.html', image_url='', video_present=False, 
                                     selected_model='yolov8n.pt', message=message)
            
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', f.filename)
            f.save(filepath)
            
            # Store filename globally for reference
            global imgpath
            predict_img.imgpath = f.filename
            
            # Determine selected model from form
            model_name = request.form.get('model', 'yolov8n.pt')
            
            try:
                # Check if it's an image file
                if is_image_file(f.filename):
                    latest_subfolder, processed_filename = process_image(filepath, model_name)
                    image_url = url_for('display', folder=latest_subfolder, filename=processed_filename)
                    return render_template('index.html', image_url=image_url, video_present=False, 
                                         selected_model=model_name)
                
                # Check if it's a video file
                elif is_video_file(f.filename):
                    latest_subfolder = process_video(filepath, model_name)
                    return render_template('index.html', image_url='', video_present=True, 
                                         video_folder=latest_subfolder, selected_model=model_name)
            
            except Exception as e:
                # Clean up uploaded file on error
                if os.path.exists(filepath):
                    os.remove(filepath)
                    
                message = f"Error processing file with model '{model_name}': {str(e)}"
                return render_template('index.html', image_url='', video_present=False, 
                                     selected_model=model_name, message=message)
    
    # Default GET response
    return render_template('index.html', selected_model='yolov8n.pt')


@app.route('/display/<folder>/<path:filename>')
def display(folder, filename):
    """
    Serve processed result files from runs/detect/<folder>/filename.
    
    Args:
        folder: Subfolder name in runs/detect
        filename: Name of the file to serve
        
    Returns:
        File from directory or 404
    """
    base = os.path.join(os.getcwd(), 'runs', 'detect')
    directory = os.path.join(base, folder)
    if not os.path.isdir(directory):
        return "Not found", 404
    return send_from_directory(directory, filename)


@app.route("/video_feed")
def video_feed():
    """
    Stream processed video as MJPEG.
    Accepts ?folder=<name> parameter to stream from specific runs/detect subfolder.
    """
    folder = request.args.get('folder', None)
    if folder:
        mp4_path = get_video_from_folder(folder)
        if mp4_path:
            return Response(stream_file_mp4_as_mjpeg(mp4_path), 
                          mimetype='multipart/x-mixed-replace; boundary=frame')
    
    # Fallback to default output.mp4
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/webcam')
def webcam_page():
    """Render webcam page with live inference and stop button."""
    return render_template('webcam.html')


@app.route('/webcam_feed')
def webcam_feed():
    """
    Stream single webcam MJPEG feed with YOLO inference.
    Accepts ?model=<model_name> parameter to select model.
    """
    model_name = request.args.get('model', 'yolov8n.pt')
    return Response(get_webcam_frame(model_name), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/webcam_feed_dual')
def webcam_feed_dual():
    """
    Stream dual webcam MJPEG feed with YOLO inference.
    Accepts ?model0=<model_name> and ?model1=<model_name> parameters.
    """
    model0 = request.args.get('model0', 'yolov8n.pt')
    model1 = request.args.get('model1', 'yolov8n.pt')
    return Response(get_dual_webcam_frame(model0, model1), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask YOLO object detection web application")
    parser.add_argument("--port", default=5000, type=int, help="Port number for Flask server")
    args = parser.parse_args()
    
    # Preload default model into cache for faster first request
    preload_default_model()
    
    app.run(host="0.0.0.0", port=args.port)
