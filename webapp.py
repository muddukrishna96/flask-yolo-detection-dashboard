"""
YOLO Object Detection Web Application
Flask-based web interface for real-time object detection using YOLOv8/v9 models.
Supports image, video, single webcam, and dual webcam processing.
"""

import argparse
import os
import tempfile
from flask import Flask, render_template, request, url_for, Response, send_from_directory

# Import backend processing modules
from backend.model_manager import get_model, preload_default_model, register_custom_model, list_models
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
    return render_template('index.html', models=list_models(), selected_model='yolov8n.pt')

    
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
                                     selected_model='yolov8n.pt', message=message, models=list_models())
            
            # Validate file format BEFORE saving
            if not is_image_file(f.filename) and not is_video_file(f.filename):
                message = ("Unsupported file format! Only these formats are accepted:<br>"
                          "<strong>Images:</strong> PNG, JPG, JPEG, BMP, GIF, WebP<br>"
                          "<strong>Videos:</strong> MP4, AVI, MOV, MKV, FLV, WMV")
                return render_template('index.html', image_url='', video_present=False, 
                                     selected_model='yolov8n.pt', message=message, models=list_models())
            
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', f.filename)
            f.save(filepath)
            
            # Store filename globally for reference
            global imgpath
            predict_img.imgpath = f.filename
            
            # Determine selected model from form
            model_name = request.form.get('model', 'yolov8n.pt')

            # Validate that the selected model is available (either built-in or uploaded)
            available = list_models()
            if model_name not in available:
                message = (f"Selected model '{model_name}' is not available. "
                           "Please upload the model or choose one from the dropdown.")
                return render_template('index.html', image_url='', video_present=False,
                                       selected_model='yolov8n.pt', message=message, models=available)
            
            try:
                # Check if it's an image file
                if is_image_file(f.filename):
                    latest_subfolder, processed_filename = process_image(filepath, model_name)
                    image_url = url_for('display', folder=latest_subfolder, filename=processed_filename)
                    return render_template('index.html', image_url=image_url, video_present=False, 
                                         selected_model=model_name, models=list_models())
                
                # Check if it's a video file
                elif is_video_file(f.filename):
                    latest_subfolder = process_video(filepath, model_name)
                    return render_template('index.html', image_url='', video_present=True, 
                                         video_folder=latest_subfolder, selected_model=model_name, models=list_models())
            
            except Exception as e:
                # Clean up uploaded file on error
                if os.path.exists(filepath):
                    os.remove(filepath)
                    
                message = f"Error processing file with model '{model_name}': {str(e)}"
                return render_template('index.html', image_url='', video_present=False, 
                                     selected_model=model_name, message=message, models=list_models())
    
    # Default GET response
    return render_template('index.html', selected_model='yolov8n.pt', models=list_models())


@app.route('/upload_model', methods=['POST'])
def upload_model():
    """
    Upload a custom .pt model, load it temporarily, register in memory, and make it available for inference.
    This writes the uploaded file to a secure temporary file only while loading and deletes it immediately.
    """
    if 'model_file' not in request.files:
        return render_template('index.html', message='No model file provided.', models=list_models(), selected_model='yolov8n.pt')

    f = request.files['model_file']
    display_name = request.form.get('model_name') or f.filename
    disclaimer = request.form.get('disclaimer')

    # Require disclaimer checkbox
    if not disclaimer:
        return render_template('index.html', message='You must accept the disclaimer before uploading a model.', models=list_models(), selected_model='yolov8n.pt')

    if f.filename == '':
        return render_template('index.html', message='No file selected.', models=list_models(), selected_model='yolov8n.pt')

    # Only accept .pt files for now
    if '.' not in f.filename or f.filename.rsplit('.', 1)[1].lower() != 'pt':
        return render_template('index.html', message='Only .pt model files are accepted.', models=list_models(), selected_model='yolov8n.pt')

    # Enforce size limit (default 200MB)
    data = f.read()
    max_size = 200 * 1024 * 1024
    if len(data) > max_size:
        return render_template('index.html', message='Model file too large (max 200 MB).', models=list_models(), selected_model='yolov8n.pt')

    tmp_path = None
    try:
        # Save to a secure temporary file while loading
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp.write(data)
            tmp.flush()
            tmp_path = tmp.name

        # Load with Ultralytics YOLO to ensure compatibility with pipeline
        from ultralytics import YOLO
        model_obj = YOLO(tmp_path)

        # Register the model in memory under the provided display name
        register_custom_model(display_name, model_obj)

    except Exception as e:
        return render_template('index.html', message=f'Failed to load model: {e}', models=list_models(), selected_model='yolov8n.pt')

    finally:
        # remove temporary file if it exists
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    return render_template('index.html', message=f"Model '{display_name}' uploaded and registered.", models=list_models(), selected_model=display_name)


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
