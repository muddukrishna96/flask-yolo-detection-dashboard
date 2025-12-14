"""
YOLO Object Detection Web Application
Flask-based web interface for real-time object detection using YOLOv8/v9 models.
Supports image, video, single webcam, and dual webcam processing.
"""

import argparse
import asyncio
import atexit
import os
import threading
from typing import Optional
from flask import Flask, render_template, request, url_for, Response, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import cv2
import time

# Import backend processing modules
from backend.model_manager import (
    get_model,
    preload_default_model,
    register_custom_model,
    list_models,
    list_detection_models,
    list_segmentation_models,
    list_pose_models,
)
from backend.image_video_processor import (
    process_image,
    process_video,
    is_image_file,
    is_video_file,
    annotate_frame,
)

from backend.webrtc_session import create_answer_for_offer, close_all_peers, get_ice_servers_for_client


# Flask app initialization
app = Flask(__name__)


_SUPPORTED_TASKS = {'detection', 'segmentation', 'pose'}
_TASK_DEFAULT_MODELS = {
    'detection': 'yolov8n.pt',
    'segmentation': 'yolo11n-seg.pt',
    'pose': 'yolov8n-pose.pt',
}
_TASK_RUN_DIRS = {
    'detection': 'detect',
    'segmentation': 'segment',
    'pose': 'pose',
}


def _normalize_task(task_value: Optional[str]) -> str:
    task = (task_value or 'detection').strip().lower()
    return task if task in _SUPPORTED_TASKS else 'detection'


def _models_for_task(task: str) -> list[str]:
    if task == 'segmentation':
        return list_segmentation_models()
    if task == 'pose':
        return list_pose_models()
    return list_detection_models()


def _default_model_for_task(task: str, models: Optional[list[str]] = None) -> str:
    available = models if models is not None else _models_for_task(task)
    fallback = _TASK_DEFAULT_MODELS.get(task, _TASK_DEFAULT_MODELS['detection'])
    if fallback in available:
        return fallback
    if available:
        return available[0]
    return 'yolov8n.pt'


def _select_model_for_task(model_name: Optional[str], task: str, models: Optional[list[str]] = None) -> str:
    available = models if models is not None else _models_for_task(task)
    if model_name:
        candidate = model_name.strip()
        if candidate in available:
            return candidate
    return _default_model_for_task(task, available)


def _task_run_dir(task: str) -> str:
    return _TASK_RUN_DIRS.get(task, 'detect')


_webrtc_loop = asyncio.new_event_loop()
_webrtc_thread = threading.Thread(target=_webrtc_loop.run_forever, daemon=True)
_webrtc_thread.start()


def _run_async(coro):
    """Execute a coroutine on the dedicated WebRTC event loop and return its result."""

    if _webrtc_loop.is_closed():
        raise RuntimeError("WebRTC event loop already closed")

    future = asyncio.run_coroutine_threadsafe(coro, _webrtc_loop)
    return future.result()

# Ensure uploads folder exists
os.makedirs(os.path.join(os.path.dirname(__file__), 'uploads'), exist_ok=True)


@app.route("/")
def hello_world():
    """Render main dashboard page."""
    task = _normalize_task(request.args.get('task'))
    models_for_task = _models_for_task(task)
    selected_model = _select_model_for_task(request.args.get('model'), task, models_for_task)
    return render_template(
        'index.html',
        models=models_for_task,
        selected_model=selected_model,
        selected_task=task,
    )

    
@app.route("/", methods=["GET", "POST"])
def predict_img():
    """
    Handle image and video upload, process with selected YOLO model.
    
    Returns:
        Rendered template with detection results
    """
    if request.method == "POST":
        task = _normalize_task(request.form.get('task'))
        models_for_task = _models_for_task(task)
        default_model = _default_model_for_task(task, models_for_task)

        uploaded = request.files.get('file')
        model_value = request.form.get('model')
        model_name = model_value.strip() if model_value else default_model

        if uploaded is None or uploaded.filename == '':
            message = "No file selected. Please choose an image or video file."
            selected_model = _select_model_for_task(model_name, task, models_for_task)
            return render_template(
                'index.html',
                image_url='',
                video_present=False,
                selected_model=selected_model,
                message=message,
                models=models_for_task,
                selected_task=task,
            )

        if not is_image_file(uploaded.filename) and not is_video_file(uploaded.filename):
            message = ("Unsupported file format! Only these formats are accepted:<br>"
                       "<strong>Images:</strong> PNG, JPG, JPEG, BMP, GIF, WebP<br>"
                       "<strong>Videos:</strong> MP4, AVI, MOV, MKV, FLV, WMV")
            selected_model = _select_model_for_task(model_name, task, models_for_task)
            return render_template(
                'index.html',
                image_url='',
                video_present=False,
                selected_model=selected_model,
                message=message,
                models=models_for_task,
                selected_task=task,
            )

        basepath = os.path.dirname(__file__)
        uploads_dir = os.path.join(basepath, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        safe_name = secure_filename(uploaded.filename)
        filepath = os.path.join(uploads_dir, safe_name)
        uploaded.save(filepath)

        # Store filename globally for reference (use secure name)
        global imgpath
        predict_img.imgpath = safe_name

        if model_name not in models_for_task:
            message = (f"Selected model '{model_name}' is not available. "
                       "Please upload the model or choose one from the dropdown.")
            if os.path.exists(filepath):
                os.remove(filepath)
            selected_model = _select_model_for_task(model_name, task, models_for_task)
            return render_template(
                'index.html',
                image_url='',
                video_present=False,
                selected_model=selected_model,
                message=message,
                models=models_for_task,
                selected_task=task,
            )

        try:
            if is_image_file(uploaded.filename):
                latest_subfolder, processed_filename = process_image(filepath, model_name, task=task)
                image_url = url_for('display', folder=latest_subfolder, filename=processed_filename, task=task)
                return render_template(
                    'index.html',
                    image_url=image_url,
                    video_present=False,
                    selected_model=model_name,
                    models=models_for_task,
                    selected_task=task,
                )

            if is_video_file(uploaded.filename):
                # Return a streaming URL for real-time video processing
                stream_url = url_for('stream_video_file', filename=safe_name, model=model_name, task=task)
                return render_template(
                    'index.html',
                    image_url='',
                    video_present=False,
                    video_stream_url=stream_url,
                    selected_model=model_name,
                    models=models_for_task,
                    selected_task=task,
                )
        except Exception as e:
            message = f"Error processing file with model '{model_name}': {str(e)}"
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception:
                    pass
            return render_template(
                'index.html',
                image_url='',
                video_present=False,
                selected_model=model_name,
                message=message,
                models=models_for_task,
                selected_task=task,
            )

    # Default GET response
    task = _normalize_task(request.args.get('task'))
    models_for_task = _models_for_task(task)
    selected_model = _select_model_for_task(request.args.get('model'), task, models_for_task)
    # Ensure the client receives ICE servers reliably
    ice_servers = None
    try:
        ice_servers = get_ice_servers_for_client()
    except Exception:
        ice_servers = None

    if not ice_servers:
        ice_file = os.environ.get('ICE_SERVERS_FILE', '')
        if ice_file and os.path.exists(ice_file):
            try:
                with open(ice_file, 'r') as f:
                    ice_servers = json.loads(f.read())
            except Exception:
                ice_servers = None

    if not ice_servers:
        ice_servers = [{"urls": "stun:stun.l.google.com:19302"}]

    try:
        app.logger.info('[ICE] Sending %d ICE server entries to client', len(ice_servers) if isinstance(ice_servers, list) else -1)
        if isinstance(ice_servers, list) and ice_servers:
            app.logger.info('[ICE] First entry urls=%s', ice_servers[0].get('urls'))
    except Exception:
        pass

    return render_template('index.html', selected_model=selected_model, models=models_for_task, selected_task=task, ice_servers=ice_servers)


@app.route('/upload_model', methods=['POST'])
def upload_model():
    """
    Upload a custom .pt model, load it temporarily, register in memory, and make it available for inference.
    This writes the uploaded file to a secure temporary file only while loading and deletes it immediately.
    """
    task = _normalize_task(request.form.get('task') or request.args.get('task'))
    models_for_task = _models_for_task(task)
    default_model = _default_model_for_task(task, models_for_task)
    if 'model_file' not in request.files:
        return render_template('index.html', message='No model file provided.', models=models_for_task,
                       selected_model=default_model, selected_task=task)

    f = request.files['model_file']
    display_name = request.form.get('model_name') or f.filename
    disclaimer = request.form.get('disclaimer')

    # Require disclaimer checkbox
    if not disclaimer:
        return render_template('index.html', message='You must accept the disclaimer before uploading a model.',
                       models=models_for_task, selected_model=default_model, selected_task=task)

    if f.filename == '':
        return render_template('index.html', message='No file selected.', models=models_for_task,
                       selected_model=default_model, selected_task=task)

    # Only accept .pt files for now
    if '.' not in f.filename or f.filename.rsplit('.', 1)[1].lower() != 'pt':
        return render_template('index.html', message='Only .pt model files are accepted.', models=models_for_task,
                       selected_model=default_model, selected_task=task)

    # Enforce size limit (default 200MB)
    data = f.read()
    max_size = 200 * 1024 * 1024
    if len(data) > max_size:
        return render_template('index.html', message='Model file too large (max 200 MB).', models=models_for_task,
                       selected_model=default_model, selected_task=task)

    tmp_path = None
    try:
        # Persist uploaded .pt into the repo models/ folder and register by filename
        basepath = os.path.dirname(__file__)
        models_dir = os.path.join(basepath, 'models')
        os.makedirs(models_dir, exist_ok=True)
        saved_name = secure_filename(f.filename)
        saved_path = os.path.join(models_dir, saved_name)

        # Write file to models/ (overwrite if present)
        with open(saved_path, 'wb') as out_f:
            out_f.write(data)

        # Load with Ultralytics YOLO to ensure compatibility with pipeline
        from ultralytics import YOLO
        model_obj = YOLO(saved_path)

        # Register the model in memory under the saved filename (use filename as the selector)
        register_custom_model(saved_name, model_obj)

    except Exception as e:
        # If something failed, remove saved file if it exists
        try:
            if 'saved_path' in locals() and os.path.exists(saved_path):
                os.remove(saved_path)
        except Exception:
            pass
        return render_template('index.html', message=f'Failed to load model: {e}', models=models_for_task,
                               selected_model=default_model, selected_task=task)

    return render_template('index.html', message=f"Model '{saved_name}' uploaded and registered.",
                           models=models_for_task, selected_model=saved_name, selected_task=task)


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
    task = _normalize_task(request.args.get('task'))
    base_dir = _task_run_dir(task)
    base = os.path.join(os.getcwd(), 'runs', base_dir)
    directory = os.path.join(base, folder)
    if not os.path.isdir(directory):
        return "Not found", 404
    return send_from_directory(directory, filename)


@app.route('/stream_video_file')
def stream_video_file():
    """
    Stream an uploaded video file with real-time YOLO processing.
    Processes frames on-the-fly and streams them back as MP4.
    
    Query parameters:
      - filename: name of uploaded video file
      - model: model name to use for inference
      - task: detection/segmentation/pose task
    """
    filename = request.args.get('filename')
    model_name = request.args.get('model', 'yolov8n.pt').strip()
    task = _normalize_task(request.args.get('task'))
    
    if not filename:
        return "Missing filename", 400
    
    # Get the uploaded file path
    uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    filepath = os.path.join(uploads_dir, secure_filename(filename))
    
    if not os.path.isfile(filepath):
        return "File not found", 404
    
    def generate_video_stream():
        """Generator that yields MP4 frames with YOLO processing."""
        try:
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                return
            
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            # Get model
            try:
                model = get_model(model_name)
            except Exception as e:
                print(f"Failed to load model {model_name}: {e}")
                return
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                try:
                    # Run inference
                    results = model(frame, verbose=False)
                    annotated = annotate_frame(frame, results[0], task=task)
                    
                    # Add model overlay
                    try:
                        from backend.image_video_processor import add_model_overlay
                        annotated = add_model_overlay(annotated, model_name)
                    except Exception:
                        pass
                    
                    # Encode frame as JPEG
                    ret2, jpeg = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if not ret2:
                        continue
                    
                    frame_bytes = jpeg.tobytes()
                    
                    # Yield MJPEG frame boundary
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n'
                           b'Content-length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' + frame_bytes + b'\r\n')
                    
                    # Log progress
                    if frame_count % 30 == 0:
                        print(f"[STREAM] Processing frame {frame_count} with {model_name}")
                    
                except Exception as e:
                    print(f"Frame processing error: {e}")
                    continue
        
        except Exception as e:
            print(f"stream_video_file error: {e}")
        finally:
            try:
                cap.release()
            except Exception:
                pass
    
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/webrtc/offer', methods=['POST'])
def webrtc_offer() -> Response:
    """Handle WebRTC offer from the browser and return an answer SDP."""

    payload = request.get_json(silent=True) or {}
    sdp = payload.get('sdp')
    offer_type = payload.get('type')
    task = _normalize_task(payload.get('task'))
    models_for_task = _models_for_task(task)
    model_name_value = payload.get('model')
    if model_name_value:
        model_name = model_name_value.strip()
    else:
        model_name = _default_model_for_task(task, models_for_task)

    if not sdp or not offer_type:
        return jsonify({'error': 'Invalid SDP payload'}), 400

    if model_name not in models_for_task:
        return jsonify({'error': f"Model '{model_name}' is not available for {task}."}), 400

    # optional scale parameter from client (1.0 == native)
    try:
        scale = float(payload.get('scale', 1.0) or 1.0)
    except Exception:
        scale = 1.0

    try:
        app.logger.info('Received WebRTC offer for model %s (%s) scale=%s from %s', model_name, task, scale, request.remote_addr)
        # optional camera_id to select which camera stream this offer corresponds to
        camera_id = payload.get('camera_id', '0')
        answer = _run_async(create_answer_for_offer(sdp, offer_type, model_name, task, scale, camera_id))
    except ValueError as err:
        return jsonify({'error': str(err)}), 400
    except Exception as err:  # pragma: no cover - defensive logging path
        app.logger.exception('WebRTC offer handling failed: {0}'.format(err))
        return jsonify({'error': 'Failed to negotiate WebRTC session'}), 500

    return jsonify({'sdp': answer.sdp, 'type': answer.type})


@app.route('/list_cameras')
def list_cameras():
    """Return a JSON list of indices for cameras that can be opened (probes 0..4)."""
    import json
    found = []
    for i in range(0, 5):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap is not None and cap.isOpened():
                found.append(i)
                cap.release()
            else:
                try:
                    cap.release()
                except Exception:
                    pass
        except Exception:
            pass
    return Response(json.dumps({'cameras': found}), mimetype='application/json')


@atexit.register
def _cleanup_webrtc() -> None:
    try:
        _run_async(close_all_peers())
    except Exception:
        pass
    finally:
        try:
            _webrtc_loop.call_soon_threadsafe(_webrtc_loop.stop)
        except Exception:
            pass
        try:
            if _webrtc_thread.is_alive():
                _webrtc_thread.join(timeout=1)
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask YOLO object detection web application")
    parser.add_argument("--port", default=5000, type=int, help="Port number for Flask server")
    args = parser.parse_args()
    
    # Preload default model into cache for faster first request
    preload_default_model()
    
    app.run(host="0.0.0.0", port=args.port)
