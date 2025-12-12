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
    is_image_file,
    is_video_file,
    annotate_frame,
)
from backend.single_camera import get_webcam_frame
from backend.dual_camera import get_dual_webcam_frame
from backend.video_stream import get_frame, get_video_from_folder, stream_file_mp4_as_mjpeg
from backend.webrtc_session import create_answer_for_offer, close_all_peers


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
                stream_url = url_for('process_video_stream', filename=safe_name, model=model_name, task=task)
                return render_template(
                    'index.html',
                    image_url='',
                    video_present=False,
                    server_video_stream_url=stream_url,
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
    return render_template('index.html', selected_model=selected_model, models=models_for_task, selected_task=task)


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


@app.route('/process_video_stream')
def process_video_stream():
    """
    Process an uploaded video file frame-by-frame and stream processed frames as MJPEG.
    Query params:
      - filename: name of file inside uploads/ (secure_filename recommended)
      - model: model name to use
    """
    filename = request.args.get('filename')
    task = _normalize_task(request.args.get('task'))
    models_for_task = _models_for_task(task)
    model_name = request.args.get('model', _default_model_for_task(task, models_for_task))
    if model_name:
        model_name = model_name.strip()
    if not filename:
        return "Missing filename", 400

    allowed_models = models_for_task
    if model_name not in allowed_models:
        choices = ', '.join(allowed_models) if allowed_models else 'None'
        msg = (f"Model '{model_name}' is not available for {task}. "
               f"Choose from: {choices}")
        return (msg, 400)

    uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    filepath = os.path.join(uploads_dir, filename)
    if not os.path.exists(filepath):
        return "File not found", 404

    # Extra debug output to trace incoming request parameters
    try:
        print('\n--- process_video_stream START ---')
        print('Request URL:', request.url)
        print('Query string raw:', request.query_string)
        print('request.args:', dict(request.args))
        try:
            from backend import model_manager
            print('model_manager.model_cache keys:', list(model_manager.model_cache.keys()))
        except Exception:
            pass
        print('Available models (list_models):', list_models())
        print('-------------------------------\n')
    except Exception:
        pass

    # Resolve the requested model BEFORE starting the stream. If the requested model is not available,
    # return a clear error so the user can pick a different model.
    resolved_model = None
    resolved_model_name = None
    try:
        # Try to resolve via get_model (handles built-ins and registered custom models)
        try:
            resolved_model = get_model(model_name)
            resolved_model_name = model_name
            print(f'Using model (get_model): {model_name}')
        except Exception as e:
            print(f'get_model failed for "{model_name}": {e}')
            try:
                from backend import model_manager
                if model_name in model_manager.model_cache:
                    resolved_model = model_manager.model_cache[model_name]
                    resolved_model_name = model_name
                    print(f'Using model from model_cache: {model_name}')
            except Exception:
                pass

        # Try loading a local model file from models/
        if resolved_model is None:
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
            candidate = os.path.join(models_dir, model_name)
            if os.path.isfile(candidate):
                try:
                    from ultralytics import YOLO as _YOLO
                    resolved_model = _YOLO(candidate)
                    resolved_model_name = os.path.basename(candidate)
                    print(f'Loaded local model file: {candidate}')
                except Exception as e:
                    print('local model load failed:', e)

        if resolved_model is None:
            available = []
            try:
                from backend import model_manager
                available = list(model_manager.model_cache.keys())
            except Exception:
                pass
            msg = f"Requested model '{model_name}' not available on server. Available: {available}"
            print(msg)
            return (msg, 400)
    except Exception as e:
        print('model resolution unexpected error:', e)
        return (f"Model resolution error: {e}", 500)

    def generate():
        try:
            cap = cv2.VideoCapture(filepath)
            # Debug info: log incoming model and available cached models
            try:
                from backend import model_manager
                print('process_video_stream request -> filename:', filename, 'model:', model_name)
                print('model_cache keys:', list(model_manager.model_cache.keys()))
                models_dir = os.path.join(os.path.dirname(__file__), 'models')
                if os.path.isdir(models_dir):
                    print('models/ dir contents:', os.listdir(models_dir))
            except Exception:
                pass

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                t_frame = time.perf_counter()

                # Run inference on the frame using the resolved model
                try:
                    t_inf0 = time.perf_counter()
                    results = resolved_model(frame)
                    t_inf1 = time.perf_counter()
                    res_plotted = annotate_frame(frame, results[0], task=task)
                    t_ann = time.perf_counter()
                except Exception as e:
                    print('inference error:', e)
                    res_plotted = frame
                    t_inf1 = time.perf_counter()
                    t_ann = t_inf1

                # Optionally add model overlay using image_video_processor.add_model_overlay if available
                try:
                    from backend.image_video_processor import add_model_overlay
                    res_plotted = add_model_overlay(res_plotted, model_name)
                    t_overlay = time.perf_counter()
                except Exception:
                    t_overlay = time.perf_counter()

                # Add a small debug label showing which model was actually used
                try:
                    label = resolved_model_name if resolved_model_name else ('unknown')
                    cv2.putText(res_plotted, f'Model used: {label}', (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                except Exception:
                    pass

                # Encode as JPEG and yield as MJPEG frame
                t_enc0 = time.perf_counter()
                ret2, jpeg = cv2.imencode('.jpg', res_plotted, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                t_enc1 = time.perf_counter()
                if not ret2:
                    continue
                frame_bytes = jpeg.tobytes()

                try:
                    inf_ms = (t_inf1 - t_inf0) * 1000.0
                    ann_ms = (t_ann - t_inf1) * 1000.0
                    overlay_ms = (t_overlay - t_ann) * 1000.0
                    enc_ms = (t_enc1 - t_enc0) * 1000.0
                    total_ms = (t_enc1 - t_frame) * 1000.0
                    print(f"[STREAM][TIMING] model={model_name} task={task} inf_ms={inf_ms:.1f} ann_ms={ann_ms:.1f} overlay_ms={overlay_ms:.1f} enc_ms={enc_ms:.1f} total_ms={total_ms:.1f}")
                except Exception:
                    pass

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except GeneratorExit:
            # client disconnected
            pass
        except Exception as e:
            try:
                print('process_video_stream error:', e)
            except Exception:
                pass
        finally:
            try:
                cap.release()
            except Exception:
                pass

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/video_feed")
def video_feed():
    """
    Stream processed video as MJPEG.
    Accepts ?folder=<name> parameter to stream from specific runs/detect subfolder.
    """
    folder = request.args.get('folder', None)
    task = _normalize_task(request.args.get('task'))
    task_dir = _task_run_dir(task)
    if folder:
        mp4_path = get_video_from_folder(folder, task_dir=task_dir)
        if mp4_path:
            return Response(stream_file_mp4_as_mjpeg(mp4_path), 
                          mimetype='multipart/x-mixed-replace; boundary=frame')
    
    # Fallback to default output.mp4
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/webcam')
def webcam_page():
    """Render webcam page with live inference and stop button."""
    task = _normalize_task(request.args.get('task'))
    models_for_task = _models_for_task(task)
    default_model = _default_model_for_task(task, models_for_task)
    return render_template(
        'webcam.html',
        models=models_for_task,
        default_model=default_model,
        selected_task=task,
    )


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
        answer = _run_async(create_answer_for_offer(sdp, offer_type, model_name, task, scale))
    except ValueError as err:
        return jsonify({'error': str(err)}), 400
    except Exception as err:  # pragma: no cover - defensive logging path
        app.logger.exception('WebRTC offer handling failed: {0}'.format(err))
        return jsonify({'error': 'Failed to negotiate WebRTC session'}), 500

    return jsonify({'sdp': answer.sdp, 'type': answer.type})


@app.route('/webcam_feed')
def webcam_feed():
    """
    Stream single webcam MJPEG feed with YOLO inference.
    Accepts ?model=<model_name> parameter to select model.
    """
    task = _normalize_task(request.args.get('task'))
    models_for_task = _models_for_task(task)
    model_name = request.args.get('model', _default_model_for_task(task, models_for_task))
    if model_name not in models_for_task:
        return Response(f"Model '{model_name}' is not available for {task}.", status=400)
    cam = int(request.args.get('cam', 0)) if request.args.get('cam') is not None else 0
    return Response(get_webcam_frame(model_name, camera_index=cam, task=task),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/webcam_feed_dual')
def webcam_feed_dual():
    """
    Stream dual webcam MJPEG feed with YOLO inference.
    Accepts ?model0=<model_name> and ?model1=<model_name> parameters.
    """
    task = _normalize_task(request.args.get('task'))
    allowed = _models_for_task(task)
    default_model = _default_model_for_task(task, allowed)
    model0 = request.args.get('model0', default_model)
    model1 = request.args.get('model1', default_model)
    if model0 not in allowed or model1 not in allowed:
        return Response('Requested model not available for selected task.', status=400)
    return Response(get_dual_webcam_frame(model0, model1, task=task),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/webcam_feed_split')
def webcam_feed_split():
    """
    Stream single-camera feed processed by two different models and combined vertically.
    Accepts query params: ?model0=<m0>&model1=<m1>
    """
    task = _normalize_task(request.args.get('task'))
    allowed = _models_for_task(task)
    default_model = _default_model_for_task(task, allowed)
    model0 = request.args.get('model0', default_model)
    model1 = request.args.get('model1', default_model)
    if model0 not in allowed or model1 not in allowed:
        return Response('Requested model not available for selected task.', status=400)
    cam = int(request.args.get('cam', 0)) if request.args.get('cam') is not None else 0
    # Lazy import of the split generator from single_camera
    from backend.single_camera import get_split_webcam_frame
    return Response(get_split_webcam_frame(model0, model1, camera_index=cam, task=task),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


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
