import multiprocessing as mp
import time
import traceback
from typing import Optional, Dict, Any

import cv2
import numpy as np

import os
from backend.model_manager import get_model
from backend.image_video_processor import annotate_frame, add_model_overlay


def _encode_jpeg(frame, quality: int = 85) -> Optional[bytes]:
    try:
        ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ret:
            return None
        return buf.tobytes()
    except Exception:
        return None


def _decode_jpeg(jpeg_bytes: bytes):
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _worker_main(in_q: mp.Queue, out_q: mp.Queue, ctrl_q: mp.Queue, model_name: str, task: str, cfg: Dict[str, Any]):
    """Process loop that runs in a separate process.

    Receives dict messages with keys: session_id, frame_id, camera, jpeg, ts
    Sends back dict messages with keys: session_id, frame_id, camera, jpeg, timings
    """
    worker_id = os.getpid()
    print(f"[WORKER {worker_id}] Starting worker for model={model_name}, task={task}")
    
    # Force CPU-only mode for worker process to avoid GPU dependency
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    except Exception:
        pass
    
    try:
        model = get_model(model_name)
        print(f"[WORKER {worker_id}] Model {model_name} loaded successfully")
    except Exception as e:
        print(f"[WORKER {worker_id}] ERROR: failed to load model {model_name}: {e}")
        out_q.put({'error': f'failed to load model {model_name}: {e}'})
        return

    heartbeat_ts = time.time()
    hb_interval = cfg.get('heartbeat_sec', 5)
    frame_count = 0
    # Debugging flags
    WORKER_DEBUG = os.environ.get('WORKER_DEBUG', '0') == '1'
    WORKER_DEBUG_EVERY = int(os.environ.get('WORKER_DEBUG_EVERY', 30))

    while True:
        # control check
        try:
            ctrl = ctrl_q.get_nowait()
            if ctrl == 'shutdown':
                break
        except Exception:
            pass

        try:
            msg = in_q.get(timeout=1.0)
        except Exception:
            # periodic heartbeat
            if time.time() - heartbeat_ts > hb_interval:
                try:
                    out_q.put({'_hb': True, 'ts': time.time()})
                except Exception:
                    pass
                heartbeat_ts = time.time()
            continue

        if msg is None:
            continue

        frame_count += 1
        session_id = msg.get('session_id', 'unknown')
        camera = msg.get('camera', '?')
        frame_id = msg.get('frame_id', -1)
        msg_perf_ts = msg.get('perf_ts', None)  # Use perf_counter timestamp from submission
        t_dequeue = time.perf_counter()
        queue_wait_ms = (t_dequeue - msg_perf_ts) * 1000.0 if msg_perf_ts is not None else 0

        try:
            jpeg = msg.get('jpeg')
            t_decode_start = time.perf_counter()
            frame = _decode_jpeg(jpeg) if jpeg is not None else None
            t_decode_end = time.perf_counter()
            
            t0 = time.perf_counter()  # inference start
            results = model(frame, verbose=False)
            t1 = time.perf_counter()  # inference end
            
            annotated = annotate_frame(frame, results[0], task=task)
            t2 = time.perf_counter()
            try:
                annotated = add_model_overlay(annotated, model_name)
            except Exception:
                pass
            t3 = time.perf_counter()
            out_jpeg = _encode_jpeg(annotated, quality=cfg.get('jpeg_quality', 85))
            t_encode_end = time.perf_counter()

            inf_ms = (t1 - t0) * 1000.0
            ann_ms = (t2 - t1) * 1000.0
            overlay_ms = (t3 - t2) * 1000.0
            decode_ms = (t_decode_end - t_decode_start) * 1000.0
            encode_ms = (t_encode_end - t3) * 1000.0
            total_ms = queue_wait_ms + decode_ms + inf_ms + ann_ms + overlay_ms + encode_ms
            
            # Log periodically; when WORKER_DEBUG=1, log more often per WORKER_DEBUG_EVERY
            if WORKER_DEBUG:
                if frame_count % max(1, WORKER_DEBUG_EVERY) == 0:
                    print(f"[WORKER {worker_id}][DEBUG] cam={camera} frame={frame_id} queue_wait={queue_wait_ms:.1f}ms decode={decode_ms:.1f}ms inf={inf_ms:.1f}ms ann={ann_ms:.1f}ms overlay={overlay_ms:.1f}ms encode={encode_ms:.1f}ms TOTAL={total_ms:.1f}ms")
            else:
                if frame_count % 30 == 0:  # default periodic log to avoid spam
                    print(f"[WORKER {worker_id}] cam={camera} frame={frame_id} queue_wait={queue_wait_ms:.1f}ms decode={decode_ms:.1f}ms inf={inf_ms:.1f}ms ann={ann_ms:.1f}ms overlay={overlay_ms:.1f}ms encode={encode_ms:.1f}ms TOTAL={total_ms:.1f}ms")

            out_q.put({
                'session_id': session_id,
                'camera': camera,
                'frame_id': frame_id,
                'jpeg': out_jpeg,
                'timings': {
                    'queue_wait_ms': queue_wait_ms,
                    'decode_ms': decode_ms,
                    'inf_ms': inf_ms,
                    'ann_ms': ann_ms,
                    'overlay_ms': overlay_ms,
                    'encode_ms': encode_ms,
                    'total_ms': total_ms,
                },
            })
        except Exception:
            print(f"[WORKER {worker_id}] ERROR processing frame cam={camera} id={frame_id}: {traceback.format_exc()}")
            out_q.put({'error': f'frame processing error: {traceback.format_exc()}'})


class WorkerProcess:
    """Simple wrapper around a multiprocessing worker process."""

    def __init__(self, model_name: str, task: str = 'detection', cfg: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.task = task
        self.cfg = cfg or {}
        # Queues for input / output / control
        # Increased queue sizes: 6 input, 12 output to reduce backpressure
        self._in_q: mp.Queue = mp.Queue(maxsize=self.cfg.get('queue_size', 6))
        self._out_q: mp.Queue = mp.Queue(maxsize=self.cfg.get('queue_size', 12))
        self._ctrl_q: mp.Queue = mp.Queue()
        self._proc: Optional[mp.Process] = None
        # Ensure spawn start method on platforms where needed
        try:
            mp.set_start_method('spawn', force=False)
        except RuntimeError:
            pass

        self._start_process()

    def _start_process(self):
        if self._proc and self._proc.is_alive():
            return
        self._proc = mp.Process(target=_worker_main, args=(self._in_q, self._out_q, self._ctrl_q, self.model_name, self.task, self.cfg), daemon=True)
        self._proc.start()

    def submit_frame(self, session_id: str, camera: str, frame_id: int, frame_ndarray) -> bool:
        """Encode frame to jpeg and submit to worker queue. Returns True if enqueued."""
        try:
            jpeg = _encode_jpeg(frame_ndarray, quality=self.cfg.get('jpeg_quality', 85))
            if jpeg is None:
                return False
            msg = {'session_id': session_id, 'camera': camera, 'frame_id': frame_id, 'jpeg': jpeg, 'perf_ts': time.perf_counter()}
            try:
                self._in_q.put_nowait(msg)
                return True
            except Exception:
                # queue full: drop the oldest frame (drop-oldest policy maintains freshness)
                try:
                    old_msg = self._in_q.get_nowait()
                    old_frame = old_msg.get('frame_id', -1)
                    old_cam = old_msg.get('camera', '?')
                    print(f"[WORKER POOL] Queue full for camera {camera}; dropped old frame {old_frame} from camera {old_cam}")
                except Exception:
                    pass
                try:
                    self._in_q.put_nowait(msg)
                    return True
                except Exception:
                    return False
        except Exception:
            return False

    def get_result(self, timeout: float = 0.5):
        """Blocking get from output queue with timeout (seconds)."""
        try:
            return self._out_q.get(timeout=timeout)
        except Exception:
            return None

    def stop(self):
        try:
            self._ctrl_q.put_nowait('shutdown')
        except Exception:
            pass
        try:
            if self._proc:
                self._proc.join(timeout=1.0)
        except Exception:
            pass
