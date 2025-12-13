import asyncio
import time
import cv2
import os
import numpy as np

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from aiortc.mediastreams import MediaStreamTrack
from av import VideoFrame

from backend.model_manager import get_model
from backend.image_video_processor import add_model_overlay, annotate_frame
from backend.worker_pool import WorkerProcess

# Worker defaults
WORKER_CFG = {
    'queue_size': int(os.environ.get('DUAL_WORKER_QUEUE_SIZE', 6)),
    'jpeg_quality': int(os.environ.get('DUAL_WORKER_JPEG_QUALITY', 80)),
    'heartbeat_sec': int(os.environ.get('DUAL_WORKER_HEARTBEAT_SEC', 5)),
}
# Track-level debug / concurrency
MAX_INFLIGHT = int(os.environ.get('DUAL_WORKER_MAX_INFLIGHT', 1))  # default 1 prevents queue piling; increase to pipeline
TRACK_DEBUG = os.environ.get('TRACK_DEBUG', '0') == '1'
TRACK_DEBUG_EVERY = int(os.environ.get('TRACK_DEBUG_EVERY', 30))


relay = MediaRelay()
peer_connections: set[RTCPeerConnection] = set()


class YOLOTransformTrack(MediaStreamTrack):
    """Media stream track that runs YOLO inference on incoming frames.

    Uses a single multiprocessing worker per track to keep CPU work off the event loop.
    """

    kind = "video"

    def __init__(self, track: MediaStreamTrack, model_name: str, task: str = 'detection', model=None, scale: float = 0.5, camera_id: str = '0'):
        super().__init__()  # initializes timestamp-related state
        self.track = track
        self.model_name = model_name
        self.task = task
        self.camera_id = camera_id
        # Requested capture/process scale (1.0 == native). If <1.0 we will downscale frames before inference.
        self.scale = float(scale or 1.0)
        # Fetch (or load) the model once; model_manager caches subsequent requests
        self.model = model or get_model(model_name)
        # Multiprocessing worker for this track (one process per track / camera)
        try:
            self.worker = WorkerProcess(self.model_name, task=self.task, cfg=WORKER_CFG)
        except Exception as e:
            print(f"[WebRTC] failed to start worker process for {self.model_name}: {e}")
            self.worker = None
        self._frame_counter = 0
        # Prevent submitting multiple frames while a worker is still processing
        self._inflight = False
        print(f"[WebRTC] YOLOTransformTrack initialized for camera={camera_id}, model={model_name}, task={task}")

    async def recv(self) -> VideoFrame:
        frame = await self.track.recv()
        t_recv = time.perf_counter()
        image = frame.to_ndarray(format="bgr24")
        orig_h, orig_w = image.shape[:2]

        annotated = None
        timing = {}
        # Try asynchronous worker processing (multiprocessing)
        if getattr(self, 'worker', None) is not None:
            try:
                # If a previous frame is still inflight, skip submitting this one to avoid queue buildup
                # Allow up to MAX_INFLIGHT concurrent frames in the pipeline
                if getattr(self, '_inflight', 0) >= MAX_INFLIGHT:
                    if self._frame_counter % max(1, TRACK_DEBUG_EVERY) == 0 and TRACK_DEBUG:
                        print(f"[WebRTC] Camera {self.camera_id} skipping frame because {self._inflight} inflight >= {MAX_INFLIGHT}")
                    annotated = None
                else:
                    self._frame_counter += 1
                    t_submit = time.perf_counter()
                    # Optionally downscale before submitting to worker so capture scale affects CPU work
                    proc_image = image
                    was_downscaled = False
                    try:
                        if self.scale and self.scale < 0.99:
                            ih, iw = image.shape[:2]
                            nw = max(2, int(round(iw * self.scale)))
                            nh = max(2, int(round(ih * self.scale)))
                            if nw != iw or nh != ih:
                                proc_image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
                                was_downscaled = True
                    except Exception:
                        proc_image = image

                    # submit frame to worker (non-blocking)
                    enqueued = self.worker.submit_frame(session_id=str(id(self)), camera=self.camera_id, frame_id=self._frame_counter, frame_ndarray=proc_image)
                    t_after_submit = time.perf_counter()
                    
                    if enqueued:
                        # increment inflight counter so we respect concurrency limit
                        self._inflight = getattr(self, '_inflight', 0) + 1
                        loop = asyncio.get_running_loop()
                        # wait for result (blocking call executed in thread-pool)
                        # timeout configurable in WORKER_CFG; use 2.0s as a safe default
                        res = await loop.run_in_executor(None, self.worker.get_result, 2.0)
                        # decrement inflight regardless of outcome so next frame can be submitted
                        try:
                            self._inflight = max(0, getattr(self, '_inflight', 1) - 1)
                        except Exception:
                            self._inflight = 0
                        
                        if res and isinstance(res, dict) and res.get('jpeg'):
                            try:
                                arr = np.frombuffer(res['jpeg'], dtype=np.uint8)
                                decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                                if decoded is not None:
                                    # If we downscaled before sending to worker, upscale annotated result back
                                    if was_downscaled and orig_h and orig_w:
                                        try:
                                            decoded = cv2.resize(decoded, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                                        except Exception:
                                            pass
                                    annotated = decoded
                                    timing = res.get('timings', {})
                                    # Log timing info periodically when debug enabled
                                    if TRACK_DEBUG and (self._frame_counter % max(1, TRACK_DEBUG_EVERY) == 0):
                                        worker_queue_ms = timing.get('queue_wait_ms', 0)
                                        worker_inf_ms = timing.get('inf_ms', 0)
                                        print(f"[WebRTC][DEBUG] Camera {self.camera_id} frame {self._frame_counter}: recv_frame={frame.pts} worker_queue={worker_queue_ms:.1f}ms worker_inference={worker_inf_ms:.1f}ms")
                            except Exception as de:
                                print(f"[WebRTC] Camera {self.camera_id} failed to decode worker result: {de}")
                        else:
                            if self._frame_counter % 30 == 0:
                                print(f"[WebRTC] Camera {self.camera_id} frame {self._frame_counter}: no result from worker (timeout or error)")
                    else:
                        # queue full or not accepted
                        if self._frame_counter % 30 == 0:
                            print(f"[WebRTC] Camera {self.camera_id} frame {self._frame_counter}: worker queue full, enqueue failed")
                        annotated = None
            except Exception as err:
                # ensure inflight flag cleared on unexpected exception
                self._inflight = False
                print(f"[WebRTC] Camera {self.camera_id} worker processing failed for model {self.model_name}: {err}")
                annotated = None

        # Fallback synchronous processing
        if annotated is None:
            try:
                loop = asyncio.get_running_loop()
                annotated, timing = await loop.run_in_executor(None, self._run_inference, image, t_recv, orig_h, orig_w)
            except Exception as err:
                print(f"[WebRTC] YOLO inference failed for model {self.model_name}: {err}")
                annotated = image
                timing = {}

        # time converting annotated image back to VideoFrame (encode/send preparation)
        t_encode_start = time.perf_counter()
        new_frame = VideoFrame.from_ndarray(annotated, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        t_return = time.perf_counter()

        return new_frame

    def _run_inference(self, image, t_recv=None, orig_h=None, orig_w=None):
        # t_recv: perf_counter timestamp when frame was received in recv()
        # orig_h, orig_w: original frame dimensions (before downscaling)
        # Log incoming frame resolution for diagnostics
        try:
            h, w = (image.shape[0], image.shape[1]) if image is not None else (0, 0)
            print(f"[WebRTC][FRAME] model={self.model_name} task={self.task} width={w} height={h}")
        except Exception:
            pass

        t0 = time.perf_counter()  # processing start

        # Optionally downscale image for faster processing if a scale < 1.0 was requested
        proc_image = image
        was_downscaled = False
        try:
            if self.scale and self.scale < 0.99:
                ih, iw = image.shape[:2]
                nw = max(2, int(round(iw * self.scale)))
                nh = max(2, int(round(ih * self.scale)))
                if nw != iw or nh != ih:
                    proc_image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
                    was_downscaled = True
                    print(f"[WebRTC][SCALE] model={self.model_name} requested_scale={self.scale:.2f} orig_w={iw} orig_h={ih} proc_w={nw} proc_h={nh}")
        except Exception:
            proc_image = image

        results = self.model.predict(proc_image, verbose=False)
        t1 = time.perf_counter()

        boxes = getattr(results[0], "boxes", None)
        count = len(boxes) if boxes is not None else 0
        print(f"[WebRTC] {self.model_name} detections: {count}")

        try:
            # annotate using the processed image and prediction
            annotated = annotate_frame(proc_image, results[0], task=self.task)
        except Exception as ann_err:
            print(f"[WebRTC] annotate_frame failed: {ann_err}")
            annotated = proc_image
        t2 = time.perf_counter()

        try:
            annotated = add_model_overlay(annotated, self.model_name)
        except Exception as overlay_err:
            print(f"[WebRTC] failed to add overlay: {overlay_err}")
        t3 = time.perf_counter()

        # If we downscaled before inference, upscale the annotated result back to original size
        # so the browser displays at full resolution with crisp overlays
        if was_downscaled and orig_h and orig_w:
            try:
                proc_h, proc_w = annotated.shape[:2]
                if (proc_h, proc_w) != (orig_h, orig_w):
                    annotated = cv2.resize(annotated, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    print(f"[WebRTC][UPSCALE] upscaled from {proc_w}x{proc_h} to {orig_w}x{orig_h}")
            except Exception as e:
                print(f"[WebRTC] upscale failed: {e}")

        try:
            infer_ms = (t1 - t0) * 1000.0
            annotate_ms = (t2 - t1) * 1000.0
            overlay_ms = (t3 - t2) * 1000.0
            total_ms = (t3 - t0) * 1000.0
        except Exception:
            pass

        # Return annotated image plus timing anchors so recv() can compute queue and encode timings
        timing = {'t0': t0, 't1': t1, 't2': t2, 't3': t3}
        return annotated, timing


async def _build_peer_connection(model_name: str, task: str = 'detection', scale: float = 1.0, camera_id: str = '0') -> RTCPeerConnection:
    try:
        model = get_model(model_name)
    except ValueError as exc:
        raise ValueError(f"Model '{model_name}' is not available on the server") from exc

    pc = RTCPeerConnection()
    peer_connections.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await _cleanup_peer(pc)

    @pc.on("track")
    def on_track(track: MediaStreamTrack) -> None:
        print(f"[WebRTC] Incoming track kind={track.kind} for camera_id={camera_id}")
        if track.kind != "video":
            return

        local_video = YOLOTransformTrack(relay.subscribe(track), model_name, task=task, model=model, scale=scale, camera_id=camera_id)
        pc.addTrack(local_video)

        @track.on("ended")
        async def on_ended() -> None:
            await local_video.stop()

    return pc


async def _gather_ice(pc: RTCPeerConnection) -> None:
    """Wait until ICE gathering is complete before returning the answer."""

    if pc.iceGatheringState == "complete":
        return

    event = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def on_icegatheringstatechange() -> None:
        if pc.iceGatheringState == "complete" and not event.is_set():
            event.set()

    await event.wait()


async def _cleanup_peer(pc: RTCPeerConnection) -> None:
    if pc in peer_connections:
        peer_connections.discard(pc)
    await pc.close()


async def create_answer_for_offer(sdp: str, offer_type: str, model_name: str, task: str = 'detection', scale: float = 1.0, camera_id: str = '0') -> RTCSessionDescription:
    """Handle a browser offer and return the answer with processed video."""

    pc = await _build_peer_connection(model_name, task, scale, camera_id)

    offer = RTCSessionDescription(sdp=sdp, type=offer_type)
    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    await _gather_ice(pc)

    return pc.localDescription


async def close_all_peers() -> None:
    """Gracefully close all peer connections (used on shutdown)."""
    await asyncio.gather(*[_cleanup_peer(pc) for pc in list(peer_connections)], return_exceptions=True)


async def create_error_response(message: str) -> dict:
    return {"error": message}
