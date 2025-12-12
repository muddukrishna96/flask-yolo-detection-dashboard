import asyncio
import time
import cv2

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from aiortc.mediastreams import MediaStreamTrack
from av import VideoFrame

from backend.model_manager import get_model
from backend.image_video_processor import add_model_overlay, annotate_frame


relay = MediaRelay()
peer_connections: set[RTCPeerConnection] = set()


class YOLOTransformTrack(MediaStreamTrack):
    """Media stream track that runs YOLO inference on incoming frames."""

    kind = "video"

    def __init__(self, track: MediaStreamTrack, model_name: str, task: str = 'detection', model=None, scale: float = 1.0):
        super().__init__()  # initializes timestamp-related state
        self.track = track
        self.model_name = model_name
        self.task = task
        # Requested capture/process scale (1.0 == native). If <1.0 we will downscale frames before inference.
        self.scale = float(scale or 1.0)
        # Fetch (or load) the model once; model_manager caches subsequent requests
        self.model = model or get_model(model_name)

    async def recv(self) -> VideoFrame:
        frame = await self.track.recv()
        t_recv = time.perf_counter()
        image = frame.to_ndarray(format="bgr24")
        orig_h, orig_w = image.shape[:2]

        try:
            loop = asyncio.get_running_loop()
            # pass t_recv and original resolution so the worker can compute queue/wait time and upscale
            annotated, timing = await loop.run_in_executor(None, self._run_inference, image, t_recv, orig_h, orig_w)
        except Exception as err:
            print(f"[WebRTC] YOLO inference failed for model {self.model_name}: {err}")
            # Inference failure should not break the stream; fall back to raw frame
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


async def _build_peer_connection(model_name: str, task: str = 'detection', scale: float = 1.0) -> RTCPeerConnection:
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
        print(f"[WebRTC] Incoming track kind={track.kind}")
        if track.kind != "video":
            return

        local_video = YOLOTransformTrack(relay.subscribe(track), model_name, task=task, model=model, scale=scale)
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


async def create_answer_for_offer(sdp: str, offer_type: str, model_name: str, task: str = 'detection', scale: float = 1.0) -> RTCSessionDescription:
    """Handle a browser offer and return the answer with processed video."""

    pc = await _build_peer_connection(model_name, task, scale)

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
