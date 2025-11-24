import asyncio

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

    def __init__(self, track: MediaStreamTrack, model_name: str, task: str = 'detection', model=None):
        super().__init__()  # initializes timestamp-related state
        self.track = track
        self.model_name = model_name
        self.task = task
        # Fetch (or load) the model once; model_manager caches subsequent requests
        self.model = model or get_model(model_name)

    async def recv(self) -> VideoFrame:
        frame = await self.track.recv()
        image = frame.to_ndarray(format="bgr24")

        try:
            loop = asyncio.get_running_loop()
            annotated = await loop.run_in_executor(None, self._run_inference, image)
        except Exception as err:
            print(f"[WebRTC] YOLO inference failed for model {self.model_name}: {err}")
            # Inference failure should not break the stream; fall back to raw frame
            annotated = image

        new_frame = VideoFrame.from_ndarray(annotated, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

    def _run_inference(self, image):
        results = self.model.predict(image, verbose=False)
        boxes = getattr(results[0], "boxes", None)
        count = len(boxes) if boxes is not None else 0
        print(f"[WebRTC] {self.model_name} detections: {count}")

        annotated = annotate_frame(image, results[0], task=self.task)

        try:
            annotated = add_model_overlay(annotated, self.model_name)
        except Exception as overlay_err:
            print(f"[WebRTC] failed to add overlay: {overlay_err}")

        return annotated


async def _build_peer_connection(model_name: str, task: str = 'detection') -> RTCPeerConnection:
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

        local_video = YOLOTransformTrack(relay.subscribe(track), model_name, task=task, model=model)
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


async def create_answer_for_offer(sdp: str, offer_type: str, model_name: str, task: str = 'detection') -> RTCSessionDescription:
    """Handle a browser offer and return the answer with processed video."""

    pc = await _build_peer_connection(model_name, task)

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
