from app.modules.stream.camera_stream import CameraStream
from app.modules.stream.stream_manager import StreamManager

try:
    from app.modules.stream.video_track import ProcessedVideoTrack
except Exception:
    class ProcessedVideoTrack:
        def __init__(self, *a, **kw):
            raise RuntimeError("aiortc/cryptography not available")

__all__ = ["CameraStream", "StreamManager", "ProcessedVideoTrack"]
