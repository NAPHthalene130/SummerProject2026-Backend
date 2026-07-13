import threading
from typing import Optional

from app.modules.stream.camera_stream import CameraStream
from app.modules.yolo.batch_detector import BatchDetector
from app.utils.camera_manager import CameraConfig, CameraManager


class StreamManager:
    _instance: Optional["StreamManager"] = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> "StreamManager":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._streams = {}
                    cls._instance._batch_detector = None
                    cls._instance._lock = threading.RLock()
        return cls._instance

    @property
    def batch_detector(self) -> BatchDetector:
        with self._lock:
            if self._batch_detector is None:
                detector = BatchDetector()
                detector.start()
                self._batch_detector = detector
            return self._batch_detector

    def get_stream(self, camera_id: str) -> Optional[CameraStream]:
        with self._lock:
            return self._streams.get(camera_id)

    def subscribe(
        self,
        camera_id: str,
        *,
        viewer: bool = False,
    ) -> Optional[CameraStream]:
        with self._lock:
            stream = self._streams.get(camera_id)
            if stream is None:
                camera_config = self._resolve_camera(camera_id)
                if camera_config is None:
                    return None
                detector = self.batch_detector
                stream = CameraStream(camera_config, detector)
                detector.register_stream(camera_id, stream)
                self._streams[camera_id] = stream
            stream.add_subscriber(viewer=viewer)
            return stream

    def unsubscribe(self, camera_id: str, *, viewer: bool = False) -> None:
        stream_to_stop: Optional[CameraStream] = None
        with self._lock:
            stream = self._streams.get(camera_id)
            if stream is not None:
                should_stop = stream.remove_subscriber(
                    viewer=viewer,
                    stop_if_unused=False,
                )
                if should_stop:
                    # Signal first, then detach this generation from YOLO.  The
                    # potentially multi-second FFmpeg join happens outside the
                    # manager lock so other cameras remain independently usable.
                    stream.request_stop()
                    if self._batch_detector is not None:
                        self._batch_detector.unregister_stream(camera_id)
                    self._streams.pop(camera_id, None)
                    stream_to_stop = stream
        if stream_to_stop is not None:
            stream_to_stop.stop()

    def get_all_streams(self) -> dict[str, "CameraStream"]:
        with self._lock:
            return dict(self._streams)

    def health_snapshots(self) -> list[dict]:
        with self._lock:
            streams = list(self._streams.values())
        return [stream.health_snapshot() for stream in streams]

    def pipeline_health(self) -> dict:
        with self._lock:
            streams = list(self._streams.values())
            detector = self._batch_detector
        return {
            "streams": [stream.health_snapshot() for stream in streams],
            "detector": detector.performance_snapshot() if detector is not None else None,
        }

    def stop_all(self) -> None:
        with self._lock:
            streams = list(self._streams.values())
            detector = self._batch_detector
            self._streams.clear()
            self._batch_detector = None
        # Signal all 30 reads first so their timeout windows overlap instead of
        # serialising up to 30 * RTSP_READ_TIMEOUT_MS during shutdown.
        for stream in streams:
            stream.request_stop()
        for stream in streams:
            stream.stop()
        if detector is not None:
            detector.stop()

    def _resolve_camera(self, camera_id: str) -> Optional[CameraConfig]:
        return CameraManager().get_by_id(camera_id)
