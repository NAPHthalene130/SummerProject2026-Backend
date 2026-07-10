from typing import Optional

from app.modules.stream.camera_stream import CameraStream
from app.modules.yolo.batch_detector import BatchDetector
from app.utils.camera_manager import CameraConfig, CameraManager


class StreamManager:
    _instance: Optional["StreamManager"] = None

    def __new__(cls) -> "StreamManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._streams: dict[str, CameraStream] = {}
            cls._instance._batch_detector: Optional[BatchDetector] = None
        return cls._instance

    @property
    def batch_detector(self) -> BatchDetector:
        if self._batch_detector is None:
            self._batch_detector = BatchDetector()
            self._batch_detector.start()
        return self._batch_detector

    def get_stream(self, camera_id: str) -> Optional[CameraStream]:
        return self._streams.get(camera_id)

    def subscribe(self, camera_id: str) -> Optional[CameraStream]:
        stream = self._streams.get(camera_id)
        if stream is None:
            camera_config = self._resolve_camera(camera_id)
            if camera_config is None:
                return None
            detector = self.batch_detector
            stream = CameraStream(camera_config, detector)
            detector.register_stream(camera_id, stream)
            self._streams[camera_id] = stream
        stream.add_subscriber()
        return stream

    def unsubscribe(self, camera_id: str) -> None:
        stream = self._streams.get(camera_id)
        if stream is not None:
            stream.remove_subscriber()
            if stream.subscriber_count == 0:
                if self._batch_detector is not None:
                    self._batch_detector.unregister_stream(camera_id)
                del self._streams[camera_id]

    def get_all_streams(self) -> dict[str, "CameraStream"]:
        return dict(self._streams)

    def stop_all(self) -> None:
        for stream in self._streams.values():
            stream.stop()
        self._streams.clear()
        if self._batch_detector is not None:
            self._batch_detector.stop()
            self._batch_detector = None

    def _resolve_camera(self, camera_id: str) -> Optional[CameraConfig]:
        return CameraManager().get_by_id(camera_id)
