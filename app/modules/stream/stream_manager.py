from typing import Optional

from app.modules.stream.camera_stream import CameraStream
from app.modules.yolo import YOLODetector
from app.utils.camera_manager import CameraConfig, CameraManager


class StreamManager:
    _instance: Optional["StreamManager"] = None

    def __new__(cls) -> "StreamManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._streams: dict[str, CameraStream] = {}
            cls._instance._yolo: Optional[YOLODetector] = None
        return cls._instance

    @property
    def yolo(self) -> YOLODetector:
        if self._yolo is None:
            self._yolo = YOLODetector()
        return self._yolo

    def get_stream(self, camera_id: str) -> Optional[CameraStream]:
        return self._streams.get(camera_id)

    def subscribe(self, camera_id: str) -> Optional[CameraStream]:
        stream = self._streams.get(camera_id)
        if stream is None:
            camera_config = self._resolve_camera(camera_id)
            if camera_config is None:
                return None
            stream = CameraStream(camera_config, self.yolo)
            self._streams[camera_id] = stream
        stream.add_subscriber()
        return stream

    def unsubscribe(self, camera_id: str) -> None:
        stream = self._streams.get(camera_id)
        if stream is not None:
            stream.remove_subscriber()
            if stream.subscriber_count == 0:
                del self._streams[camera_id]

    def stop_all(self) -> None:
        for stream in self._streams.values():
            stream.stop()
        self._streams.clear()

    def _resolve_camera(self, camera_id: str) -> Optional[CameraConfig]:
        return CameraManager().get_by_id(camera_id)
