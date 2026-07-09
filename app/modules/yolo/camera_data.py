import threading
from typing import Optional

from pydantic import BaseModel


class BoundingBoxItem(BaseModel):
    track_id: int
    class_name: str
    confidence: float
    bbox: list[float]


class CameraData(BaseModel):
    camera_id: str
    total_vehicle_count: int
    boxes: list[BoundingBoxItem]


class CameraDataStore:
    _instance: Optional["CameraDataStore"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "CameraDataStore":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._data: dict[str, CameraData] = {}
        return cls._instance

    def update(
        self,
        camera_id: str,
        total_vehicle_count: int,
        boxes: list[BoundingBoxItem],
    ) -> None:
        self._data[camera_id] = CameraData(
            camera_id=camera_id,
            total_vehicle_count=total_vehicle_count,
            boxes=boxes,
        )

    def get_by_camera_id(self, camera_id: str) -> Optional[CameraData]:
        return self._data.get(camera_id)

    def get_all(self) -> dict[str, CameraData]:
        return dict(self._data)
