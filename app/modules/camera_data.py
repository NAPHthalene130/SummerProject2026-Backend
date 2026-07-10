import threading
from typing import Optional


CONSECUTIVE_NORMAL_THRESHOLD = 10


class BoundingBoxItem:
    def __init__(self, track_id: int, class_name: str, confidence: float, bbox: list[float]):
        self.track_id = track_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox


class CameraData:
    def __init__(self, camera_id: str, total_vehicle_count: int, boxes: list[BoundingBoxItem]):
        self.camera_id = camera_id
        self.total_vehicle_count = total_vehicle_count
        self.boxes = boxes


class TrafficIncidentResult:
    def __init__(self, camera_id: str, incident_detected: bool, incident_type: str, description: str):
        self.camera_id = camera_id
        self.incident_detected = incident_detected
        self.incident_type = incident_type
        self.description = description

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "incident_detected": self.incident_detected,
            "incident_type": self.incident_type,
            "description": self.description,
        }


INCIDENT_RANK_MAP = {
    "车辆碰撞": 3,
    "车辆起火": 3,
    "交通拥堵": 2,
    "行人闯入": 2,
    "恶劣天气": 2,
    "车辆抛锚": 1,
    "异常停车": 1,
    "道路障碍": 1,
    "其他事故": 1,
    "正常": 0,
}


def get_incident_rank(incident_type: str) -> int:
    return INCIDENT_RANK_MAP.get(incident_type, 1)


class CameraDataStore:
    _instance: Optional["CameraDataStore"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "CameraDataStore":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._data: dict[str, CameraData] = {}
                    cls._instance._incident_data: dict[str, TrafficIncidentResult] = {}
                    cls._instance._incident_active: dict[str, bool] = {}
                    cls._instance._consecutive_normal_count: dict[str, int] = {}
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

    def update_incident_result(self, camera_id: str, result: TrafficIncidentResult) -> None:
        self._incident_data[camera_id] = result

    def get_incident_result(self, camera_id: str) -> Optional[TrafficIncidentResult]:
        return self._incident_data.get(camera_id)

    def get_all_incident_results(self) -> dict[str, TrafficIncidentResult]:
        return dict(self._incident_data)

    def is_incident_active(self, camera_id: str) -> bool:
        return self._incident_active.get(camera_id, False)

    def activate_incident(self, camera_id: str) -> bool:
        was_active = self._incident_active.get(camera_id, False)
        self._incident_active[camera_id] = True
        self._consecutive_normal_count[camera_id] = 0
        return not was_active

    def register_normal_result(self, camera_id: str) -> bool:
        if not self._incident_active.get(camera_id, False):
            return False
        count = self._consecutive_normal_count.get(camera_id, 0) + 1
        self._consecutive_normal_count[camera_id] = count
        if count >= CONSECUTIVE_NORMAL_THRESHOLD:
            self._incident_active[camera_id] = False
            self._consecutive_normal_count[camera_id] = 0
            return True
        return False

    def get_consecutive_normal_count(self, camera_id: str) -> int:
        return self._consecutive_normal_count.get(camera_id, 0)
