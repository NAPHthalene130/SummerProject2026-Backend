from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel


class CarInfo(BaseModel):
    car_license: str
    car_speed: float


class CameraInfo(BaseModel):
    car_info: List[CarInfo] = []
    weather: str = ""


class CameraConfig(BaseModel):
    id: str
    name: str
    url: str
    longitude: float
    latitude: float


class CameraConfigList(BaseModel):
    cameras: List[CameraConfig]


class CameraManager:
    _instance: Optional["CameraManager"] = None
    _cameras: List[CameraConfig] = []
    _camera_info_map: Dict[str, CameraInfo] = {}

    def __new__(cls) -> "CameraManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        config_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        config_list = CameraConfigList.model_validate(data)
        self._cameras = config_list.cameras
        self._camera_info_map = {}

    def reload(self) -> None:
        self._load()

    def get_all(self) -> List[CameraConfig]:
        return self._cameras

    def get_by_id(self, camera_id: str) -> Optional[CameraConfig]:
        for cam in self._cameras:
            if cam.id == camera_id:
                return cam
        return None

    def get_camera_info(self, camera_id: str) -> Optional[CameraInfo]:
        return self._camera_info_map.get(camera_id)

    def update_camera_info(self, camera_id: str, camera_info: CameraInfo) -> None:
        self._camera_info_map[camera_id] = camera_info

    def get_all_camera_info(self) -> Dict[str, CameraInfo]:
        return self._camera_info_map
