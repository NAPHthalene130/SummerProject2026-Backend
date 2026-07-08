from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel


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

    def reload(self) -> None:
        self._load()

    def get_all(self) -> List[CameraConfig]:
        return self._cameras

    def get_by_id(self, camera_id: str) -> Optional[CameraConfig]:
        for cam in self._cameras:
            if cam.id == camera_id:
                return cam
        return None
