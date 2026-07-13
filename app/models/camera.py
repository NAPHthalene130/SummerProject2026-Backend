from pydantic import BaseModel


class CameraResponse(BaseModel):
    id: str
    name: str
    longitude: float
    latitude: float


class CameraStatsItem(BaseModel):
    camera_id: str
    total_vehicle_count: int
    boxes: list[dict] = []


class CameraStatsResponse(BaseModel):
    cameras: list[CameraStatsItem]
