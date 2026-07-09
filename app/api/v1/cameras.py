from typing import List

from fastapi import APIRouter

from app.models.camera import CameraResponse, CameraStatsItem, CameraStatsResponse
from app.modules.yolo import CameraDataStore
from app.utils.camera_manager import CameraManager

cameras_router = APIRouter()


@cameras_router.get("/", response_model=List[CameraResponse])
async def list_cameras():
    cameras = CameraManager().get_all()
    return [
        CameraResponse(
            id=cam.id,
            name=cam.name,
            longitude=cam.longitude,
            latitude=cam.latitude,
        )
        for cam in cameras
    ]


@cameras_router.get("/stats", response_model=CameraStatsResponse)
async def get_camera_stats():
    store = CameraDataStore()
    items: list[CameraStatsItem] = []
    for cam_id, data in store.get_all().items():
        items.append(CameraStatsItem(
            camera_id=cam_id,
            total_vehicle_count=data.total_vehicle_count,
        ))
    return CameraStatsResponse(cameras=items)
