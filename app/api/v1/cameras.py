from typing import List

from fastapi import APIRouter

from app.models.camera import CameraResponse
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
