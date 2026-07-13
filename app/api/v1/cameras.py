import asyncio
import json
from typing import List

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.models.camera import CameraResponse, CameraStatsItem, CameraStatsResponse
from app.modules.stream import StreamManager
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


@cameras_router.get("/stream-health")
async def get_stream_health():
    """Runtime RTSP/YOLO diagnostics for locating an unhealthy feed or bottleneck."""
    return StreamManager().pipeline_health()


@cameras_router.get("/boxes/stream")
async def stream_all_boxes():
    """Push lightweight YOLO metadata independently of the WebRTC video path."""

    async def event_stream():
        while True:
            payload: dict[str, object] = {}
            frame_metadata: dict[str, dict[str, float | int]] = {}
            for camera_id, data in CameraDataStore().get_all().items():
                payload[camera_id] = [
                    {
                        "track_id": box.track_id,
                        "class_name": box.class_name,
                        "confidence": box.confidence,
                        "bbox": box.bbox,
                    }
                    for box in data.boxes
                ]
                frame_metadata[camera_id] = {
                    "width": data.frame_width,
                    "height": data.frame_height,
                    "updated_at": data.updated_at,
                }
            payload["_frames"] = frame_metadata
            yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
