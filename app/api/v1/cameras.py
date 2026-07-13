import asyncio
import json
from typing import List

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.models.camera import CameraResponse, CameraStatsItem, CameraStatsResponse
from app.modules.yolo import CameraDataStore, BatchDetector
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
        boxes_data = [
            {"track_id": b.track_id, "class_name": b.class_name, "confidence": b.confidence, "bbox": b.bbox}
            for b in data.boxes
        ]
        items.append(CameraStatsItem(
            camera_id=cam_id,
            total_vehicle_count=data.total_vehicle_count,
            boxes=boxes_data,
        ))
    return CameraStatsResponse(cameras=items)


@cameras_router.get("/boxes/stream")
async def stream_all_boxes():
    async def event_stream():
        while True:
            store = CameraDataStore()
            payload = {}
            for cam_id, data in store.get_all().items():
                payload[cam_id] = [
                    {"track_id": b.track_id, "class_name": b.class_name, "confidence": b.confidence, "bbox": b.bbox}
                    for b in data.boxes
                ]
            try:
                detector = BatchDetector()
                payload["_traffic"] = detector.get_all_traffic_flow()
                payload["_lanes"] = {}
                for cam_id, data in store.get_all().items():
                    payload["_lanes"][cam_id] = data.lane_count
            except Exception:
                pass
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(0.1)
    return StreamingResponse(event_stream(), media_type="text/event-stream")


class DetectionUpdateRequest(BaseModel):
    camera_id: str
    boxes: list[dict]


@cameras_router.post("/boxes/update")
async def update_detections(req: DetectionUpdateRequest):
    from app.modules.camera_data import BoundingBoxItem
    store = CameraDataStore()
    box_items = [
        BoundingBoxItem(
            track_id=b.get("track_id", 0),
            class_name=b.get("class_name", ""),
            confidence=b.get("confidence", 0.0),
            bbox=b.get("bbox", [0, 0, 0, 0]),
        )
        for b in req.boxes
    ]
    store.update(camera_id=req.camera_id, total_vehicle_count=len(box_items), boxes=box_items)
    return {"status": "ok", "count": len(box_items)}


@cameras_router.get("/{camera_id}/boxes")
async def get_camera_boxes(camera_id: str):
    store = CameraDataStore()
    data = store.get_by_camera_id(camera_id)
    if data is None:
        return {"boxes": []}
    return {
        "boxes": [
            {"track_id": b.track_id, "class_name": b.class_name, "confidence": b.confidence, "bbox": b.bbox}
            for b in data.boxes
        ]
    }


@cameras_router.get("/{camera_id}/traffic")
async def get_traffic_flow(camera_id: str):
    try:
        detector = BatchDetector()
        return detector.get_traffic_flow(camera_id)
    except Exception:
        return {"entry_count": 0, "exit_count": 0, "flow_per_min": 0.0}
