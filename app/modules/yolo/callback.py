"""Callback module: pass inference results back to API layer"""

from app.modules.camera_data import CameraDataStore
from app.modules.yolo.batch_detector import BatchDetector


def get_traffic_flow(cam_id: str) -> dict:
    try:
        return BatchDetector().get_traffic_flow(cam_id)
    except Exception:
        return {"entry_count": 0, "exit_count": 0, "flow_per_min": 0.0}


def get_all_traffic_flow() -> dict:
    try:
        return BatchDetector().get_all_traffic_flow()
    except Exception:
        return {}


def get_camera_stats(cam_id: str = None) -> dict:
    store = CameraDataStore()
    if cam_id:
        data = store.get_by_camera_id(cam_id)
        if data is None:
            return {"boxes": []}
        return {"boxes": [{"track_id": b.track_id, "class_name": b.class_name, "confidence": b.confidence, "bbox": b.bbox} for b in data.boxes]}
    all_data = store.get_all()
    return {
        cam: {
            "total_vehicle_count": d.total_vehicle_count,
            "boxes": [{"track_id": b.track_id, "class_name": b.class_name, "confidence": b.confidence, "bbox": b.bbox} for b in d.boxes],
            "lane_count": d.lane_count,
            "avg_speed": d.avg_speed,
            "max_speed": d.max_speed,
            "car_count": d.car_count,
            "truck_count": d.truck_count,
            "bus_count": d.bus_count,
            "moto_count": d.moto_count,
        }
        for cam, d in all_data.items()
    }
