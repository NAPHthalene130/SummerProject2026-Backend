"""道路车流聚合 API"""

from fastapi import APIRouter

from app.modules.camera_data import CameraDataStore
from app.modules.yolo import BatchDetector

router = APIRouter()


@router.get("/traffic")
async def get_road_traffic():
    store = CameraDataStore()
    all_data = store.get_all()
    try:
        detector = BatchDetector()
    except Exception:
        detector = None

    result = {}
    for cam_id, data in all_data.items():
        flow = {"entry_count": 0, "exit_count": 0, "flow_per_min": 0.0}
        if detector:
            try:
                flow = detector.get_traffic_flow(cam_id)
            except Exception:
                pass
        result[cam_id] = {
            "total_vehicle_count": data.total_vehicle_count,
            "lane_count": data.lane_count,
            "avg_speed": getattr(data, "avg_speed", 0.0),
            "max_speed": getattr(data, "max_speed", 0.0),
            "car_count": getattr(data, "car_count", 0),
            "truck_count": getattr(data, "truck_count", 0),
            "bus_count": getattr(data, "bus_count", 0),
            "moto_count": getattr(data, "moto_count", 0),
            **flow,
        }
    return {"cameras": result}
