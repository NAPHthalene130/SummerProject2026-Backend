"""Road traffic aggregation API"""

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
            "avg_speed": data.avg_speed,
            "max_speed": data.max_speed,
            "car_count": data.car_count,
            "truck_count": data.truck_count,
            "bus_count": data.bus_count,
            "moto_count": data.moto_count,
            **flow,
        }
    return {"cameras": result}
