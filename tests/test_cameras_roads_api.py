"""摄像头与道路流量 API 测试:检测框缓存、统计聚合与探测器故障回退。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.api.v1.cameras import (
    DetectionUpdateRequest,
    get_camera_boxes,
    get_camera_stats,
    get_traffic_flow,
    list_cameras,
    update_detections,
)
from app.api.v1.roads import get_road_traffic
from app.modules.camera_data import BoundingBoxItem, CameraDataStore


def _camera_config(camera_id: str, name: str) -> SimpleNamespace:
    return SimpleNamespace(id=camera_id, name=name, longitude=116.3, latitude=39.97)


class ListCamerasApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_maps_camera_configs_to_response(self) -> None:
        cameras = [_camera_config("cam-01", "路口一"), _camera_config("cam-02", "路口二")]
        with patch("app.api.v1.cameras.CameraManager") as manager_cls:
            manager_cls.return_value.get_all.return_value = cameras
            result = await list_cameras()

        self.assertEqual([c.id for c in result], ["cam-01", "cam-02"])
        self.assertEqual(result[0].name, "路口一")
        self.assertEqual(result[0].longitude, 116.3)
        self.assertEqual(result[0].latitude, 39.97)


class CameraBoxesApiTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.store = CameraDataStore()
        self.store._data.clear()

    def tearDown(self) -> None:
        self.store._data.clear()

    async def test_unknown_camera_returns_empty_boxes(self) -> None:
        result = await get_camera_boxes("cam-missing")

        self.assertEqual(result, {"boxes": []})

    async def test_boxes_are_mapped_with_track_fields(self) -> None:
        self.store.update(
            "cam-01",
            1,
            [BoundingBoxItem(7, "car", 0.88, [1.0, 2.0, 3.0, 4.0])],
        )
        result = await get_camera_boxes("cam-01")

        self.assertEqual(
            result["boxes"],
            [{"track_id": 7, "class_name": "car", "confidence": 0.88, "bbox": [1.0, 2.0, 3.0, 4.0]}],
        )

    async def test_stats_include_all_cameras(self) -> None:
        self.store.update("cam-01", 1, [BoundingBoxItem(1, "car", 0.9, [0, 0, 1, 1])])
        self.store.update("cam-02", 0, [])

        result = await get_camera_stats()

        by_id = {item.camera_id: item for item in result.cameras}
        self.assertEqual(set(by_id), {"cam-01", "cam-02"})
        self.assertEqual(by_id["cam-01"].total_vehicle_count, 1)
        self.assertEqual(len(by_id["cam-01"].boxes), 1)
        self.assertEqual(by_id["cam-02"].boxes, [])

    async def test_update_detections_defaults_missing_fields(self) -> None:
        request = DetectionUpdateRequest(camera_id="cam-09", boxes=[{}, {"track_id": 3}])

        result = await update_detections(request)

        self.assertEqual(result, {"status": "ok", "count": 2})
        data = self.store.get_by_camera_id("cam-09")
        self.assertEqual(data.total_vehicle_count, 2)
        self.assertEqual(data.boxes[0].bbox, [0, 0, 0, 0])
        self.assertEqual(data.boxes[1].track_id, 3)


class TrafficFlowApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_returns_detector_flow(self) -> None:
        detector = SimpleNamespace(
            get_traffic_flow=lambda cam_id: {"entry_count": 5, "exit_count": 3, "flow_per_min": 1.5}
        )
        with patch("app.api.v1.cameras.BatchDetector", return_value=detector):
            result = await get_traffic_flow("cam-01")

        self.assertEqual(result["entry_count"], 5)
        self.assertEqual(result["flow_per_min"], 1.5)

    async def test_detector_failure_falls_back_to_zeros(self) -> None:
        with patch("app.api.v1.cameras.BatchDetector", side_effect=RuntimeError("no model")):
            result = await get_traffic_flow("cam-01")

        self.assertEqual(result, {"entry_count": 0, "exit_count": 0, "flow_per_min": 0.0})


class RoadTrafficApiTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.store = CameraDataStore()
        self.store._data.clear()

    def tearDown(self) -> None:
        self.store._data.clear()

    async def test_merges_store_metrics_with_detector_flow(self) -> None:
        self.store.update(
            "cam-01",
            4,
            [],
            lane_count=3,
            avg_speed=45.0,
            max_speed=70.0,
            car_count=2,
            truck_count=1,
            bus_count=1,
            moto_count=0,
        )
        detector = SimpleNamespace(
            get_traffic_flow=lambda cam_id: {"entry_count": 9, "exit_count": 7, "flow_per_min": 2.0}
        )
        with patch("app.api.v1.roads.BatchDetector", return_value=detector):
            result = await get_road_traffic()

        cam = result["cameras"]["cam-01"]
        self.assertEqual(cam["total_vehicle_count"], 4)
        self.assertEqual(cam["lane_count"], 3)
        self.assertEqual(cam["avg_speed"], 45.0)
        self.assertEqual(cam["entry_count"], 9)
        self.assertEqual(cam["flow_per_min"], 2.0)

    async def test_detector_init_failure_still_returns_store_metrics(self) -> None:
        self.store.update("cam-01", 2, [], lane_count=2, avg_speed=30.0)

        with patch("app.api.v1.roads.BatchDetector", side_effect=RuntimeError("no model")):
            result = await get_road_traffic()

        cam = result["cameras"]["cam-01"]
        self.assertEqual(cam["total_vehicle_count"], 2)
        self.assertEqual(cam["entry_count"], 0)
        self.assertEqual(cam["flow_per_min"], 0.0)

    async def test_detector_query_failure_falls_back_per_camera(self) -> None:
        self.store.update("cam-01", 1, [])
        detector = SimpleNamespace(
            get_traffic_flow=lambda cam_id: (_ for _ in ()).throw(RuntimeError("broken"))
        )
        with patch("app.api.v1.roads.BatchDetector", return_value=detector):
            result = await get_road_traffic()

        self.assertEqual(result["cameras"]["cam-01"]["entry_count"], 0)


if __name__ == "__main__":
    unittest.main()
