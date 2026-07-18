"""CameraDataStore 核心状态逻辑测试:检测数据缓存、预测窗口聚合、事故生命周期。"""

import time
import unittest

from app.modules.camera_data import (
    CONSECUTIVE_NORMAL_THRESHOLD,
    PREDICTION_SAMPLE_INTERVAL_SECONDS,
    BoundingBoxItem,
    CameraDataStore,
    TrafficIncidentResult,
    get_incident_rank,
)


def _boxes(*track_ids: int) -> list[BoundingBoxItem]:
    return [
        BoundingBoxItem(track_id, "car", 0.9, [0.0, 0.0, 10.0, 10.0], speed=30.0)
        for track_id in track_ids
    ]


class CameraDataUpdateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = CameraDataStore()
        for attr in ("_data", "_prediction_tracks", "_traffic_metrics", "_lane_count_cache"):
            getattr(self.store, attr).clear()

    def test_update_stores_camera_data_with_all_metrics(self) -> None:
        boxes = _boxes(1, 2)
        self.store.update(
            "cam-1",
            total_vehicle_count=2,
            boxes=boxes,
            lane_count=3,
            avg_speed=42.5,
            max_speed=60.0,
            car_count=1,
            truck_count=1,
            bus_count=0,
            moto_count=0,
            avg_headway=25.0,
        )

        data = self.store.get_by_camera_id("cam-1")
        self.assertIsNotNone(data)
        self.assertEqual(data.total_vehicle_count, 2)
        self.assertEqual(data.lane_count, 3)
        self.assertEqual(data.avg_speed, 42.5)
        self.assertEqual(data.truck_count, 1)
        self.assertEqual(data.avg_headway, 25.0)
        self.assertIsNone(self.store.get_by_camera_id("cam-missing"))

    def test_get_all_returns_a_shallow_copy(self) -> None:
        self.store.update("cam-1", 0, [])
        snapshot = self.store.get_all()
        snapshot.pop("cam-1")

        self.assertIn("cam-1", self.store.get_all())

    def test_negative_track_ids_are_not_recorded_for_prediction(self) -> None:
        self.store.update("cam-1", 2, _boxes(-1, 5))

        self.assertEqual(set(self.store._prediction_tracks["cam-1"]), {5})

    def test_lane_count_cache_roundtrip(self) -> None:
        self.assertIsNone(self.store.get_lane_count("cam-1"))

        self.store.update_lane_count("cam-1", 4)

        self.assertEqual(self.store.get_lane_count("cam-1"), 4)


class TrafficMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = CameraDataStore()
        self.store._traffic_metrics.clear()
        self.store._prediction_speed_samples.clear()
        self.store._last_prediction_sample.clear()

    def test_metrics_are_sampled_at_most_once_per_interval(self) -> None:
        self.store.update_traffic_metrics("cam-1", avg_speed_kmh=30.0, vehicle_count=2)
        self.store.update_traffic_metrics("cam-1", avg_speed_kmh=50.0, vehicle_count=9)

        self.assertEqual(len(self.store._prediction_speed_samples["cam-1"]), 1)
        # 实时指标总是被刷新,只有采样被限频
        self.assertEqual(
            self.store.get_traffic_metrics("cam-1"),
            {"avg_speed_kmh": 50.0, "vehicle_count": 9.0},
        )

    def test_new_sample_is_added_after_the_interval(self) -> None:
        self.store.update_traffic_metrics("cam-1", avg_speed_kmh=30.0, vehicle_count=2)
        self.store._last_prediction_sample["cam-1"] -= PREDICTION_SAMPLE_INTERVAL_SECONDS + 1
        self.store.update_traffic_metrics("cam-1", avg_speed_kmh=50.0, vehicle_count=3)

        self.assertEqual(len(self.store._prediction_speed_samples["cam-1"]), 2)

    def test_get_traffic_metrics_returns_copy_and_missing_camera(self) -> None:
        self.assertIsNone(self.store.get_traffic_metrics("cam-1"))

        self.store.update_traffic_metrics("cam-1", avg_speed_kmh=30.0, vehicle_count=2)
        metrics = self.store.get_traffic_metrics("cam-1")
        metrics["avg_speed_kmh"] = -1.0

        self.assertEqual(self.store.get_traffic_metrics("cam-1")["avg_speed_kmh"], 30.0)


class PredictionWindowMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = CameraDataStore()
        self.store._prediction_tracks.clear()
        self.store._prediction_speed_samples.clear()

    def test_empty_camera_returns_zeroed_metrics(self) -> None:
        metrics = self.store.get_prediction_window_metrics("cam-1")

        self.assertEqual(metrics["cumulative_vehicle_count"], 0)
        self.assertIsNone(metrics["avg_speed_kmh"])
        self.assertEqual(metrics["sample_count"], 0)
        self.assertEqual(metrics["observed_seconds"], 0.0)

    def test_window_cutoff_excludes_old_tracks_and_samples(self) -> None:
        from collections import deque

        now = time.time()
        self.store._prediction_tracks["cam-1"] = {1: now - 100, 2: now - 10, 3: now - 4000}
        samples = self.store._prediction_speed_samples.setdefault("cam-1", deque())
        samples.append((now - 100, 40.0, 2))
        samples.append((now - 4000, 80.0, 5))

        metrics = self.store.get_prediction_window_metrics("cam-1", window_seconds=900, now=now)

        self.assertEqual(metrics["cumulative_vehicle_count"], 2)
        self.assertEqual(metrics["sample_count"], 1)
        self.assertEqual(metrics["avg_speed_kmh"], 40.0)

    def test_average_speed_is_weighted_by_vehicle_count(self) -> None:
        from collections import deque

        now = time.time()
        samples = self.store._prediction_speed_samples.setdefault("cam-1", deque())
        samples.append((now - 10, 30.0, 1))
        samples.append((now - 5, 60.0, 3))
        # 速度为0或车辆数为0的样本不参与加权
        samples.append((now - 1, 0.0, 9))

        metrics = self.store.get_prediction_window_metrics("cam-1", window_seconds=900, now=now)

        self.assertAlmostEqual(metrics["avg_speed_kmh"], (30.0 * 1 + 60.0 * 3) / 4)
        self.assertEqual(metrics["sample_count"], 3)

    def test_observed_seconds_tracks_earliest_in_window_observation(self) -> None:
        now = time.time()
        self.store._prediction_tracks["cam-1"] = {1: now - 500, 2: now - 100}

        metrics = self.store.get_prediction_window_metrics("cam-1", window_seconds=900, now=now)

        self.assertEqual(metrics["observed_seconds"], 500.0)

    def test_observations_older_than_window_are_not_counted(self) -> None:
        now = time.time()
        self.store._prediction_tracks["cam-1"] = {1: now - 5000}

        metrics = self.store.get_prediction_window_metrics("cam-1", window_seconds=900, now=now)

        self.assertEqual(metrics["cumulative_vehicle_count"], 0)
        self.assertEqual(metrics["observed_seconds"], 0.0)


class IncidentLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = CameraDataStore()
        self.store._incident_data.clear()
        self.store._active_incidents.clear()
        self.store._consecutive_normal_count.clear()

    def test_incident_result_roundtrip(self) -> None:
        result = TrafficIncidentResult("cam-1", True, "车辆碰撞", "两车追尾", 0.95)
        self.store.update_incident_result("cam-1", result)

        self.assertIs(self.store.get_incident_result("cam-1"), result)
        self.assertEqual(self.store.get_all_incident_results(), {"cam-1": result})
        self.assertEqual(result.to_dict()["incident_type"], "车辆碰撞")

    def test_activate_incident_is_idempotent_and_resets_normal_streak(self) -> None:
        self.store._consecutive_normal_count["cam-1"] = 5

        self.assertTrue(self.store.activate_incident("cam-1", "车辆碰撞"))
        self.assertFalse(self.store.activate_incident("cam-1", "车辆碰撞"))

        self.assertTrue(self.store.is_incident_active("cam-1"))
        self.assertTrue(self.store.is_incident_active("cam-1", "车辆碰撞"))
        self.assertFalse(self.store.is_incident_active("cam-1", "交通拥堵"))
        self.assertEqual(self.store.get_consecutive_normal_count("cam-1"), 0)

    def test_observed_incident_resets_normal_streak_without_clearing_others(self) -> None:
        self.store.activate_incident("cam-1", "交通拥堵")
        self.store._consecutive_normal_count["cam-1"] = CONSECUTIVE_NORMAL_THRESHOLD - 1

        cleared = self.store.register_analysis_result("cam-1", "车辆碰撞")

        self.assertEqual(cleared, [])
        self.assertEqual(self.store.get_consecutive_normal_count("cam-1"), 0)
        self.assertTrue(self.store.is_incident_active("cam-1", "交通拥堵"))

    def test_normal_results_clear_all_types_at_threshold(self) -> None:
        self.store.activate_incident("cam-1", "交通拥堵")
        self.store.activate_incident("cam-1", "车辆碰撞")

        for _ in range(CONSECUTIVE_NORMAL_THRESHOLD - 1):
            cleared = self.store.register_analysis_result("cam-1", None)
            self.assertEqual(cleared, [])
        self.assertTrue(self.store.is_incident_active("cam-1"))

        cleared = self.store.register_analysis_result("cam-1", None)

        self.assertEqual(cleared, ["交通拥堵", "车辆碰撞"])
        self.assertFalse(self.store.is_incident_active("cam-1"))
        # 计数器随事故清除一并复位
        self.assertEqual(self.store.get_consecutive_normal_count("cam-1"), 0)

    def test_normal_result_without_active_incidents_drops_counter(self) -> None:
        self.store._consecutive_normal_count["cam-1"] = 3

        cleared = self.store.register_analysis_result("cam-1", None)

        self.assertEqual(cleared, [])
        self.assertNotIn("cam-1", self.store._consecutive_normal_count)

    def test_register_normal_result_reports_only_threshold_crossing(self) -> None:
        self.store.activate_incident("cam-1", "车辆碰撞")

        for _ in range(CONSECUTIVE_NORMAL_THRESHOLD - 1):
            self.assertFalse(self.store.register_normal_result("cam-1"))
        self.assertTrue(self.store.register_normal_result("cam-1"))

    def test_consecutive_normal_count_is_zero_for_inactive_type(self) -> None:
        self.store._consecutive_normal_count["cam-1"] = 4

        self.assertEqual(self.store.get_consecutive_normal_count("cam-1", "车辆碰撞"), 0)
        self.assertEqual(self.store.get_consecutive_normal_count("cam-1"), 4)

    def test_get_active_incident_types_returns_copy(self) -> None:
        self.store.activate_incident("cam-1", "车辆碰撞")
        types = self.store.get_active_incident_types("cam-1")
        types.add("交通拥堵")

        self.assertEqual(self.store.get_active_incident_types("cam-1"), {"车辆碰撞"})


class IncidentRankTest(unittest.TestCase):
    def test_known_types_map_to_configured_ranks(self) -> None:
        self.assertEqual(get_incident_rank("车辆碰撞"), 3)
        self.assertEqual(get_incident_rank("车辆起火"), 3)
        self.assertEqual(get_incident_rank("交通拥堵"), 2)
        self.assertEqual(get_incident_rank("车辆抛锚"), 1)
        self.assertEqual(get_incident_rank("正常"), 0)

    def test_unknown_type_defaults_to_rank_one(self) -> None:
        self.assertEqual(get_incident_rank("未收录类型"), 1)


if __name__ == "__main__":
    unittest.main()
