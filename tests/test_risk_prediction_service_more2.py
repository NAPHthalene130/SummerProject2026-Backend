"""风险预测服务扩展测试:predict 流程、车辆上下文、Live API 端点补充。"""

import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd

from app.modules.camera_data import BoundingBoxItem, CameraDataStore
from app.modules.risk_prediction.service import LiveRiskPredictionService


class PredictEmptySegmentsTest(unittest.IsolatedAsyncioTestCase):
    async def test_predict_empty_segments_returns_metadata_only(self) -> None:
        service = LiveRiskPredictionService()

        result = await service.predict([])

        self.assertEqual(len(result["predictions"]), 0)
        self.assertIn("generated_at", result)
        self.assertIn("date_context", result)
        self.assertIn("forecast_minutes", result)

    async def test_predict_with_default_road_type(self) -> None:
        service = LiveRiskPredictionService()
        segments = [{
            "segment_id": "seg-1", "latitude": 39.97, "longitude": 116.31,
            "name": "海淀南路", "road_type": "primary",
            "lane_count": 4, "speed_limit": 60, "camera_ids": [],
            "traffic_flow": 60, "avg_speed": 40,
            "historical_accidents_24h": 0, "historical_accidents_7d": 0,
        }]

        with (patch.object(service, "_fetch_weather", new_callable=AsyncMock) as mock_w,
              patch.object(service, "_get_model", return_value=SimpleNamespace(predict=lambda df: [0.35]))):
            mock_w.return_value = {
                "temperature_2m": 25.0, "relative_humidity_2m": 60.0,
                "precipitation": 0.0, "rain": 0.0, "snowfall": 0.0,
                "wind_speed_10m": 8.0, "weather_code": 0,
                "source": "open-meteo", "fallback": False,
            }
            result = await service.predict(segments)

        self.assertEqual(len(result["predictions"]), 1)
        pred = result["predictions"][0]
        self.assertEqual(pred["segment_id"], "seg-1")
        self.assertIn("risk_score", pred)
        self.assertIn("risk_level", pred)
        self.assertEqual(result["weather"]["source"], "open-meteo")

    async def test_predict_fallback_weather_on_http_failure(self) -> None:
        service = LiveRiskPredictionService()
        segments = [{
            "segment_id": "seg-1", "latitude": 39.97, "longitude": 116.31,
            "name": "", "road_type": "unknown",
            "lane_count": 2, "speed_limit": 40, "camera_ids": [],
            "traffic_flow": 30, "avg_speed": 30,
            "historical_accidents_24h": 0, "historical_accidents_7d": 0,
        }]

        with (patch.object(service, "_fetch_weather", new_callable=AsyncMock) as mock_w,
              patch.object(service, "_get_model", return_value=SimpleNamespace(predict=lambda df: [0.2]))):
            mock_w.return_value = {
                "temperature_2m": 20.0, "relative_humidity_2m": 60.0,
                "precipitation": 0.0, "rain": 0.0, "snowfall": 0.0,
                "wind_speed_10m": 8.0, "weather_code": 0,
                "source": "fallback", "fallback": True,
                "error": "offline",
            }
            result = await service.predict(segments)

        self.assertEqual(result["weather"]["source"], "fallback")
        self.assertTrue(result["weather"]["fallback"])


class VehicleContextTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = CameraDataStore()
        self.store._data.clear()
        self.store._prediction_tracks.clear()
        self.store._prediction_speed_samples.clear()
        self.store._active_incidents.clear()

    def tearDown(self) -> None:
        self.store._data.clear()
        self.store._active_incidents.clear()

    def test_empty_camera_ids_falls_back_to_historical_baseline(self) -> None:
        service = LiveRiskPredictionService()
        now = datetime(2026, 7, 13, 12, 0)

        context = service._vehicle_context({
            "camera_ids": [], "traffic_flow": 60, "avg_speed": 40,
        }, now)

        self.assertIn("historical-road-baseline", context["source"])
        self.assertEqual(context["flow_comparison"], "基本持平")
        self.assertIsInstance(context["window_vehicle_count"], int)

    def test_camera_ids_mapped_through_backend_conversion(self) -> None:
        service = LiveRiskPredictionService()
        now = datetime(2026, 7, 13, 12, 0)
        import time
        from collections import deque
        self.store.update("cam-01", 5, [
            BoundingBoxItem(1, "car", 0.9, [0, 0, 10, 10], 30.0)
        ])
        self.store._prediction_tracks["cam-01"] = {1: time.time()}
        self.store._prediction_speed_samples["cam-01"] = deque(
            [(time.time() - 60, 36.0, 1)]
        )
        self.store.activate_incident("cam-01", "车辆碰撞")

        context = service._vehicle_context({
            "camera_ids": ["C01"], "traffic_flow": 60, "avg_speed": 40,
        }, now)

        self.assertEqual(context["instantaneous_vehicle_count"], 5)
        self.assertEqual(context["active_incidents"], 1)


class PredictFrameTest(unittest.TestCase):
    def test_predict_frame_clips_output(self) -> None:
        service = LiveRiskPredictionService()

        class FakeModel:
            @staticmethod
            def predict(df):
                return np.array([0.7, 1.5, -0.2])

        with patch.object(service, "_get_model", return_value=FakeModel()):
            result = service._predict_frame(pd.DataFrame([{} for _ in range(3)]))

        self.assertEqual(list(result), [0.7, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
