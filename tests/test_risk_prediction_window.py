from datetime import datetime

from app.modules.camera_data import BoundingBoxItem, CameraDataStore
from app.modules.risk_prediction.service import LiveRiskPredictionService


def test_prediction_window_collects_unique_tracks_and_speed_samples():
    store = CameraDataStore()
    camera_id = "test-risk-window-camera"
    boxes = [
        BoundingBoxItem(101, "car", 0.9, [0, 0, 10, 10]),
        BoundingBoxItem(102, "car", 0.9, [10, 0, 20, 10]),
    ]

    store.update(camera_id, total_vehicle_count=2, boxes=boxes)
    store.update_traffic_metrics(camera_id, avg_speed_kmh=36.0, vehicle_count=2)
    metrics = store.get_prediction_window_metrics(camera_id)

    assert metrics["cumulative_vehicle_count"] == 2
    assert metrics["sample_count"] >= 1
    assert metrics["avg_speed_kmh"] == 36.0


def test_prediction_reasons_include_date_flow_weather_road_and_history():
    now = datetime(2026, 7, 11, 12, 0)
    vehicle = {
        "window_vehicle_count": 180,
        "historical_baseline_count": 150,
        "flow_comparison": "偏多",
        "flow_change_percent": 20.0,
        "avg_speed_kmh": 28.0,
        "source": "yolo-window",
    }
    weather = {
        "temperature_2m": 26.0,
        "relative_humidity_2m": 70.0,
        "precipitation": 0.2,
        "rain": 0.2,
        "snowfall": 0.0,
        "wind_speed_10m": 8.0,
        "fallback": False,
    }
    road = {"road_type": "primary", "maxspeed": 60, "lanes": 4, "surface": "asphalt"}
    segment = {"speed_limit": 60, "lane_count": 4, "historical_accidents_7d": 2}

    reasons = LiveRiskPredictionService._build_reasons(
        0.62, weather, vehicle, road, segment, now
    )
    reason_text = "；".join(reasons)

    assert "2026-07-11 周六" in reason_text
    assert "相较历史同星期同时段基线 150 辆偏多 20.0%" in reason_text
    assert "近 15 分钟车辆加权平均速度 28.0 km/h" in reason_text
    assert "天气为降雨" in reason_text
    assert "primary 类型道路" in reason_text
    assert "近 7 天记录 2 起" in reason_text


def test_flow_change_is_the_primary_prediction_calibration_signal():
    weather = {
        "rain": 0.0,
        "snowfall": 0.0,
        "wind_speed_10m": 8.0,
    }
    road = {"road_type": "secondary", "maxspeed": 60, "lanes": 2}
    segment = {"road_type": "secondary", "speed_limit": 60, "lane_count": 2}
    common = {"avg_speed_kmh": 50.0}

    low_flow_score = LiveRiskPredictionService._calibrate_score(
        0.4, weather, {**common, "flow_change_percent": -30.0}, road, segment
    )
    high_flow_score = LiveRiskPredictionService._calibrate_score(
        0.4, weather, {**common, "flow_change_percent": 30.0}, road, segment
    )

    assert high_flow_score > low_flow_score
