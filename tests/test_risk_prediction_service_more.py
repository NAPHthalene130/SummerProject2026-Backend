"""风险预测服务扩展测试：校准、天气评估、时段、车流基线、构建理由等纯函数。"""

import unittest
from datetime import datetime

from app.modules.risk_prediction.service import LiveRiskPredictionService


class BackendCameraIdTest(unittest.TestCase):
    def test_conversions(self) -> None:
        self.assertEqual(LiveRiskPredictionService._backend_camera_id("cam-01"), "cam-01")
        self.assertEqual(LiveRiskPredictionService._backend_camera_id("CAM-09"), "cam-09")
        self.assertEqual(LiveRiskPredictionService._backend_camera_id("C05"), "cam-05")
        self.assertEqual(LiveRiskPredictionService._backend_camera_id("30"), "cam-30")
        self.assertEqual(LiveRiskPredictionService._backend_camera_id("1"), "cam-01")
        self.assertEqual(LiveRiskPredictionService._backend_camera_id("unknown"), "unknown")


class RiskLevelTest(unittest.TestCase):
    def test_boundaries(self) -> None:
        self.assertEqual(LiveRiskPredictionService._risk_level(0.0), "normal")
        self.assertEqual(LiveRiskPredictionService._risk_level(0.29), "normal")
        self.assertEqual(LiveRiskPredictionService._risk_level(0.3), "busy")
        self.assertEqual(LiveRiskPredictionService._risk_level(0.49), "busy")
        self.assertEqual(LiveRiskPredictionService._risk_level(0.5), "danger")
        self.assertEqual(LiveRiskPredictionService._risk_level(0.99), "danger")


class DateContextTest(unittest.TestCase):
    def test_weekday_and_weekend(self) -> None:
        monday = datetime(2026, 7, 13, 8, 0)
        context = LiveRiskPredictionService._date_context(monday)
        self.assertEqual(context["weekday"], "周一")
        self.assertFalse(context["is_weekend"])
        self.assertEqual(context["period"], "早高峰")

        saturday = datetime(2026, 7, 18, 14, 0)
        context = LiveRiskPredictionService._date_context(saturday)
        self.assertEqual(context["weekday"], "周六")
        self.assertTrue(context["is_weekend"])
        self.assertEqual(context["period"], "日间平峰")


class PeriodNameTest(unittest.TestCase):
    def test_periods(self) -> None:
        self.assertEqual(LiveRiskPredictionService._period_name(7), "早高峰")
        self.assertEqual(LiveRiskPredictionService._period_name(9), "早高峰")
        self.assertEqual(LiveRiskPredictionService._period_name(17), "晚高峰")
        self.assertEqual(LiveRiskPredictionService._period_name(19), "晚高峰")
        self.assertEqual(LiveRiskPredictionService._period_name(0), "夜间低峰")
        self.assertEqual(LiveRiskPredictionService._period_name(5), "夜间低峰")
        self.assertEqual(LiveRiskPredictionService._period_name(10), "日间平峰")
        self.assertEqual(LiveRiskPredictionService._period_name(6), "日间平峰")


class HistoricalFlowBaselineTest(unittest.TestCase):
    def test_weekday_and_period_factors(self) -> None:
        segment = {"traffic_flow": 60.0}
        # Monday mid-day
        now = datetime(2026, 7, 13, 12, 0)
        baseline = LiveRiskPredictionService._historical_flow_baseline(segment, now)
        self.assertEqual(baseline, 60.0 * 0.98 * 1.0)

        # Friday evening peak
        now = datetime(2026, 7, 17, 18, 0)
        baseline = LiveRiskPredictionService._historical_flow_baseline(segment, now)
        self.assertEqual(baseline, 60.0 * 1.08 * 1.2)

        # Sunday night
        now = datetime(2026, 7, 19, 3, 0)
        baseline = LiveRiskPredictionService._historical_flow_baseline(segment, now)
        self.assertEqual(baseline, 60.0 * 0.82 * 0.55)

    def test_minimum_flow_is_one(self) -> None:
        now = datetime(2026, 7, 13, 12, 0)
        baseline = LiveRiskPredictionService._historical_flow_baseline({"traffic_flow": 0}, now)
        self.assertGreater(baseline, 0)


class ComparisonTextTest(unittest.TestCase):
    def test_thresholds(self) -> None:
        self.assertEqual(LiveRiskPredictionService._comparison_text(10.0), "偏多")
        self.assertEqual(LiveRiskPredictionService._comparison_text(0.0), "基本持平")
        self.assertEqual(LiveRiskPredictionService._comparison_text(-10.0), "偏少")
        self.assertEqual(LiveRiskPredictionService._comparison_text(9.9), "基本持平")
        self.assertEqual(LiveRiskPredictionService._comparison_text(-9.9), "基本持平")


class WeatherTextTest(unittest.TestCase):
    def test_conditions(self) -> None:
        base = {"temperature_2m": 20.0, "relative_humidity_2m": 65.0, "wind_speed_10m": 10}
        self.assertIn("降雪", LiveRiskPredictionService._weather_text({**base, "snowfall": 1.0, "rain": 0, "precipitation": 0}))
        self.assertIn("降雨", LiveRiskPredictionService._weather_text({**base, "snowfall": 0, "rain": 1.0, "precipitation": 0}))
        self.assertIn("降雨", LiveRiskPredictionService._weather_text({**base, "snowfall": 0, "rain": 0, "precipitation": 0.1}))
        self.assertIn("大风", LiveRiskPredictionService._weather_text({**base, "snowfall": 0, "rain": 0, "precipitation": 0, "wind_speed_10m": 25}))
        self.assertIn("无明显降水", LiveRiskPredictionService._weather_text({**base, "snowfall": 0, "rain": 0, "precipitation": 0, "wind_speed_10m": 10}))

    def test_fallback_note(self) -> None:
        text = LiveRiskPredictionService._weather_text(
            {"snowfall": 0, "rain": 0, "precipitation": 0, "wind_speed_10m": 10, "fallback": True,
             "temperature_2m": 20.0, "relative_humidity_2m": 65.0}
        )
        self.assertIn("（天气服务暂不可用", text)


class FallbackRoadContextTest(unittest.TestCase):
    def test_maps_segment_fields(self) -> None:
        segment = {"name": "海淀南路", "road_type": "primary", "speed_limit": 60, "lane_count": 4}
        ctx = LiveRiskPredictionService._fallback_road_context(segment)

        self.assertEqual(ctx["name"], "海淀南路")
        self.assertEqual(ctx["road_type"], "primary")
        self.assertEqual(ctx["maxspeed"], 60)
        self.assertEqual(ctx["lanes"], 4)
        self.assertTrue(ctx["fallback"])

    def test_defaults_on_missing_fields(self) -> None:
        ctx = LiveRiskPredictionService._fallback_road_context({})

        self.assertEqual(ctx["name"], "")
        self.assertEqual(ctx["road_type"], "unknown")


class CalibrateScoreTest(unittest.TestCase):
    def _weather(self, **kw):
        default = {"temperature_2m": 20, "relative_humidity_2m": 60, "precipitation": 0, "rain": 0, "snowfall": 0,
                   "wind_speed_10m": 10, "weather_code": 0, "source": "open-meteo", "fallback": False}
        default.update(kw)
        return default

    def _vehicle(self, speed=50.0, change=0.0):
        return {"avg_speed_kmh": speed, "flow_change_percent": change, "flow_per_min": 30, "window_vehicle_count": 450,
                "window_minutes": 15, "historical_baseline_count": 450, "flow_comparison": "基本持平",
                "source": "yolo-window"}

    def _road(self, **kw):
        default = {"name": "", "road_type": None, "maxspeed": None, "lanes": None, "surface": None}
        default.update(kw)
        return default

    def _segment(self, **kw):
        default = {"speed_limit": 60, "lane_count": 4, "road_type": "primary", "name": ""}
        default.update(kw)
        return default

    def test_flow_change_caps_at_18_percent(self) -> None:
        vehicle = self._vehicle(change=500.0)
        score = LiveRiskPredictionService._calibrate_score(
            0.4, self._weather(), vehicle, self._road(), self._segment()
        )
        # adjustment from flow: 500/500=1.0 capped→0.18. primary +0.02. lanes≥4 +0.01. = 0.4+0.18+0.02+0.01=0.61
        self.assertAlmostEqual(score, 0.4 + 0.18 + 0.02 + 0.01, places=2)

    def test_negative_flow_change_floored_at_minus_12(self) -> None:
        vehicle = self._vehicle(change=-500.0)
        score = LiveRiskPredictionService._calibrate_score(
            0.4, self._weather(), vehicle, self._road(road_type="residential"), self._segment(road_type="residential")
        )
        # flow floor -0.12. lane≥4 +0.01. = 0.4-0.12+0.01=0.29
        self.assertAlmostEqual(score, 0.4 - 0.12 + 0.01, places=2)

    def test_speed_ratio_adjustments(self) -> None:
        vehicle_slow = self._vehicle(speed=20.0)
        vehicle_medium = self._vehicle(speed=35.0)

        score_slow = LiveRiskPredictionService._calibrate_score(
            0.4, self._weather(), vehicle_slow, self._road(), self._segment()
        )
        score_medium = LiveRiskPredictionService._calibrate_score(
            0.4, self._weather(), vehicle_medium, self._road(), self._segment()
        )

        self.assertGreater(score_slow, score_medium)

    def test_no_speed_limit_skips_ratio_adjustment(self) -> None:
        score = LiveRiskPredictionService._calibrate_score(
            0.4, self._weather(), self._vehicle(), self._road(), self._segment(speed_limit=0)
        )
        # primary +0.02, lanes≥4 +0.01. No speed ratio bonus, no weather bonus. = 0.4 + 0.02 + 0.01 = 0.43
        self.assertAlmostEqual(score, 0.4 + 0.02 + 0.01, places=2)

    def test_weather_bonus_and_lane_bonus(self) -> None:
        weather = self._weather(rain=5.0, snow=0, wind_speed_10m=30)
        score = LiveRiskPredictionService._calibrate_score(
            0.4, weather, self._vehicle(), self._road(lanes="6"), self._segment()
        )
        # flow_change 0 → +0. speed >0.4*60 → no speed bonus. rain → +0.04. motorway → +0.02. lanes≥4 → +0.01
        self.assertAlmostEqual(score, 0.4 + 0.04 + 0.02 + 0.01, places=2)

    def test_non_numeric_speed_limit_and_lanes(self) -> None:
        score = LiveRiskPredictionService._calibrate_score(
            0.4, self._weather(), self._vehicle(), self._road(maxspeed="unknown"), self._segment(speed_limit=0)
        )
        # Lane from road (None) or segment default lane_count=4 → lanes≥4 → +0.01
        # No motorway bonus (road.road_type is None, segment.road_type="primary" → primary IS in set → +0.02)
        self.assertAlmostEqual(score, 0.4 + 0.02 + 0.01, places=2)

    def test_score_clamped_to_zero_one(self) -> None:
        vehicle = self._vehicle(change=500.0)
        score = LiveRiskPredictionService._calibrate_score(
            0.95, self._weather(rain=10), vehicle, self._road(), self._segment()
        )
        self.assertEqual(score, 1.0)

        vehicle = self._vehicle(change=-500.0)
        score = LiveRiskPredictionService._calibrate_score(
            0.05, self._weather(), vehicle, self._road(road_type="liv"), self._segment(road_type="liv")
        )
        self.assertEqual(score, 0.0)


class BuildReasonsExtendedTest(unittest.TestCase):
    def test_historical_baseline_source_note(self) -> None:
        now = datetime(2026, 7, 13, 12, 0)
        vehicle = {
            "window_vehicle_count": 120, "historical_baseline_count": 100,
            "flow_comparison": "偏多", "flow_change_percent": 20.0,
            "avg_speed_kmh": 45.0, "source": "historical-road-baseline",
        }
        weather = {
            "temperature_2m": 26.0, "relative_humidity_2m": 65.0,
            "rain": 0, "precipitation": 0, "snowfall": 0, "wind_speed_10m": 8.0,
            "fallback": False, "weather_code": 0,
        }
        road = {"road_type": "secondary", "maxspeed": 60, "lanes": 2, "surface": "asphalt"}
        segment = {"speed_limit": 60, "lane_count": 2, "historical_accidents_7d": 0}

        reasons = LiveRiskPredictionService._build_reasons(0.5, weather, vehicle, road, segment, now)

        self.assertIn("历史基线估算", reasons[1])
        self.assertEqual(len(reasons), 7)


class BuildModelRowTest(unittest.TestCase):
    def test_speed_mph_conversion_and_sine_cosine(self) -> None:
        service = LiveRiskPredictionService()
        now = datetime(2026, 7, 13, 12, 0)  # Monday, hour=12, minute=0
        segment = {"latitude": 39.97, "longitude": 116.31}
        weather = {"temperature_2m": 25, "relative_humidity_2m": 60, "precipitation": 0, "rain": 0, "snowfall": 0, "wind_speed_10m": 10}
        vehicle = {"flow_per_min": 30, "avg_speed_kmh": 50, "window_vehicle_count": 450}

        row = service._build_model_row(segment, weather, vehicle, None, now)

        self.assertAlmostEqual(row["speed_mph"], 50.0 * 0.621371)
        self.assertEqual(row["hour"], 12)
        self.assertEqual(row["minute"], 0)
        self.assertEqual(row["day_of_week"], 0)  # Monday
        self.assertEqual(row["is_weekend"], 0)
        self.assertAlmostEqual(row["time_sin"], 0.0, places=5)  # sin(pi) = 0
        self.assertAlmostEqual(row["time_cos"], -1.0, places=5)  # cos(pi) = -1
        self.assertEqual(row["borough"], "BEIJING")

    def test_weekend_flag_for_saturday(self) -> None:
        service = LiveRiskPredictionService()
        now = datetime(2026, 7, 18, 15, 30)
        segment = {"latitude": 39.97, "longitude": 116.31}
        weather = {"temperature_2m": 30, "relative_humidity_2m": 50, "precipitation": 0, "rain": 0, "snowfall": 0, "wind_speed_10m": 5}
        vehicle = {"flow_per_min": 20, "avg_speed_kmh": 0, "window_vehicle_count": 300}

        row = service._build_model_row(segment, weather, vehicle, None, now)
        self.assertEqual(row["is_weekend"], 1)
        self.assertEqual(row["speed_mph"], 40.0 * 0.621371)


if __name__ == "__main__":
    unittest.main()
