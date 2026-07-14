from __future__ import annotations

import asyncio
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import joblib
import numpy as np
import pandas as pd

from app.config import settings
from app.modules.camera_data import CameraDataStore


MODEL_COLUMNS = [
    "grid_lat",
    "grid_lon",
    "borough",
    "speed_mph",
    "traffic_volume_typical",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "wind_speed_10m",
    "crashes_last_24h",
    "crashes_last_7d",
    "hour",
    "minute",
    "day_of_week",
    "is_weekend",
    "time_sin",
    "time_cos",
    "dow_sin",
    "dow_cos",
]

PREDICTION_WINDOW_MINUTES = 15
PREDICTION_WINDOW_SECONDS = PREDICTION_WINDOW_MINUTES * 60
WEEKDAY_NAMES = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


@dataclass
class _CacheEntry:
    value: dict[str, Any]
    expires_at: float


class LiveRiskPredictionService:
    def __init__(self) -> None:
        default_model = Path(__file__).resolve().parent / "models" / "catboost_risk_model.joblib"
        configured = settings.RISK_MODEL_PATH
        self.model_path = Path(configured) if configured else default_model
        self._model: Any | None = None
        self._model_lock = threading.Lock()
        self._weather_cache: dict[str, _CacheEntry] = {}
        self._road_cache: dict[str, _CacheEntry] = {}

    def _get_model(self) -> Any:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    if not self.model_path.exists():
                        raise FileNotFoundError(f"risk model not found: {self.model_path}")
                    self._model = joblib.load(self.model_path)
        return self._model

    async def predict(self, segments: list[dict[str, Any]], selected_segment_id: str | None = None) -> dict[str, Any]:
        if not segments:
            now = datetime.now()
            return {
                "generated_at": now.isoformat(timespec="seconds"),
                "model": self.model_path.name,
                "forecast_minutes": PREDICTION_WINDOW_MINUTES,
                "aggregation_window_minutes": PREDICTION_WINDOW_MINUTES,
                "date_context": self._date_context(now),
                "weather": {},
                "predictions": [],
            }

        center_lat = float(np.mean([float(item["latitude"]) for item in segments]))
        center_lon = float(np.mean([float(item["longitude"]) for item in segments]))
        weather = await self._fetch_weather(center_lat, center_lon)

        selected_context: dict[str, Any] | None = None
        selected = next((item for item in segments if item.get("segment_id") == selected_segment_id), None)
        if selected is not None:
            selected_context = await self._fetch_road_context(
                float(selected["latitude"]),
                float(selected["longitude"]),
                str(selected.get("name") or ""),
                str(selected.get("road_type") or "unknown"),
            )

        now = datetime.now()
        rows: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        for segment in segments:
            vehicle = self._vehicle_context(segment, now)
            road_context = selected_context if segment.get("segment_id") == selected_segment_id else None
            row = self._build_model_row(segment, weather, vehicle, road_context, now)
            rows.append(row)
            evidence.append(
                {
                    "segment_id": str(segment["segment_id"]),
                    "vehicle": vehicle,
                    "road": road_context or self._fallback_road_context(segment),
                }
            )

        frame = pd.DataFrame(rows, columns=MODEL_COLUMNS)
        scores = await asyncio.to_thread(self._predict_frame, frame)

        predictions = []
        for segment, raw_score, item_evidence in zip(segments, scores, evidence):
            score = self._calibrate_score(
                float(raw_score),
                weather,
                item_evidence["vehicle"],
                item_evidence["road"],
                segment,
            )
            reasons = self._build_reasons(
                score,
                weather,
                item_evidence["vehicle"],
                item_evidence["road"],
                segment,
                now,
            )
            predictions.append(
                {
                    "segment_id": str(segment["segment_id"]),
                    "risk_score": float(score),
                    "model_score": float(raw_score),
                    "risk_level": self._risk_level(float(score)),
                    "reason": reasons,
                    **item_evidence,
                }
            )

        return {
            "generated_at": now.isoformat(timespec="seconds"),
            "model": self.model_path.name,
            "forecast_minutes": PREDICTION_WINDOW_MINUTES,
            "aggregation_window_minutes": PREDICTION_WINDOW_MINUTES,
            "date_context": self._date_context(now),
            "weather": weather,
            "predictions": predictions,
        }

    def _predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self._get_model().predict(frame)
        return np.clip(np.asarray(raw, dtype=float), 0.0, 1.0)

    async def _fetch_weather(self, latitude: float, longitude: float) -> dict[str, Any]:
        key = f"{latitude:.3f},{longitude:.3f}"
        cached = self._weather_cache.get(key)
        if cached and cached.expires_at > time.time():
            return cached.value
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,relative_humidity_2m,precipitation,rain,"
                "snowfall,wind_speed_10m,weather_code"
            ),
            "timezone": "Asia/Shanghai",
        }
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                response = await client.get("https://api.open-meteo.com/v1/forecast", params=params)
                response.raise_for_status()
                current = response.json().get("current", {})
            value = {
                "temperature_2m": float(current.get("temperature_2m", 20.0)),
                "relative_humidity_2m": float(current.get("relative_humidity_2m", 60.0)),
                "precipitation": float(current.get("precipitation", 0.0)),
                "rain": float(current.get("rain", 0.0)),
                "snowfall": float(current.get("snowfall", 0.0)),
                "wind_speed_10m": float(current.get("wind_speed_10m", 8.0)),
                "weather_code": int(current.get("weather_code", 0)),
                "source": "open-meteo",
                "fallback": False,
            }
        except Exception as exc:
            value = {
                "temperature_2m": 20.0,
                "relative_humidity_2m": 60.0,
                "precipitation": 0.0,
                "rain": 0.0,
                "snowfall": 0.0,
                "wind_speed_10m": 8.0,
                "weather_code": 0,
                "source": "fallback",
                "fallback": True,
                "error": str(exc),
            }
        self._weather_cache[key] = _CacheEntry(value, time.time() + 600)
        return value

    async def _fetch_road_context(
        self, latitude: float, longitude: float, fallback_name: str, fallback_type: str
    ) -> dict[str, Any]:
        key = f"{latitude:.5f},{longitude:.5f}"
        cached = self._road_cache.get(key)
        if cached and cached.expires_at > time.time():
            return cached.value
        try:
            overpass_query = (
                f"[out:json];way(around:100,{latitude},{longitude})[highway];out tags 10;"
            )
            async with httpx.AsyncClient(
                timeout=12.0,
                headers={"User-Agent": "SummerProject2026-RiskPrediction/1.0"},
            ) as client:
                overpass_response, geocode_response = await asyncio.gather(
                    client.get(
                        "https://overpass-api.de/api/interpreter",
                        params={"data": overpass_query},
                    ),
                    client.get(
                        "https://api.bigdatacloud.net/data/reverse-geocode-client",
                        params={
                            "latitude": latitude,
                            "longitude": longitude,
                            "localityLanguage": "zh",
                        },
                    ),
                )
                overpass_response.raise_for_status()
                geocode_response.raise_for_status()
            elements = overpass_response.json().get("elements") or []
            named = next((item for item in elements if (item.get("tags") or {}).get("name")), None)
            road_tags = (named or (elements[0] if elements else {})).get("tags") or {}
            geocode = geocode_response.json()
            locality = (
                geocode.get("locality")
                or geocode.get("city")
                or geocode.get("principalSubdivision")
                or "BEIJING"
            )
            road_name = road_tags.get("name:zh") or road_tags.get("name") or fallback_name
            display_parts = [
                geocode.get("principalSubdivision"),
                geocode.get("city"),
                locality,
                road_name,
            ]
            value = {
                "name": road_name,
                "display_name": " · ".join(dict.fromkeys(str(item) for item in display_parts if item)),
                "road_type": road_tags.get("highway") or fallback_type,
                "district": locality,
                "maxspeed": road_tags.get("maxspeed"),
                "lanes": road_tags.get("lanes"),
                "surface": road_tags.get("surface"),
                "source": "openstreetmap-overpass+bigdatacloud",
                "fallback": False,
            }
        except Exception as exc:
            value = {
                "name": fallback_name,
                "display_name": fallback_name,
                "road_type": fallback_type,
                "district": "BEIJING",
                "maxspeed": None,
                "lanes": None,
                "surface": None,
                "source": "frontend-road-metadata",
                "fallback": True,
                "error": str(exc),
            }
        self._road_cache[key] = _CacheEntry(value, time.time() + 3600)
        return value

    def _vehicle_context(self, segment: dict[str, Any], now: datetime) -> dict[str, Any]:
        store = CameraDataStore()
        camera_ids = segment.get("camera_ids") or []
        backend_ids = [self._backend_camera_id(item) for item in camera_ids]
        instantaneous_counts: list[int] = []
        window_counts: list[int] = []
        speeds: list[tuple[float, int]] = []
        observed_seconds: list[float] = []
        active_incidents = 0
        for camera_id in backend_ids:
            data = store.get_by_camera_id(camera_id)
            if data is not None:
                instantaneous_counts.append(int(data.total_vehicle_count))
            metrics = store.get_prediction_window_metrics(camera_id, PREDICTION_WINDOW_SECONDS)
            window_counts.append(int(metrics["cumulative_vehicle_count"] or 0))
            observed_seconds.append(float(metrics["observed_seconds"] or 0.0))
            if metrics.get("avg_speed_kmh") is not None:
                speeds.append((float(metrics["avg_speed_kmh"]), max(1, int(metrics["sample_count"] or 0))))
            active_incidents += len(store.get_active_incident_types(camera_id))

        observed_minutes = max(observed_seconds, default=0.0) / 60.0
        observed_vehicle_count = int(sum(window_counts))
        baseline_flow = self._historical_flow_baseline(segment, now)
        baseline_window_count = baseline_flow * PREDICTION_WINDOW_MINUTES
        has_stable_window = observed_minutes >= 1.0 and observed_vehicle_count > 0
        if has_stable_window:
            flow_per_min = observed_vehicle_count / observed_minutes
            window_vehicle_count = flow_per_min * PREDICTION_WINDOW_MINUTES
            flow_source = "yolo-window"
        else:
            flow_per_min = baseline_flow
            window_vehicle_count = baseline_window_count
            flow_source = "historical-road-baseline"

        speed_weight = sum(weight for _, weight in speeds)
        window_speed = (
            sum(speed * weight for speed, weight in speeds) / speed_weight
            if speed_weight > 0
            else float(segment.get("avg_speed") or segment.get("speed_limit") or 40.0)
        )
        change_percent = (
            (window_vehicle_count - baseline_window_count) / baseline_window_count * 100.0
            if baseline_window_count > 0
            else 0.0
        )
        return {
            "camera_ids": backend_ids,
            "vehicle_count": int(round(window_vehicle_count)),
            "instantaneous_vehicle_count": int(sum(instantaneous_counts)),
            "cumulative_vehicle_count": observed_vehicle_count,
            "window_vehicle_count": int(round(window_vehicle_count)),
            "window_minutes": PREDICTION_WINDOW_MINUTES,
            "observed_minutes": round(observed_minutes, 1),
            "flow_per_min": round(flow_per_min, 1),
            "historical_baseline_count": int(round(baseline_window_count)),
            "flow_change_percent": round(change_percent, 1),
            "flow_comparison": self._comparison_text(change_percent),
            "avg_speed_kmh": float(window_speed),
            "active_incidents": active_incidents,
            "source": flow_source,
        }

    @staticmethod
    def _backend_camera_id(camera_id: str) -> str:
        text = str(camera_id)
        if text.lower().startswith("cam-"):
            return text.lower()
        digits = "".join(ch for ch in text if ch.isdigit())
        return f"cam-{int(digits):02d}" if digits else text

    def _build_model_row(
        self,
        segment: dict[str, Any],
        weather: dict[str, Any],
        vehicle: dict[str, Any],
        road: dict[str, Any] | None,
        now: datetime,
    ) -> dict[str, Any]:
        traffic_volume = max(float(vehicle["flow_per_min"]), 1.0)
        speed_kmh = vehicle.get("avg_speed_kmh")
        if not speed_kmh or speed_kmh <= 0:
            speed_kmh = float(segment.get("avg_speed") or segment.get("speed_limit") or 40.0)
        minute_of_day = now.hour * 60 + now.minute
        day_of_week = now.weekday()
        return {
            "grid_lat": float(segment["latitude"]),
            "grid_lon": float(segment["longitude"]),
            "borough": str((road or {}).get("district") or "BEIJING").upper(),
            "speed_mph": float(speed_kmh) * 0.621371,
            "traffic_volume_typical": traffic_volume,
            "temperature_2m": weather["temperature_2m"],
            "relative_humidity_2m": weather["relative_humidity_2m"],
            "precipitation": weather["precipitation"],
            "rain": weather["rain"],
            "snowfall": weather["snowfall"],
            "wind_speed_10m": weather["wind_speed_10m"],
            "crashes_last_24h": float(segment.get("historical_accidents_24h") or 0.0),
            "crashes_last_7d": float(segment.get("historical_accidents_7d") or 0.0),
            "hour": now.hour,
            "minute": now.minute,
            "day_of_week": day_of_week,
            "is_weekend": int(day_of_week >= 5),
            "time_sin": math.sin(2 * math.pi * minute_of_day / 1440.0),
            "time_cos": math.cos(2 * math.pi * minute_of_day / 1440.0),
            "dow_sin": math.sin(2 * math.pi * day_of_week / 7.0),
            "dow_cos": math.cos(2 * math.pi * day_of_week / 7.0),
        }

    @staticmethod
    def _fallback_road_context(segment: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": segment.get("name") or "",
            "display_name": segment.get("name") or "",
            "road_type": segment.get("road_type") or "unknown",
            "district": "BEIJING",
            "maxspeed": segment.get("speed_limit"),
            "lanes": segment.get("lane_count"),
            "surface": None,
            "source": "frontend-road-metadata",
            "fallback": True,
        }

    @staticmethod
    def _calibrate_score(
        model_score: float,
        weather: dict[str, Any],
        vehicle: dict[str, Any],
        road: dict[str, Any],
        segment: dict[str, Any],
    ) -> float:
        """Blend model output with road metadata absent from the trained feature set."""
        adjustment = max(-0.12, min(0.18, float(vehicle["flow_change_percent"]) / 500.0))
        speed_limit_value = road.get("maxspeed") or segment.get("speed_limit") or 0
        try:
            speed_limit = float(str(speed_limit_value).split()[0])
        except (TypeError, ValueError):
            speed_limit = 0.0
        if speed_limit > 0:
            speed_ratio = float(vehicle["avg_speed_kmh"]) / speed_limit
            if speed_ratio < 0.4:
                adjustment += 0.06
            elif speed_ratio < 0.65:
                adjustment += 0.03
        if weather["rain"] > 0 or weather["snowfall"] > 0 or weather["wind_speed_10m"] >= 25:
            adjustment += 0.04
        if (road.get("road_type") or segment.get("road_type")) in {
            "motorway",
            "trunk",
            "primary",
            "main",
        }:
            adjustment += 0.02
        lane_value = road.get("lanes") or segment.get("lane_count") or 0
        try:
            lane_count = float(str(lane_value).split()[0])
        except (TypeError, ValueError):
            lane_count = 0.0
        if lane_count >= 4:
            adjustment += 0.01
        return float(np.clip(model_score + adjustment, 0.0, 1.0))

    @staticmethod
    def _risk_level(score: float) -> str:
        if score < 0.3:
            return "normal"
        if score < 0.5:
            return "busy"
        return "danger"

    @staticmethod
    def _date_context(now: datetime) -> dict[str, Any]:
        weekday_index = now.weekday()
        return {
            "date": now.strftime("%Y-%m-%d"),
            "weekday": WEEKDAY_NAMES[weekday_index],
            "weekday_index": weekday_index,
            "is_weekend": weekday_index >= 5,
            "period": LiveRiskPredictionService._period_name(now.hour),
        }

    @staticmethod
    def _period_name(hour: int) -> str:
        if 7 <= hour < 10:
            return "早高峰"
        if 17 <= hour < 20:
            return "晚高峰"
        if 0 <= hour < 6:
            return "夜间低峰"
        return "日间平峰"

    @staticmethod
    def _historical_flow_baseline(segment: dict[str, Any], now: datetime) -> float:
        base_flow = max(float(segment.get("traffic_flow") or 1.0), 1.0)
        weekday_factor = (0.98, 1.0, 1.0, 1.02, 1.08, 0.9, 0.82)[now.weekday()]
        if 7 <= now.hour < 10 or 17 <= now.hour < 20:
            period_factor = 1.2
        elif 0 <= now.hour < 6:
            period_factor = 0.55
        else:
            period_factor = 1.0
        return base_flow * weekday_factor * period_factor

    @staticmethod
    def _comparison_text(change_percent: float) -> str:
        if change_percent >= 10:
            return "偏多"
        if change_percent <= -10:
            return "偏少"
        return "基本持平"

    @staticmethod
    def _weather_text(weather: dict[str, Any]) -> str:
        if weather["snowfall"] > 0:
            condition = "降雪"
        elif weather["rain"] > 0 or weather["precipitation"] > 0:
            condition = "降雨"
        elif weather["wind_speed_10m"] >= 25:
            condition = "大风"
        else:
            condition = "无明显降水"
        fallback_note = "（天气服务暂不可用，采用默认值）" if weather.get("fallback") else ""
        return (
            f"天气为{condition}，气温 {weather['temperature_2m']:.1f}℃，"
            f"湿度 {weather['relative_humidity_2m']:.0f}%，"
            f"风速 {weather['wind_speed_10m']:.1f} km/h{fallback_note}"
        )

    @staticmethod
    def _build_reasons(
        score: float,
        weather: dict[str, Any],
        vehicle: dict[str, Any],
        road: dict[str, Any],
        segment: dict[str, Any],
        now: datetime,
    ) -> list[str]:
        date_context = LiveRiskPredictionService._date_context(now)
        baseline = int(vehicle["historical_baseline_count"])
        window_count = int(vehicle["window_vehicle_count"])
        flow_source_note = (
            "基于 YOLO 轨迹窗口统计"
            if vehicle["source"] == "yolo-window"
            else "当前观测不足 1 分钟，暂采用道路历史基线估算"
        )
        maxspeed = road.get("maxspeed") or segment.get("speed_limit") or "未知"
        lanes = road.get("lanes") or segment.get("lane_count") or "未知"
        surface = road.get("surface") or "路面信息未知"
        historical_accidents = int(segment.get("historical_accidents_7d") or 0)
        return [
            (
                f"日期：今天是 {date_context['date']} {date_context['weekday']}，"
                f"属于{'周末' if date_context['is_weekend'] else '工作日'}的{date_context['period']}时段"
            ),
            (
                f"累计车流：近 {PREDICTION_WINDOW_MINUTES} 分钟约 {window_count} 辆，"
                f"相较历史同星期同时段基线 {baseline} 辆{vehicle['flow_comparison']} "
                f"{abs(float(vehicle['flow_change_percent'])):.1f}%（{flow_source_note}）"
            ),
            (
                f"窗口速度：近 {PREDICTION_WINDOW_MINUTES} 分钟车辆加权平均速度 "
                f"{float(vehicle['avg_speed_kmh']):.1f} km/h，路段限速 {maxspeed} km/h"
            ),
            f"天气：{LiveRiskPredictionService._weather_text(weather)}",
            (
                f"道路条件：{road.get('road_type') or segment.get('road_type') or 'unknown'} 类型道路，"
                f"{lanes} 车道，路面 {surface}"
            ),
            f"历史事故：近 7 天记录 {historical_accidents} 起（无记录时按 0 起输入模型）",
            f"综合上述日期、累计车流、窗口速度、天气、道路和历史事故因素，模型预测风险 {score * 100:.1f}%",
        ]


live_risk_prediction_service = LiveRiskPredictionService()
