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
            return {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "model": self.model_path.name,
                "forecast_minutes": 15,
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
            vehicle = self._vehicle_context(segment.get("camera_ids") or [])
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
        for segment, score, item_evidence in zip(segments, scores, evidence):
            reasons = self._build_reasons(score, weather, item_evidence["vehicle"], item_evidence["road"])
            predictions.append(
                {
                    "segment_id": str(segment["segment_id"]),
                    "risk_score": float(score),
                    "risk_level": self._risk_level(float(score)),
                    "reason": reasons,
                    **item_evidence,
                }
            )

        return {
            "generated_at": now.isoformat(timespec="seconds"),
            "model": self.model_path.name,
            "forecast_minutes": 15,
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

    def _vehicle_context(self, camera_ids: list[str]) -> dict[str, Any]:
        store = CameraDataStore()
        backend_ids = [self._backend_camera_id(item) for item in camera_ids]
        counts: list[int] = []
        speeds: list[float] = []
        active_incidents = 0
        for camera_id in backend_ids:
            data = store.get_by_camera_id(camera_id)
            if data is not None:
                counts.append(int(data.total_vehicle_count))
            metrics = store.get_traffic_metrics(camera_id)
            if metrics and metrics.get("avg_speed_kmh") is not None:
                speeds.append(float(metrics["avg_speed_kmh"]))
            active_incidents += len(store.get_active_incident_types(camera_id))
        return {
            "camera_ids": backend_ids,
            "vehicle_count": int(sum(counts)),
            "avg_speed_kmh": float(np.mean(speeds)) if speeds else None,
            "active_incidents": active_incidents,
            "source": "yolo-camera-store" if counts else "segment-fallback",
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
        vehicle_count = int(vehicle["vehicle_count"])
        fallback_flow = float(segment.get("traffic_flow") or 60.0)
        traffic_volume = max(float(vehicle_count * 12), fallback_flow if vehicle_count == 0 else 1.0)
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
            "crashes_last_24h": float(vehicle["active_incidents"]),
            "crashes_last_7d": float(vehicle["active_incidents"]),
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
    def _risk_level(score: float) -> str:
        if score < 0.3:
            return "normal"
        if score < 0.5:
            return "busy"
        return "danger"

    @staticmethod
    def _build_reasons(
        score: float,
        weather: dict[str, Any],
        vehicle: dict[str, Any],
        road: dict[str, Any],
    ) -> list[str]:
        reasons: list[str] = []
        if vehicle["vehicle_count"] >= 10:
            reasons.append("YOLO 检测车辆较多")
        if vehicle.get("avg_speed_kmh") is not None and vehicle["avg_speed_kmh"] < 20:
            reasons.append("车辆平均速度偏低")
        if weather["rain"] > 0 or weather["precipitation"] > 0:
            reasons.append("当前存在降雨")
        if weather["snowfall"] > 0:
            reasons.append("当前存在降雪")
        if weather["wind_speed_10m"] >= 25:
            reasons.append("风速较高")
        if vehicle["active_incidents"] > 0:
            reasons.append("摄像头存在活动异常事件")
        if road.get("road_type") in {"motorway", "trunk", "primary", "main"}:
            reasons.append("主干道路交通暴露度较高")
        if not reasons:
            reasons.append("当前天气与车辆状态平稳")
        reasons.append(f"模型预测风险 {score * 100:.1f}%")
        return reasons


live_risk_prediction_service = LiveRiskPredictionService()
