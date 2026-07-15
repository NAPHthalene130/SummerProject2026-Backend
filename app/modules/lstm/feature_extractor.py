import json
import logging
import math
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

WINDOW_SEC = 15
STEP_SEC = 5
ACC_THRESHOLD = -9.8
TTC_THRESHOLD = 3.0
MAX_WINDOWS_STORED = 4


@dataclass
class RawDetection:
    track_id: int
    frame_id: int
    timestamp_ms: int
    section_id: int
    velocity: float
    acceleration: float
    preceding_id: int
    space_headway: float


@dataclass
class MacroFeatures:
    section_id: int
    time_window: int
    volume: float
    avg_speed: float
    speed_var: float
    density: float
    avg_space_headway: float
    space_headway_var: float
    deceleration_freq: float
    avg_drac: float
    total_conflicts: float
    road_type: int

    def to_array(self) -> list[float]:
        return [
            self.volume,
            self.avg_speed,
            self.speed_var,
            self.density,
            self.avg_space_headway,
            self.space_headway_var,
            self.deceleration_freq,
            float(self.road_type),
        ]


class StandardScalerWrapper:
    def __init__(self, params_file: Optional[str] = None):
        self.mean_: np.ndarray = np.zeros(8, dtype=np.float64)
        self.scale_: np.ndarray = np.ones(8, dtype=np.float64)
        self._loaded = False
        if params_file and os.path.exists(params_file):
            self._load(params_file)

    def _load(self, filepath: str):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.mean_ = np.array(data["mean"], dtype=np.float64)
            self.scale_ = np.array(data["scale"], dtype=np.float64)
            self._loaded = True
            logger.info("StandardScaler loaded from %s", filepath)
        except Exception as e:
            logger.warning("Failed to load StandardScaler from %s: %s, using defaults", filepath, e)
            self._use_defaults()

    def _use_defaults(self):
        self.mean_ = np.array([15.0, 15.0, 20.0, 0.3, 50.0, 1000.0, 0.02, 0.5], dtype=np.float64)
        self.scale_ = np.array([10.0, 8.0, 25.0, 0.2, 30.0, 1500.0, 0.03, 0.5], dtype=np.float64)
        logger.warning("Using fallback StandardScaler defaults -- predictions may be inaccurate")

    def transform(self, features: list[float]) -> np.ndarray:
        arr = np.array(features, dtype=np.float64)
        return (arr - self.mean_) / np.maximum(self.scale_, 1e-8)

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class TrafficFeatureExtractor:
    def __init__(self, scaler: StandardScalerWrapper):
        self.scaler = scaler
        self._buffers: dict[str, dict[int, deque[RawDetection]]] = defaultdict(
            lambda: defaultdict(lambda: deque[RawDetection]())
        )
        self._window_queues: dict[str, dict[int, deque[MacroFeatures]]] = {}
        self._last_agg_time: dict[str, float] = {}
        self._window_counter: dict[str, int] = {}

    def add_detection(self, camera_id: str, detection: RawDetection):
        sec_id = detection.section_id
        self._buffers[camera_id][sec_id].append(detection)
        max_age = (WINDOW_SEC + 5) * 1000
        self._prune_old(camera_id, sec_id, detection.timestamp_ms, max_age)

    def _prune_old(self, camera_id: str, section_id: int, current_ms: int, max_age_ms: int):
        buf = self._buffers[camera_id][section_id]
        while buf and (current_ms - buf[0].timestamp_ms) > max_age_ms:
            buf.popleft()

    def should_aggregate(self, camera_id: str) -> bool:
        now = time.time()
        last = self._last_agg_time.get(camera_id, 0.0)
        if now - last >= STEP_SEC:
            self._last_agg_time[camera_id] = now
            return True
        return False

    def aggregate_and_normalize(self, camera_id: str, road_type: int = 0) -> Optional[list[float]]:
        now_ms = int(time.time() * 1000)
        min_ms = now_ms - (WINDOW_SEC * 1000)

        all_features: list[MacroFeatures] = []

        for section_id, buf in self._buffers.get(camera_id, {}).items():
            window_dets = [d for d in buf if d.timestamp_ms >= min_ms]
            if len(window_dets) < 3:
                continue

            window_dets.sort(key=lambda d: d.timestamp_ms)

            processed = self._calculate_micro_metrics(window_dets)
            macro = self._aggregate_window(processed, section_id, road_type)
            if macro:
                all_features.append(macro)

        if not all_features:
            return None

        combined = self._combine_sections(all_features, road_type)
        window_id = self._window_counter.get(camera_id, 0)
        self._window_counter[camera_id] = window_id + 1

        if camera_id not in self._window_queues:
            self._window_queues[camera_id] = {}
        for mf in all_features:
            if mf.section_id not in self._window_queues[camera_id]:
                self._window_queues[camera_id][mf.section_id] = deque[MacroFeatures]()
            q = self._window_queues[camera_id][mf.section_id]
            q.append(mf)
            while len(q) > MAX_WINDOWS_STORED:
                q.popleft()

        return combined

    def get_lstm_input(self, camera_id: str) -> Optional[np.ndarray]:
        queue_map = self._window_queues.get(camera_id, {})
        if not queue_map:
            return None

        for section_id in queue_map:
            q = queue_map[section_id]
            if len(q) < MAX_WINDOWS_STORED:
                continue
            sequence = []
            for mf in q:
                arr = mf.to_array()
                normalized = self.scaler.transform(arr)
                sequence.append(normalized.tolist())
            return np.array(sequence, dtype=np.float32)

        return None

    def _calculate_micro_metrics(self, detections: list[RawDetection]) -> list[dict]:
        processed: list[dict] = []
        vel_map: dict[int, float] = {}
        for d in detections:
            vel_map[d.track_id] = d.velocity

        for d in detections:
            preceding_vel = vel_map.get(d.preceding_id, 0.0) if d.preceding_id >= 0 else 0.0
            delta_v = d.velocity - preceding_vel

            ttc = float("nan")
            drac = float("nan")
            risk_flag = 0

            if delta_v > 0 and d.space_headway > 0:
                ttc = d.space_headway / delta_v
                drac = (delta_v * delta_v) / (2.0 * d.space_headway)
                if 0 < ttc < TTC_THRESHOLD:
                    risk_flag = 1

            processed.append({
                "track_id": d.track_id,
                "frame_id": d.frame_id,
                "timestamp_ms": d.timestamp_ms,
                "section_id": d.section_id,
                "velocity": d.velocity,
                "acceleration": d.acceleration,
                "preceding_id": d.preceding_id,
                "space_headway": d.space_headway,
                "ttc": ttc,
                "drac": drac,
                "risk_flag": risk_flag,
            })

        return processed

    def _aggregate_window(
        self, detections: list[dict], section_id: int, road_type: int
    ) -> Optional[MacroFeatures]:
        n = len(detections)
        if n < 2:
            return None

        unique_ids = {d["track_id"] for d in detections}
        velocities = [d["velocity"] for d in detections if not math.isnan(d["velocity"])]
        headways = [d["space_headway"] for d in detections if not math.isnan(d["space_headway"]) and d["space_headway"] > 0]
        accelerations = [d["acceleration"] for d in detections if not math.isnan(d["acceleration"])]
        dracs = [d["drac"] for d in detections if not math.isnan(d["drac"]) and d["drac"] > 0]
        risks = [d["risk_flag"] for d in detections]

        if not velocities:
            return None

        avg_speed = float(np.mean(velocities))
        speed_var = float(np.var(velocities)) if len(velocities) > 1 else 0.0
        avg_headway = float(np.mean(headways)) if headways else 0.0
        headway_var = float(np.var(headways)) if len(headways) > 1 else 0.0
        volume = float(len(unique_ids))
        decel_freq = sum(1 for a in accelerations if a < ACC_THRESHOLD) / len(accelerations) if accelerations else 0.0
        avg_drac = float(np.mean(dracs)) if dracs else 0.0
        total_conflicts = float(sum(risks))
        density = volume / avg_headway if avg_headway > 0 else 0.0

        return MacroFeatures(
            section_id=section_id,
            time_window=0,
            volume=volume,
            avg_speed=avg_speed,
            speed_var=speed_var,
            density=density,
            avg_space_headway=avg_headway,
            space_headway_var=headway_var,
            deceleration_freq=decel_freq,
            avg_drac=avg_drac,
            total_conflicts=total_conflicts,
            road_type=road_type,
        )

    def _combine_sections(self, features: list[MacroFeatures], road_type: int) -> list[float]:
        if len(features) == 1:
            return features[0].to_array()

        total_volume = sum(f.volume for f in features)
        total_speed_sum = sum(f.avg_speed * f.volume for f in features)
        avg_speed = total_speed_sum / total_volume if total_volume > 0 else 0.0
        speed_var = np.mean([f.speed_var for f in features]) if features else 0.0
        headway_list = [f.avg_space_headway for f in features if f.avg_space_headway > 0]
        avg_headway = float(np.mean(headway_list)) if headway_list else 50.0
        headway_var = np.mean([f.space_headway_var for f in features]) if features else 0.0
        density = total_volume / avg_headway if avg_headway > 0 else 0.0
        decel_freq = np.mean([f.deceleration_freq for f in features]) if features else 0.0
        drac_list = [f.avg_drac for f in features if f.avg_drac > 0]
        avg_drac = float(np.mean(drac_list)) if drac_list else 0.0
        total_conflicts = sum(f.total_conflicts for f in features)

        combined = MacroFeatures(
            section_id=0,
            time_window=0,
            volume=float(total_volume),
            avg_speed=float(avg_speed),
            speed_var=float(speed_var),
            density=float(density),
            avg_space_headway=float(avg_headway),
            space_headway_var=float(headway_var),
            deceleration_freq=float(decel_freq),
            avg_drac=float(avg_drac) if not math.isnan(avg_drac) else 0.0,
            total_conflicts=float(total_conflicts),
            road_type=road_type,
        )
        return combined.to_array()
