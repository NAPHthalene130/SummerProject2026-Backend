import logging
import time
from typing import Optional

import numpy as np

from app.modules.yolo.RoadDetector import RoadDetector

logger = logging.getLogger(__name__)


class SpeedCalculator:
    def __init__(self, road_detector: RoadDetector):
        self.road = road_detector
        self._prev_positions: dict[str, dict[int, dict]] = {}
        self._prev_velocities: dict[str, dict[int, float]] = {}
        self._prev_timestamps: dict[str, int] = {}

    def unregister(self, cam_id: str) -> None:
        self._prev_positions.pop(cam_id, None)
        self._prev_velocities.pop(cam_id, None)
        self._prev_timestamps.pop(cam_id, None)

    def compute(
        self,
        cam_id: str,
        track_id: int,
        cx: float,
        cy: float,
        bbox: list[float],
        class_name: str,
        confidence: float,
        detections_list: list[dict],
        current_positions: dict[int, dict],
    ) -> dict:
        now_ms = int(time.time() * 1000)
        prev_ms = self._prev_timestamps.get(cam_id, now_ms - 33)
        self._prev_timestamps[cam_id] = now_ms
        dt_sec = max((now_ms - prev_ms) / 1000.0, 0.001)

        if cam_id not in self._prev_positions:
            self._prev_positions[cam_id] = {}
        if cam_id not in self._prev_velocities:
            self._prev_velocities[cam_id] = {}

        prev = self._prev_positions[cam_id].get(track_id)
        if prev:
            velocity = self.road.pixel_distance_to_meters(prev["cy"], cy, abs(cx - prev["cx"]))
            velocity = velocity / dt_sec
        else:
            velocity = 0.0
        prev_vel = self._prev_velocities[cam_id].get(track_id, velocity)
        acceleration = (velocity - prev_vel) / dt_sec

        section_id = 0
        if prev and cx < prev["cx"]:
            section_id = 1

        preceding_id, headway = -1, 0.0
        for other_id, other in current_positions.items():
            if other_id == track_id:
                continue
            dy = cy - other["cy"]
            if dy > 0:
                dist = abs(dy)
                if dist < (headway if headway > 0 else float("inf")):
                    headway = dist
                    preceding_id = other_id
        space_headway = self.road.pixel_distance_to_meters(cy, other.get("cy", cy), headway) if preceding_id >= 0 else 0.0

        self._prev_velocities[cam_id][track_id] = velocity
        self._prev_positions[cam_id][track_id] = {"cx": cx, "cy": cy}

        return {
            "track_id": track_id,
            "class_name": class_name,
            "confidence": confidence,
            "bbox": bbox,
            "velocity": round(velocity, 2),
            "acceleration": round(acceleration, 2),
            "section_id": section_id,
            "preceding_id": preceding_id,
            "space_headway": round(space_headway, 2),
        }

    def update_positions(self, cam_id: str, current: dict[int, dict]) -> None:
        self._prev_positions[cam_id] = current
