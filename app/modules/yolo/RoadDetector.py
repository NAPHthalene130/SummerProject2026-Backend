import logging
from enum import Enum
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

LANE_CLASSES = {0, 2, 3, 5, 7}


class RoadOrientation(Enum):
    LONGITUDINAL = "longitudinal"
    TRANSVERSE = "transverse"


class LaneInfo:
    def __init__(self, poly: np.ndarray, angle: float):
        self.poly = poly
        self.angle = angle


class RoadDetector:
    def __init__(self, model_path: str = "yolo11n-seg.pt", known_lane_width_m: float = 3.5):
        self.known_lane_width_m = known_lane_width_m
        self.model = YOLO(model_path)
        self.vp_x: Optional[float] = None
        self.vp_y: Optional[float] = None
        self.scale_rows: Optional[np.ndarray] = None
        self.scale_cols: Optional[np.ndarray] = None
        self.orientation: Optional[RoadOrientation] = None
        self.frame_h: int = 0
        self.frame_w: int = 0
        self._calibrated = False
        self._lane_count = 0

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def detect_lanes(self, frame: np.ndarray) -> list[LaneInfo]:
        results = self.model(frame, verbose=False)[0]
        lanes: list[LaneInfo] = []
        if results.masks is None:
            return lanes
        h, w = frame.shape[:2]
        road_mask = np.zeros((h, w), dtype=np.uint8)
        road_box_mask = np.zeros((h, w), dtype=np.uint8)
        for i, cls_id in enumerate(results.boxes.cls.int().tolist() if results.boxes is not None else []):
            if cls_id in LANE_CLASSES:
                xy = results.masks.xy[i]
                poly = np.array([(int(p[0]), int(p[1])) for p in xy], dtype=np.int32)
                cv2.fillPoly(road_mask, [poly], 255)
                x1, y1, x2, y2 = results.boxes.xyxy[i].int().tolist()
                cv2.rectangle(road_box_mask, (x1, y1), (x2, y2), 255, -1)
        road_area = cv2.bitwise_or(road_mask, road_box_mask)
        road_area = cv2.dilate(road_area, np.ones((21, 21), np.uint8), iterations=2)
        road_area = cv2.erode(road_area, np.ones((5, 5), np.uint8), iterations=1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        equalized = cv2.equalizeHist(blur)
        edges = cv2.Canny(equalized, 30, 100)
        roi = cv2.bitwise_and(edges, road_area)
        lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 60, minLineLength=50, maxLineGap=60)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx, dy = x2 - x1, y2 - y1
                length = np.sqrt(dx * dx + dy * dy)
                if length < 40:
                    continue
                angle = np.degrees(np.arctan2(dy, dx))
                poly = np.array([(x1, y1), (x2, y2)], dtype=np.int32)
                lanes.append(LaneInfo(poly, angle))
        return lanes

    def _lane_orientations(self, lanes: list[LaneInfo]) -> tuple[list[float], list[float]]:
        longitudinal, transverse = [], []
        for lane in lanes:
            angle = abs(lane.angle % 180)
            if 30 < angle < 150:
                transverse.append(lane.angle)
            else:
                longitudinal.append(lane.angle)
        return longitudinal, transverse

    def _calibrate_longitudinal(self, lanes: list[LaneInfo]) -> bool:
        points = []
        for lane in lanes:
            x1, y1 = lane.poly[0]
            x2, y2 = lane.poly[1]
            if abs(x2 - x1) < 1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            points.append((slope, intercept))
        if len(points) < 2:
            return False
        left_lines = [p for p in points if p[0] < -0.1]
        right_lines = [p for p in points if p[0] > 0.1]
        if not left_lines or not right_lines:
            left_lines = points[:len(points)//2]
            right_lines = points[len(points)//2:]
        if not left_lines or not right_lines:
            return False
        left_avg = np.mean([p[0] for p in left_lines]), np.mean([p[1] for p in left_lines])
        right_avg = np.mean([p[0] for p in right_lines]), np.mean([p[1] for p in right_lines])
        a1, b1 = left_avg
        a2, b2 = right_avg
        denom = a1 - a2
        if abs(denom) < 1e-6:
            return False
        vp_x = (b2 - b1) / denom
        vp_y = a1 * vp_x + b1
        if vp_y >= self.frame_h or vp_y < 0:
            return False
        self.vp_x = vp_x
        self.vp_y = vp_y
        rows = np.arange(self.frame_h, dtype=np.float32)
        dy_from_vp = np.maximum(rows - vp_y, 1.0)
        ref_dy = self.frame_h - vp_y
        ref_pixel_width = abs(self.frame_w / 2 - vp_x) * 2
        ref_m_per_pixel = self.known_lane_width_m / max(ref_pixel_width, 1.0)
        self.scale_rows = ref_m_per_pixel * (dy_from_vp / ref_dy)
        self.scale_cols = None
        return True

    def _calibrate_transverse(self, lanes: list[LaneInfo]) -> bool:
        x_positions = []
        for lane in lanes:
            cx = int(np.mean([p[0] for p in lane.poly]))
            x_positions.append(cx)
        if len(x_positions) < 2:
            return False
        x_positions.sort()
        gaps = [x_positions[i+1] - x_positions[i] for i in range(len(x_positions)-1)]
        valid_gaps = [g for g in gaps if g > 20]
        if not valid_gaps:
            return False
        avg_gap = np.mean(valid_gaps)
        cols = np.arange(self.frame_w, dtype=np.float32)
        mid = self.frame_w / 2.0
        self.scale_cols = self.known_lane_width_m / max(avg_gap, 1.0) * np.ones_like(cols)
        self.vp_x = mid
        self.vp_y = 0.0
        return True

    def calibrate(self, frame: np.ndarray) -> bool:
        self.frame_h, self.frame_w = frame.shape[:2]
        lanes = self.detect_lanes(frame)
        if not lanes:
            return False
        longitudinal, transverse = self._lane_orientations(lanes)
        if len(transverse) >= 2:
            self.orientation = RoadOrientation.TRANSVERSE
            ok = self._calibrate_transverse(lanes)
        elif len(longitudinal) >= 2:
            self.orientation = RoadOrientation.LONGITUDINAL
            ok = self._calibrate_longitudinal(lanes)
        else:
            return False
        self._lane_count = max(len(longitudinal), len(transverse))
        self._calibrated = ok
        if ok:
            logger.info(
                "RoadDetector calibrated: orientation=%s, vp=(%.0f, %.0f), lanes=%d, frame=%dx%d",
                self.orientation.value, self.vp_x or 0, self.vp_y or 0,
                self._lane_count, self.frame_w, self.frame_h,
            )
        return ok

    def pixel_distance_to_meters(self, y1: float, y2: float, dx_px: float) -> float:
        if not self._calibrated:
            return dx_px * 0.05
        if self.orientation == RoadOrientation.TRANSVERSE and self.scale_cols is not None:
            return dx_px * float(self.scale_cols[0])
        m_per_px = (self.get_m_per_pixel(int(y1)) + self.get_m_per_pixel(int(y2))) / 2.0
        return dx_px * m_per_px

    def get_m_per_pixel(self, row: int) -> float:
        if not self._calibrated:
            return 0.05
        if self.orientation == RoadOrientation.TRANSVERSE and self.scale_cols is not None:
            return float(self.scale_cols[0])
        if self.scale_rows is None:
            return 0.05
        row = max(0, min(row, self.frame_h - 1))
        return float(self.scale_rows[row])

    def draw_debug(self, frame: np.ndarray) -> None:
        if not self._calibrated:
            return
        if self.vp_x is not None and self.vp_y is not None:
            cv2.circle(frame, (int(self.vp_x), int(self.vp_y)), 5, (0, 0, 255), -1)
            label = f"VP ({self.orientation.value})" if self.orientation else "VP"
            cv2.putText(frame, label, (int(self.vp_x) + 8, int(self.vp_y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if self.orientation == RoadOrientation.LONGITUDINAL:
            for row in range(0, self.frame_h, self.frame_h // 5):
                mpp = self.get_m_per_pixel(row)
                cv2.putText(frame, f"row{row}: {mpp:.4f}m/px",
                            (10, row), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
        else:
            mpp = self.get_m_per_pixel(0)
            cv2.putText(frame, f"scale: {mpp:.4f}m/px",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
