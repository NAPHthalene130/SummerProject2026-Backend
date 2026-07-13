import logging
from collections import deque
from enum import Enum
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

CAR_CLASSES = {2, 3, 5, 7}
CAR_KNOWN_WIDTH_M = 1.8


class RoadOrientation(Enum):
    LONGITUDINAL = "longitudinal"
    TRANSVERSE = "transverse"
    CURVED = "curved"


class CalibrationMethod(Enum):
    VANISHING_POINT = "vanishing_point"
    TRAPEZOID = "trapezoid"
    CURVE_FIT = "curve_fit"
    VEHICLE_SIZE = "vehicle_size"


class LaneInfo:
    def __init__(self, poly: np.ndarray, angle: float):
        self.poly = poly
        self.angle = angle


class RoadDetector:
    def __init__(self, model_path: str = "yolo11m-seg.pt", known_lane_width_m: float = 3.5):
        self.known_lane_width_m = known_lane_width_m
        self.model = YOLO(model_path)
        self.vp_x: Optional[float] = None
        self.vp_y: Optional[float] = None
        self.scale_rows: Optional[np.ndarray] = None
        self.scale_cols: Optional[np.ndarray] = None
        self.orientation: Optional[RoadOrientation] = None
        self.method: Optional[CalibrationMethod] = None
        self.frame_h: int = 0
        self.frame_w: int = 0
        self._calibrated = False
        self._lane_count = 0
        self._vehicle_samples: deque = deque(maxlen=120)

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def _get_road_and_vehicles(self, frame: np.ndarray):
        results = self.model(frame, verbose=False)[0]
        road_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        vehicle_boxes = []
        if results.masks is None:
            return road_mask, vehicle_boxes
        for i, cls_id in enumerate(results.boxes.cls.int().tolist() if results.boxes is not None else []):
            if cls_id in CAR_CLASSES:
                xy = results.masks.xy[i]
                poly = np.array([(int(p[0]), int(p[1])) for p in xy], dtype=np.int32)
                cv2.fillPoly(road_mask, [poly], 255)
                x1, y1, x2, y2 = results.boxes.xyxy[i].int().tolist()
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                pw = x2 - x1
                ph = y2 - y1
                vehicle_boxes.append({"cx": cx, "cy": cy, "w": pw, "h": ph, "cls": int(cls_id)})
        road_mask = cv2.dilate(road_mask, np.ones((21, 21), np.uint8), iterations=2)
        road_mask = cv2.erode(road_mask, np.ones((5, 5), np.uint8), iterations=1)
        return road_mask, vehicle_boxes

    def detect_lanes(self, frame: np.ndarray) -> list[LaneInfo]:
        h, w = frame.shape[:2]
        road_mask, _ = self._get_road_and_vehicles(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        equalized = cv2.equalizeHist(blur)
        edges = cv2.Canny(equalized, 30, 100)
        roi = cv2.bitwise_and(edges, road_mask)
        lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 60, minLineLength=50, maxLineGap=60)
        lanes = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx, dy = x2 - x1, y2 - y1
                if np.sqrt(dx * dx + dy * dy) < 40:
                    continue
                angle = np.degrees(np.arctan2(dy, dx))
                lanes.append(LaneInfo(np.array([(x1, y1), (x2, y2)], dtype=np.int32), angle))
        return lanes

    def _detect_road_boundary_curves(self, frame: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        road_mask, _ = self._get_road_and_vehicles(frame)
        contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        road_contour = max(contours, key=cv2.contourArea)
        pts = road_contour.squeeze()
        if pts.ndim != 2 or pts.shape[0] < 5:
            return None
        left_pts, right_pts = [], []
        for pt in pts:
            x, y = int(pt[0]), int(pt[1])
            mid = w // 2
            if x < mid:
                left_pts.append((x, y))
            else:
                right_pts.append((x, y))
        if len(left_pts) < 3 or len(right_pts) < 3:
            return None
        left_pts = np.array(sorted(left_pts, key=lambda p: p[1]))
        right_pts = np.array(sorted(right_pts, key=lambda p: p[1]))
        left_fit = np.polyfit(left_pts[:, 1], left_pts[:, 0], 2)
        right_fit = np.polyfit(right_pts[:, 1], right_pts[:, 0], 2)
        return np.array([left_fit, right_fit])

    def _calibrate_from_vehicles(self, vehicle_boxes: list) -> bool:
        samples = []
        for v in vehicle_boxes:
            m_per_px = CAR_KNOWN_WIDTH_M / max(v["w"], 1.0)
            if m_per_px < 0.001 or m_per_px > 1.0:
                continue
            samples.append({"row": int(v["cy"]), "mpp": m_per_px})
            self._vehicle_samples.append({"row": int(v["cy"]), "mpp": m_per_px})
        if len(samples) < 3 and len(self._vehicle_samples) < 5:
            return False
        pool = list(self._vehicle_samples) + samples
        if len(pool) < 5:
            return False
        rows = np.array([s["row"] for s in pool], dtype=np.float32)
        mpps = np.array([s["mpp"] for s in pool], dtype=np.float32)
        try:
            coeffs = np.polyfit(rows, mpps, 1)
        except np.linalg.LinAlgError:
            return False
        all_rows = np.arange(self.frame_h, dtype=np.float32)
        self.scale_rows = np.clip(np.polyval(coeffs, all_rows), 0.001, 1.0)
        self.orientation = RoadOrientation.LONGITUDINAL
        self.method = CalibrationMethod.VEHICLE_SIZE
        self._calibrated = True
        logger.info("RoadDetector calibrated via vehicle size: %d samples", len(pool))
        return True

    def _calibrate_from_curves(self, curves: np.ndarray) -> bool:
        left_fit, right_fit = curves
        h = self.frame_h
        rows = np.arange(h, dtype=np.float32)
        left_x = np.polyval(left_fit, rows)
        right_x = np.polyval(right_fit, rows)
        valid = (right_x - left_x) > 10
        if not valid.any():
            return False
        bot_row = h - 1
        top_row = int(np.argmax(valid[::-1]))
        top_row = h - 1 - top_row
        top_row = max(0, min(top_row, h - 1))
        bot_width = right_x[bot_row] - left_x[bot_row]
        top_width = right_x[top_row] - left_x[top_row]
        ref_mpp = self.known_lane_width_m / max(bot_width, 1.0)
        decay = max(top_width / max(bot_width, 1.0), 0.1)
        norm_y = (rows - top_row) / max(bot_row - top_row, 1.0)
        self.scale_rows = ref_mpp * (decay + norm_y * (1.0 - decay))
        self.orientation = RoadOrientation.CURVED
        self.method = CalibrationMethod.CURVE_FIT
        self._calibrated = True
        logger.info("RoadDetector calibrated via curve fit: top=%.0f bot=%.0f decay=%.3f",
                     top_width, bot_width, decay)
        return True

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
        if abs(denom) >= 1e-6:
            vp_x = (b2 - b1) / denom
            vp_y = a1 * vp_x + b1
            vp_in_frame = 0 <= vp_y < self.frame_h
        else:
            vp_x, vp_y = None, None
            vp_in_frame = False
        rows = np.arange(self.frame_h, dtype=np.float32)
        if vp_in_frame:
            self.vp_x = vp_x
            self.vp_y = vp_y
            dy_from_vp = np.maximum(rows - vp_y, 1.0)
            ref_dy = self.frame_h - vp_y
            ref_width = abs(self.frame_w / 2 - vp_x) * 2
            ref_mpp = self.known_lane_width_m / max(ref_width, 1.0)
            self.scale_rows = ref_mpp * (dy_from_vp / ref_dy)
            self.method = CalibrationMethod.VANISHING_POINT
        else:
            def line_x(y, slope, intercept):
                return (y - intercept) / slope if abs(slope) > 1e-6 else 0
            top_y = min(max(0, int(min(p[1] for p in points))), self.frame_h - 1)
            bot_y = self.frame_h - 1
            top_width = abs(line_x(top_y, a1, b1) - line_x(top_y, a2, b2))
            bot_width = abs(line_x(bot_y, a1, b1) - line_x(bot_y, a2, b2))
            bot_width = max(bot_width, top_width)
            ref_mpp = self.known_lane_width_m / max(bot_width, 1.0)
            decay = max(top_width / bot_width, 0.1) if bot_width > 0 else 0.5
            norm_y = (rows - top_y) / max(bot_y - top_y, 1.0)
            self.scale_rows = ref_mpp * (decay + norm_y * (1.0 - decay))
            self.vp_x = None
            self.vp_y = None
            self.method = CalibrationMethod.TRAPEZOID
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
        self.method = CalibrationMethod.VANISHING_POINT
        return True

    def calibrate(self, frame: np.ndarray) -> bool:
        self.frame_h, self.frame_w = frame.shape[:2]
        road_mask, vehicle_boxes = self._get_road_and_vehicles(frame)
        lanes = self._detect_lane_lines(frame, road_mask)
        if lanes:
            longitudinal, transverse = self._lane_orientations(lanes)
            if len(transverse) >= 2:
                self.orientation = RoadOrientation.TRANSVERSE
                if self._calibrate_transverse(lanes):
                    self._calibrated = True
                    logger.info("RoadDetector calibrated: transverse, method=%s", self.method.value)
                    return True
            if len(longitudinal) >= 2:
                self.orientation = RoadOrientation.LONGITUDINAL
                if self._calibrate_longitudinal(lanes):
                    self._calibrated = True
                    logger.info("RoadDetector calibrated: longitudinal, %d lanes, method=%s",
                                 max(len(longitudinal), len(transverse)), self.method.value)
                    return True
        curves = self._detect_road_boundary_curves(frame)
        if curves is not None:
            self.orientation = RoadOrientation.CURVED
            if self._calibrate_from_curves(curves):
                return True
        if self._calibrate_from_vehicles(vehicle_boxes):
            return True
        self._calibrated = False
        return False

    def _detect_lane_lines(self, frame: np.ndarray, road_mask: np.ndarray) -> list[LaneInfo]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        equalized = cv2.equalizeHist(blur)
        edges = cv2.Canny(equalized, 30, 100)
        roi = cv2.bitwise_and(edges, road_mask)
        lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 60, minLineLength=50, maxLineGap=60)
        lanes = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx, dy = x2 - x1, y2 - y1
                if np.sqrt(dx * dx + dy * dy) < 40:
                    continue
                angle = np.degrees(np.arctan2(dy, dx))
                lanes.append(LaneInfo(np.array([(x1, y1), (x2, y2)], dtype=np.int32), angle))
        return lanes

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
        method_label = self.method.value if self.method else "none"
        orient_label = self.orientation.value if self.orientation else "?"
        cv2.putText(frame, f"Cal: {method_label} ({orient_label})",
                    (10, self.frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if self.vp_x is not None and self.vp_y is not None:
            cv2.circle(frame, (int(self.vp_x), int(self.vp_y)), 5, (0, 0, 255), -1)
            cv2.putText(frame, "VP", (int(self.vp_x) + 8, int(self.vp_y) - 8),
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
