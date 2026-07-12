import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RoadDetector:
    def __init__(self, known_lane_width_m: float = 3.5):
        self.known_lane_width_m = known_lane_width_m
        self.vp_x: Optional[float] = None
        self.vp_y: Optional[float] = None
        self.scale_rows: Optional[np.ndarray] = None
        self.frame_h: int = 0
        self.frame_w: int = 0
        self._calibrated = False

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def detect_lanes(self, frame: np.ndarray) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        h, w = frame.shape[:2]
        mask = np.zeros_like(edges)
        poly = np.array([[
            (0, h),
            (w, h),
            (int(w * 0.55), int(h * 0.5)),
            (int(w * 0.45), int(h * 0.5)),
        ]], dtype=np.int32)
        cv2.fillPoly(mask, poly, 255)
        masked = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=50)
        if lines is None:
            return None, None

        left_xs, left_ys, right_xs, right_ys = [], [], [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if slope < -0.3:
                left_xs.extend([x1, x2])
                left_ys.extend([y1, y2])
            elif slope > 0.3:
                right_xs.extend([x1, x2])
                right_ys.extend([y1, y2])

        left_line = None
        if len(left_xs) > 1:
            left_fit = np.polyfit(left_xs, left_ys, 1)
            left_line = left_fit

        right_line = None
        if len(right_xs) > 1:
            right_fit = np.polyfit(right_xs, right_ys, 1)
            right_line = right_fit

        return left_line, right_line

    def calibrate(self, frame: np.ndarray) -> bool:
        self.frame_h, self.frame_w = frame.shape[:2]
        left, right = self.detect_lanes(frame)
        if left is None or right is None:
            return False

        a1, b1 = left
        a2, b2 = right
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

        self._calibrated = True
        logger.info(
            "RoadDetector calibrated: vp=(%.0f, %.0f), frame=%dx%d",
            vp_x, vp_y, self.frame_w, self.frame_h,
        )
        return True

    def pixel_distance_to_meters(self, y1: float, y2: float, dx_px: float) -> float:
        if not self._calibrated:
            return dx_px * 0.05
        m_per_px = (self.get_m_per_pixel(int(y1)) + self.get_m_per_pixel(int(y2))) / 2.0
        return dx_px * m_per_px

    def get_m_per_pixel(self, row: int) -> float:
        if not self._calibrated or self.scale_rows is None:
            return 0.05
        row = max(0, min(row, self.frame_h - 1))
        return float(self.scale_rows[row])

    def draw_debug(self, frame: np.ndarray) -> None:
        if not self._calibrated:
            return
        if self.vp_x is not None and self.vp_y is not None:
            cv2.circle(frame, (int(self.vp_x), int(self.vp_y)), 5, (0, 0, 255), -1)
            cv2.putText(frame, "VP", (int(self.vp_x) + 8, int(self.vp_y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        for row in range(0, self.frame_h, self.frame_h // 5):
            mpp = self.get_m_per_pixel(row)
            cv2.putText(frame, f"row{row}: {mpp:.4f}m/px",
                        (10, row), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
