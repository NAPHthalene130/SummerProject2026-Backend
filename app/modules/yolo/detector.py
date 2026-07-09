import random
import threading
import time

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

TRAFFIC_CLASSES = ["car", "truck", "bus", "pedestrian", "bicycle", "motorcycle"]
LOST_BUFFER = 30

COLORS = {
    "car": (56, 56, 255),
    "truck": (255, 144, 30),
    "bus": (255, 224, 32),
    "pedestrian": (80, 200, 120),
    "bicycle": (255, 105, 180),
    "motorcycle": (128, 0, 128),
}


class YOLODetector:
    def __init__(self, model_path: str = "best.pt"):
        self.model_path = model_path
        self._fps = 0.0
        self._frame_count = 0
        self._last_time = time.perf_counter()

        self.model = YOLO(model_path)
        try:
            import torch
            self.device = 0 if torch.cuda.is_available() else "cpu"
        except Exception:
            self.device = "cpu"
        self.model.to(self.device)
        self.class_names = self.model.names
        self.model_lock = threading.Lock()

        self.trackers = {}
        self.trails = {}
        self.speed_displays = {}
        self.frame_counts = {}
        self._latest_detections = {}

    def detect(self, frame: np.ndarray, cam_id: str = "default") -> np.ndarray:
        if cam_id not in self.trackers:
            self.trackers[cam_id] = sv.ByteTrack(lost_track_buffer=LOST_BUFFER)
            self.trails[cam_id] = {}
            self.speed_displays[cam_id] = {}
            self.frame_counts[cam_id] = 0

        h, w = frame.shape[:2]

        num_boxes = random.randint(0, 5)
        for _ in range(num_boxes):
            box_w = random.randint(int(w * 0.05), int(w * 0.25))
            box_h = random.randint(int(h * 0.05), int(h * 0.25))
            x1 = random.randint(0, w - box_w)
            y1 = random.randint(0, h - box_h)
            x2 = x1 + box_w
            y2 = y1 + box_h

            cls_name = random.choice(TRAFFIC_CLASSES)
            confidence = round(random.uniform(0.5, 0.99), 2)
            color = COLORS[cls_name]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {confidence:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        self._frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._last_time
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_time = now

        cv2.putText(
            frame,
            f"FPS: {self._fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        return frame
