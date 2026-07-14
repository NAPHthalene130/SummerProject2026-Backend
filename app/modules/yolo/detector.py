import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from app.config import settings
from app.modules.camera_data import BoundingBoxItem, CameraDataStore

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "best.pt"

LOST_BUFFER = 30
TRAIL_MAX_AGE = 30

logger = logging.getLogger(__name__)


class YOLODetector:
    def __init__(self, model_path: str | None = None):
        model_path = model_path or settings.YOLO_MODEL_PATH or str(DEFAULT_MODEL_PATH)
        self.model_path = model_path
        self._fps = 0.0
        self._frame_count = 0
        self._last_time = time.perf_counter()

        self.model = YOLO(model_path)
        try:
            import torch
            if torch.cuda.is_available():
                self.device = 0
                gpu_name = torch.cuda.get_device_name(0)
                logger.info("YOLO running on GPU: %s", gpu_name)
            else:
                self.device = "cpu"
                logger.warning("YOLO running on CPU (CUDA not available)")
        except Exception:
            self.device = "cpu"
            logger.warning("YOLO running on CPU (torch import failed)")
        self.model.to(self.device)

        if self.device != "cpu":
            logger.info("YOLO detector using GPU (CUDA device %s)", self.device)
        else:
            logger.warning("YOLO detector using CPU -- install CUDA-enabled PyTorch for GPU acceleration")
        self.class_names = self.model.names
        self.model_lock = threading.Lock()

        self.trackers: dict[str, sv.ByteTrack] = {}
        self.trails: dict[str, dict[int, list[tuple[float, float]]]] = {}
        self.trail_age: dict[str, dict[int, int]] = {}
        self.frame_counts: dict[str, int] = {}
        self._latest_detections: dict[str, list[dict[str, object]]] = {}

        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()

    def detect(self, frame: np.ndarray, cam_id: str = "default") -> np.ndarray:
        if cam_id not in self.trackers:
            self.trackers[cam_id] = sv.ByteTrack(lost_track_buffer=LOST_BUFFER)
            self.trails[cam_id] = {}
            self.frame_counts[cam_id] = 0

        self.frame_counts[cam_id] += 1

        with self.model_lock:
            results = self.model(frame, device=self.device, verbose=False)

        detections = sv.Detections.from_ultralytics(results[0])

        if len(detections) > 0:
            detections = self.trackers[cam_id].update_with_detections(detections)
        else:
            detections.tracker_id = np.array([], dtype=int)

        labels: list[str] = []
        detection_list: list[dict[str, object]] = []
        for i in range(len(detections)):
            if detections.tracker_id is not None and i < len(detections.tracker_id):
                track_id = int(detections.tracker_id[i])
            else:
                track_id = -1

            class_id = int(detections.class_id[i]) if detections.class_id is not None else -1
            class_name = self.class_names.get(class_id, f"cls_{class_id}")
            conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
            xyxy = detections.xyxy[i].tolist() if detections.xyxy is not None else [0, 0, 0, 0]

            labels.append(f"#{track_id} {class_name} {conf:.2f}")

            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2
            if track_id not in self.trails[cam_id]:
                self.trails[cam_id][track_id] = []
            self.trails[cam_id][track_id].append((cx, cy))
            if len(self.trails[cam_id][track_id]) > 30:
                self.trails[cam_id][track_id].pop(0)

            self.trail_age.setdefault(cam_id, {})[track_id] = self.frame_counts[cam_id]

            detection_list.append({
                "track_id": track_id,
                "class_name": class_name,
                "confidence": conf,
                "bbox": xyxy,
            })
        self._latest_detections[cam_id] = detection_list

        CameraDataStore().update(
            camera_id=cam_id,
            total_vehicle_count=len(detection_list),
            boxes=[
                BoundingBoxItem(
                    track_id=int(d["track_id"]),
                    class_name=str(d["class_name"]),
                    confidence=float(d["confidence"]),
                    bbox=list(d["bbox"]),
                )
                for d in detection_list
            ],
        )

        frame = self.box_annotator.annotate(scene=frame, detections=detections)
        frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        current_frame = self.frame_counts[cam_id]
        stale_ids: list[int] = []
        for track_id in list(self.trails[cam_id].keys()):
            age = self.trail_age.get(cam_id, {}).get(track_id, 0)
            if current_frame - age > TRAIL_MAX_AGE:
                stale_ids.append(track_id)
                continue
            trail = self.trails[cam_id][track_id]
            if len(trail) < 2:
                continue
            for j in range(1, len(trail)):
                pt1 = (int(trail[j - 1][0]), int(trail[j - 1][1]))
                pt2 = (int(trail[j][0]), int(trail[j][1]))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 1)
        for track_id in stale_ids:
            del self.trails[cam_id][track_id]
            self.trail_age.get(cam_id, {}).pop(track_id, None)

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
