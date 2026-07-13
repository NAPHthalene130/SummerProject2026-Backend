import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from app.modules.camera_data import BoundingBoxItem, CameraDataStore
from app.modules.lstm.predictor import risk_predictor

LOST_BUFFER = 30
TRAIL_MAX_AGE = 30
BATCH_SIZE = 8

logger = logging.getLogger(__name__)


class BatchDetector:
    _instance: Optional["BatchDetector"] = None
    _instance_lock = threading.Lock()

    def __new__(cls, model_path: str = "best.pt") -> "BatchDetector":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init(model_path)
        return cls._instance

    def _init(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = YOLO(model_path)

        try:
            import torch
            if torch.cuda.is_available():
                self.device = 0
                gpu_name = torch.cuda.get_device_name(0)
                logger.info("BatchDetector running on GPU: %s", gpu_name)
            else:
                self.device = "cpu"
                logger.warning("BatchDetector running on CPU (CUDA not available)")
        except Exception:
            self.device = "cpu"
            logger.warning("BatchDetector running on CPU (torch import failed)")
        self.model.to(self.device)
        self.class_names = self.model.names

        self._streams: dict[str, object] = {}
        self._pending_frames: dict[str, np.ndarray] = {}
        self._pending_lock = threading.Lock()

        self.trackers: dict[str, sv.ByteTrack] = {}
        self.trails: dict[str, dict[int, list[tuple[float, float]]]] = {}
        self.trail_age: dict[str, dict[int, int]] = {}
        self.frame_counts: dict[str, int] = {}

        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._fps = 0.0
        self._total_frames_processed = 0
        self._last_fps_time = time.perf_counter()

        self._prev_positions: dict[str, dict[int, dict]] = {}
        self._prev_velocities: dict[str, dict[int, float]] = {}
        self._prev_timestamps: dict[str, int] = {}
        self.PIXEL_TO_METER = 0.05

    def register_stream(self, cam_id: str, stream: object) -> None:
        self._streams[cam_id] = stream

    def unregister_stream(self, cam_id: str) -> None:
        self._streams.pop(cam_id, None)
        with self._pending_lock:
            self._pending_frames.pop(cam_id, None)
        self.trackers.pop(cam_id, None)
        self.trails.pop(cam_id, None)
        self.trail_age.pop(cam_id, None)
        self.frame_counts.pop(cam_id, None)

    def submit(self, cam_id: str, frame: np.ndarray) -> None:
        with self._pending_lock:
            self._pending_frames[cam_id] = frame.copy()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("BatchDetector started (batch_size=%d)", BATCH_SIZE)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("BatchDetector stopped (total frames: %d, avg FPS: %.1f)",
                     self._total_frames_processed, self._fps)

    def _loop(self) -> None:
        while self._running:
            batch_cam_ids: list[str] = []
            batch_frames: list[np.ndarray] = []

            with self._pending_lock:
                items = list(self._pending_frames.items())
                self._pending_frames.clear()

            for cam_id, frame in items:
                if cam_id not in self._streams:
                    continue
                batch_cam_ids.append(cam_id)
                batch_frames.append(frame)
                if len(batch_frames) >= BATCH_SIZE:
                    self._process_batch(batch_cam_ids, batch_frames)
                    batch_cam_ids = []
                    batch_frames = []

            if batch_frames:
                self._process_batch(batch_cam_ids, batch_frames)

            time.sleep(0.01)

    def _process_batch(self, cam_ids: list[str], frames: list[np.ndarray]) -> None:
        results = self.model(frames, device=self.device, verbose=False)
        for i, (cam_id, frame) in enumerate(zip(cam_ids, frames)):
            self._process_single(cam_id, frame, results[i])

        self._total_frames_processed += len(frames)
        now = time.perf_counter()
        elapsed = now - self._last_fps_time
        if elapsed >= 5.0:
            self._fps = self._total_frames_processed / elapsed
            self._total_frames_processed = 0
            self._last_fps_time = now
            logger.info("BatchDetector FPS: %.1f (%d cameras active)",
                         self._fps, len(self._streams))

    def _process_single(self, cam_id: str, frame: np.ndarray, result) -> None:
        if cam_id not in self.trackers:
            self.trackers[cam_id] = sv.ByteTrack(lost_track_buffer=LOST_BUFFER)
            self.trails[cam_id] = {}
            self.frame_counts[cam_id] = 0

        self.frame_counts[cam_id] += 1

        detections = sv.Detections.from_ultralytics(result)

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

        timestamp_ms = int(time.time() * 1000)
        prev_ms = self._prev_timestamps.get(cam_id, timestamp_ms - 33)
        self._prev_timestamps[cam_id] = timestamp_ms
        dt_sec = max((timestamp_ms - prev_ms) / 1000.0, 0.001)

        if cam_id not in self._prev_positions:
            self._prev_positions[cam_id] = {}
        if cam_id not in self._prev_velocities:
            self._prev_velocities[cam_id] = {}

        current_positions: dict[int, dict] = {}
        enriched: list[dict] = []

        for det in detection_list:
            track_id = int(det.get("track_id", -1))
            if track_id < 0:
                continue
            bbox = det.get("bbox", [0, 0, 0, 0])
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            current_positions[track_id] = {"cx": cx, "cy": cy}

            prev = self._prev_positions[cam_id].get(track_id)
            if prev:
                dp = np.sqrt((cx - prev["cx"])**2 + (cy - prev["cy"])**2)
                vel = (dp * self.PIXEL_TO_METER) / dt_sec
            else:
                vel = 0.0
            prev_vel = self._prev_velocities[cam_id].get(track_id, vel)
            acc = (vel - prev_vel) / dt_sec

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

            enriched.append({
                "track_id": track_id,
                "class_name": det.get("class_name", ""),
                "confidence": det.get("confidence", 0.0),
                "bbox": bbox,
                "velocity": vel,
                "acceleration": acc,
                "section_id": section_id,
                "preceding_id": preceding_id,
                "space_headway": headway * self.PIXEL_TO_METER,
            })

            self._prev_velocities[cam_id][track_id] = vel

        self._prev_positions[cam_id] = current_positions

        if enriched:
            moving_speeds = [float(item["velocity"]) * 3.6 for item in enriched if float(item["velocity"]) > 0]
            avg_speed_kmh = float(np.mean(moving_speeds)) if moving_speeds else 0.0
            CameraDataStore().update_traffic_metrics(
                cam_id,
                avg_speed_kmh=avg_speed_kmh,
                vehicle_count=len(enriched),
            )

        if enriched:
            try:
                risk_predictor.process_frame(cam_id, self.frame_counts[cam_id], timestamp_ms, enriched)
            except Exception:
                pass

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

        stream = self._streams.get(cam_id)
        if stream is not None:
            stream.set_processed_frame(frame)
