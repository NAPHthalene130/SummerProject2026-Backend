import collections
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
FLOW_WINDOW_SEC = 60
ZONE_MARGIN = 0.15

logger = logging.getLogger(__name__)


class BatchDetector:
    _instance: Optional["BatchDetector"] = None
    _instance_lock = threading.Lock()

    MIN_CONFIDENCE = 0.1
    NMS_OVERLAP = 0.5

    def __new__(cls, model_path: str = "yolo11m-seg.pt") -> "BatchDetector":
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
        for _ in range(2):
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy, imgsz=640, verbose=False, conf=self.MIN_CONFIDENCE)
        logger.info("YOLO model warm-up complete (device=%s)", self.device)

        self._streams: dict[str, object] = {}
        self._pending_frames: dict[str, np.ndarray] = {}
        self._pending_lock = threading.Lock()

        self.trackers: dict[str, sv.ByteTrack] = {}
        self.trails: dict[str, dict[int, list[tuple[float, float]]]] = {}
        self.trail_age: dict[str, dict[int, int]] = {}
        self.frame_counts: dict[str, int] = {}

        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        self.zone_rects: dict[str, tuple[int, int, int, int]] = {}
        self._inside_zone: dict[str, dict[int, bool]] = {}
        self._entry_events: dict[str, collections.deque] = {}
        self._exit_events: dict[str, collections.deque] = {}

        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._fps = 0.0
        self._total_frames_processed = 0
        self._last_fps_time = time.perf_counter()

        self._prev_positions: dict[str, dict[int, dict]] = {}
        self._prev_velocities: dict[str, dict[int, float]] = {}
        self._prev_timestamps: dict[str, int] = {}
        self._trajectories: dict[str, dict[int, list]] = {}
        self._speed_last_update: dict[str, dict[int, float]] = {}
        self._speed_stable: dict[str, dict[int, float]] = {}
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
        self.zone_rects.pop(cam_id, None)
        self._inside_zone.pop(cam_id, None)
        self._entry_events.pop(cam_id, None)
        self._exit_events.pop(cam_id, None)

    def _prune_events(self, events: collections.deque, now: float) -> None:
        while events and now - events[0][0] > FLOW_WINDOW_SEC:
            events.popleft()

    def get_traffic_flow(self, cam_id: str) -> dict:
        now = time.time()
        entries = self._entry_events.get(cam_id)
        exits = self._exit_events.get(cam_id)
        if entries is None or exits is None:
            return {"entry_count": 0, "exit_count": 0, "flow_per_min": 0.0}
        self._prune_events(entries, now)
        self._prune_events(exits, now)
        total = len(entries) + len(exits)
        return {
            "entry_count": len(entries),
            "exit_count": len(exits),
            "flow_per_min": round(total * 60.0 / FLOW_WINDOW_SEC, 1),
        }

    def get_all_traffic_flow(self) -> dict[str, dict]:
        return {cam_id: self.get_traffic_flow(cam_id) for cam_id in self.zone_rects}

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
        results = self.model(frames, imgsz=640, device=self.device, verbose=False)
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
            self._inside_zone[cam_id] = {}
            self._entry_events[cam_id] = collections.deque()
            self._exit_events[cam_id] = collections.deque()

            h, w = frame.shape[:2]
            margin_x = int(w * ZONE_MARGIN)
            margin_y = int(h * ZONE_MARGIN)
            self.zone_rects[cam_id] = (margin_x, margin_y, w - margin_x, h - margin_y)

        self.frame_counts[cam_id] += 1
        now = time.time()
        now_ms = now * 1000

        detections = sv.Detections.from_ultralytics(result)

        conf_mask = np.array([True] * len(detections))
        if len(detections) > 0 and detections.confidence is not None:
            conf_mask = detections.confidence >= self.MIN_CONFIDENCE
            detections = detections[conf_mask]

        if len(detections) > 0 and detections.xyxy is not None:
            keep = []
            boxes = detections.xyxy
            for i in range(len(boxes)):
                keep_i = True
                for j in range(i):
                    if j not in keep:
                        continue
                    xi1, yi1, xi2, yi2 = boxes[i]
                    xj1, yj1, xj2, yj2 = boxes[j]
                    ix1, iy1 = max(xi1, xj1), max(yi1, yj1)
                    ix2, iy2 = min(xi2, xj2), min(yi2, yj2)
                    if ix1 < ix2 and iy1 < iy2:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        area_i = (xi2 - xi1) * (yi2 - yi1)
                        area_j = (xj2 - xj1) * (yj2 - yj1)
                        iou = inter / min(area_i, area_j)
                        if iou > self.NMS_OVERLAP:
                            keep_i = False
                            break
                if keep_i:
                    keep.append(i)
            detections = detections[keep]

        if len(detections) > 0:
            detections = self.trackers[cam_id].update_with_detections(detections)
        else:
            detections.tracker_id = np.array([], dtype=int)

        detection_list: list[dict[str, object]] = []
        zx1, zy1, zx2, zy2 = self.zone_rects[cam_id]
        inside = self._inside_zone[cam_id]
        speed_map: dict[int, float] = {}
        for i in range(len(detections)):
            if detections.tracker_id is not None and i < len(detections.tracker_id):
                track_id = int(detections.tracker_id[i])
            else:
                track_id = -1

            class_id = int(detections.class_id[i]) if detections.class_id is not None else -1
            class_name = self.class_names.get(class_id, f"cls_{class_id}")
            conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
            xyxy = detections.xyxy[i].tolist() if detections.xyxy is not None else [0, 0, 0, 0]

            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2
            if track_id not in self.trails[cam_id]:
                self.trails[cam_id][track_id] = []
            self.trails[cam_id][track_id].append((cx, cy))
            if len(self.trails[cam_id][track_id]) > 30:
                self.trails[cam_id][track_id].pop(0)

            self.trail_age.setdefault(cam_id, {})[track_id] = self.frame_counts[cam_id]

            was_inside = inside.get(track_id, False)
            is_inside = zx1 < cx < zx2 and zy1 < cy < zy2
            if not was_inside and is_inside:
                self._entry_events[cam_id].append((now, track_id))
            elif was_inside and not is_inside:
                self._exit_events[cam_id].append((now, track_id))
            inside[track_id] = is_inside

            detection_list.append({
                "track_id": track_id,
                "class_name": class_name,
                "confidence": conf,
                "bbox": xyxy,
            })

        lane_positions = []
        for d in detection_list:
            bbox = d.get("bbox", [0, 0, 0, 0])
            cx = (bbox[0] + bbox[2]) / 2
            lane_positions.append(cx)
        lane_count = 0
        if len(lane_positions) >= 3:
            lane_positions.sort()
            gaps = [lane_positions[i+1] - lane_positions[i] for i in range(len(lane_positions)-1)]
            mean_gap = np.mean(gaps) if gaps else 0
            if mean_gap > 5:
                clusters = 1
                for g in gaps:
                    if g > mean_gap * 0.6:
                        clusters += 1
                lane_count = max(1, min(clusters, 8))
        else:
            lane_count = max(1, len(lane_positions))

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
            lane_count=lane_count,
        )

        timestamp_ms = int(time.time() * 1000)
        prev_ms = self._prev_timestamps.get(cam_id, timestamp_ms - 33)
        self._prev_timestamps[cam_id] = timestamp_ms
        dt_sec = max((timestamp_ms - prev_ms) / 1000.0, 0.001)

        if cam_id not in self._prev_positions:
            self._prev_positions[cam_id] = {}
        if cam_id not in self._prev_velocities:
            self._prev_velocities[cam_id] = {}

        if cam_id not in self._trajectories:
            self._trajectories[cam_id] = {}
        if cam_id not in self._speed_last_update:
            self._speed_last_update[cam_id] = {}
        if cam_id not in self._speed_stable:
            self._speed_stable[cam_id] = {}

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

            traj = self._trajectories[cam_id].setdefault(track_id, [])
            traj.append((cx, cy, now))
            cutoff = now - 1.5
            self._trajectories[cam_id][track_id] = [(x, y, t) for x, y, t in traj if t > cutoff]

            vel = 0.0
            last_update = self._speed_last_update[cam_id].get(track_id, 0.0)
            stable = self._speed_stable[cam_id].get(track_id, 0.0)
            if now - last_update >= 0.5:
                pts = self._trajectories[cam_id][track_id]
                if len(pts) >= 5:
                    xs = np.array([p[0] for p in pts])
                    ys = np.array([p[1] for p in pts])
                    ts = np.array([p[2] for p in pts])
                    dt_total = ts[-1] - ts[0]
                    if dt_total > 0.3:
                        total_dist = 0.0
                        for j in range(1, len(pts)):
                            total_dist += np.sqrt((xs[j] - xs[j-1])**2 + (ys[j] - ys[j-1])**2)
                        speed_px = total_dist / dt_total
                        vel = speed_px * self.PIXEL_TO_METER
                        vel_kmh = vel * 3.6
                        if stable > 0 and abs(vel_kmh - stable) < 5:
                            vel_kmh = stable
                        self._speed_stable[cam_id][track_id] = vel_kmh
                        self._speed_last_update[cam_id][track_id] = now
                    else:
                        prev = self._prev_positions[cam_id].get(track_id)
                        if prev:
                            dp = np.sqrt((cx - prev["cx"])**2 + (cy - prev["cy"])**2)
                            vel = (dp * self.PIXEL_TO_METER) / max(dt_sec, 0.01)
                else:
                    prev = self._prev_positions[cam_id].get(track_id)
                    if prev:
                        dp = np.sqrt((cx - prev["cx"])**2 + (cy - prev["cy"])**2)
                        vel = (dp * self.PIXEL_TO_METER) / max(dt_sec, 0.01)

            vel_kmh = self._speed_stable[cam_id].get(track_id, vel * 3.6)
            prev_vel = self._prev_velocities[cam_id].get(track_id, vel)
            acc = (vel - prev_vel) / max(dt_sec, 0.01)

            section_id = 0
            prev_pos = self._prev_positions[cam_id].get(track_id)
            if prev_pos and cx < prev_pos["cx"]:
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
            speed_map[track_id] = vel_kmh

        self._prev_positions[cam_id] = current_positions

        if enriched:
            try:
                risk_predictor.process_frame(cam_id, self.frame_counts[cam_id], timestamp_ms, enriched)
            except Exception:
                pass

        TRACK_COLORS = [
            (56, 56, 255), (255, 144, 30), (255, 224, 32),
            (80, 200, 120), (255, 105, 180), (128, 0, 128),
            (0, 200, 255), (200, 0, 200), (0, 255, 128), (255, 0, 128),
        ]
        for i in range(len(detections)):
            if detections.tracker_id is None or i >= len(detections.tracker_id):
                continue
            track_id = int(detections.tracker_id[i])
            xyxy = detections.xyxy[i].tolist() if detections.xyxy is not None else [0, 0, 0, 0]
            x1, y1, x2, y2 = map(int, xyxy)
            class_id = int(detections.class_id[i]) if detections.class_id is not None else -1
            cls_name = self.class_names.get(class_id, "?")
            spd = speed_map.get(track_id, 0.0)
            color = TRACK_COLORS[track_id % len(TRACK_COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {spd:.0f}km/h"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        if cam_id in self.zone_rects:
            self._prune_events(self._entry_events[cam_id], now)
            self._prune_events(self._exit_events[cam_id], now)
            zx1, zy1, zx2, zy2 = self.zone_rects[cam_id]
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 255, 255), 2)
            cv2.putText(frame, "COUNT ZONE", (zx1 + 5, zy1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            flow = self.get_traffic_flow(cam_id)
            info = [
                f"Entry: {flow['entry_count']}",
                f"Exit: {flow['exit_count']}",
                f"Flow: {flow['flow_per_min']}/min",
            ]
            for j, line in enumerate(info):
                cv2.putText(frame, line, (zx1 + 5, zy2 - 10 - j * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

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
            inside.pop(track_id, None)

        stream = self._streams.get(cam_id)
        if stream is not None:
            stream.set_processed_frame(frame)
