import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
# 后处理线程池：每帧后处理（ByteTrack/速度/标注/轨迹）是 CPU 且会释放 GIL，并行可显著提升吞吐
POST_PROCESS_WORKERS = 4

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
                logger.info("BatchDetector running on GPU: %s", torch.cuda.get_device_name(0))
            else:
                self.device = "cpu"
                logger.warning("BatchDetector running on CPU (CUDA not available)")
        except Exception:
            self.device = "cpu"
            logger.warning("BatchDetector running on CPU (torch import failed)")
        self.model.to(self.device)
        self.class_names = self.model.names

        self._streams: dict[str, object] = {}
        # 有界丢旧队列：每路仅保留最新帧，避免探测器串行后处理导致帧积压卡顿
        self._pending: dict[str, np.ndarray] = {}
        self._pending_lock = threading.Lock()
        self._pending_event = threading.Event()

        self.trackers: dict[str, sv.ByteTrack] = {}
        self.trails: dict[str, dict[int, list[tuple[float, float]]]] = {}
        self.trail_age: dict[str, dict[int, int]] = {}
        self.frame_counts: dict[str, int] = {}

        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._post_pool: Optional[ThreadPoolExecutor] = None

        self._fps = 0.0
        self._total = 0
        self._last_fps = time.perf_counter()

        self._prev_pos: dict[str, dict[int, dict]] = {}
        self._prev_vel: dict[str, dict[int, float]] = {}
        self._prev_ts: dict[str, int] = {}
        self.PIXEL_TO_METER = 0.05

    def register_stream(self, cam_id, stream):
        self._streams[cam_id] = stream

    def unregister_stream(self, cam_id):
        self._streams.pop(cam_id, None)
        with self._pending_lock:
            self._pending.pop(cam_id, None)
        self.trackers.pop(cam_id, None)
        self.trails.pop(cam_id, None)
        self.trail_age.pop(cam_id, None)
        self.frame_counts.pop(cam_id, None)

    def submit(self, cam_id, frame: np.ndarray):
        with self._pending_lock:
            # 丢旧策略：只保留最新帧
            self._pending[cam_id] = frame
        self._pending_event.set()

    def start(self):
        if self._running:
            return
        self._running = True
        self._post_pool = ThreadPoolExecutor(max_workers=POST_PROCESS_WORKERS,
                                            thread_name_prefix="yolo-post")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("BatchDetector started (batch=%d, post_workers=%d)",
                    BATCH_SIZE, POST_PROCESS_WORKERS)

    def stop(self):
        self._running = False
        self._pending_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        if self._post_pool is not None:
            self._post_pool.shutdown(wait=False)
            self._post_pool = None
        logger.info("BatchDetector stopped (total=%d, fps=%.1f)", self._total, self._fps)

    def _loop(self):
        while self._running:
            batch_cams, batch_frames = [], []
            with self._pending_lock:
                # 取全部待检帧（每路最新一帧），清空待检
                for cid, frame in self._pending.items():
                    batch_cams.append(cid)
                    batch_frames.append(frame)
                self._pending.clear()
            self._pending_event.clear()

            if not batch_cams:
                # 无帧：短等待，避免空转
                self._pending_event.wait(timeout=0.005)
                self._pending_event.clear()
                continue

            # 大批量推理：一次 model() 处理最多 BATCH_SIZE 帧，减少调用开销
            i = 0
            while i < len(batch_cams):
                end = min(i + BATCH_SIZE, len(batch_cams))
                self._process_batch(batch_cams[i:end], batch_frames[i:end])
                i = end

        # 清理残留待检
        with self._pending_lock:
            self._pending.clear()

    def _process_batch(self, cams, frames):
        results = self.model(frames, device=self.device, verbose=False)
        pool = self._post_pool
        if pool is None or not self._running:
            # 停止中，串行降级
            for i, (cid, frame) in enumerate(zip(cams, frames)):
                self._process_single(cid, frame, results[i])
            return
        # 后处理并行：ByteTrack/速度/标注/轨迹 是 CPU 且会释放 GIL，多线程并行可显著提升 30 路吞吐
        futs = [pool.submit(self._process_single, cid, frame, results[i])
                for i, (cid, frame) in enumerate(zip(cams, frames))]
        for f in futs:
            try:
                f.result()
            except Exception:
                logger.exception("post process failed")
        self._total += len(frames)
        now = time.perf_counter()
        if now - self._last_fps >= 5.0:
            self._fps = self._total / (now - self._last_fps)
            self._total = 0
            self._last_fps = now
            logger.info("BatchDetector FPS: %.1f (%d cams)", self._fps, len(self._streams))

    def _process_single(self, cam_id, frame, result):
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
        det_list: list[dict] = []
        for i in range(len(detections)):
            track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None and i < len(detections.tracker_id) else -1
            class_id = int(detections.class_id[i]) if detections.class_id is not None else -1
            class_name = self.class_names.get(class_id, f"cls_{class_id}")
            conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
            xyxy = detections.xyxy[i].tolist() if detections.xyxy is not None else [0,0,0,0]
            labels.append(f"#{track_id} {class_name} {conf:.2f}")
            cx, cy = (xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2
            self.trails[cam_id].setdefault(track_id, []).append((cx, cy))
            if len(self.trails[cam_id][track_id]) > 30:
                self.trails[cam_id][track_id].pop(0)
            self.trail_age.setdefault(cam_id, {})[track_id] = self.frame_counts[cam_id]
            det_list.append({
                "track_id": track_id, "class_name": class_name,
                "confidence": conf, "bbox": xyxy,
            })

        CameraDataStore().update(
            camera_id=cam_id, total_vehicle_count=len(det_list),
            boxes=[BoundingBoxItem(int(d["track_id"]), str(d["class_name"]),
                                  float(d["confidence"]), list(d["bbox"])) for d in det_list],
        )

        ts = int(time.time() * 1000)
        prev = self._prev_ts.get(cam_id, ts - 33)
        self._prev_ts[cam_id] = ts
        dt = max((ts - prev) / 1000.0, 0.001)

        self._prev_pos.setdefault(cam_id, {})
        self._prev_vel.setdefault(cam_id, {})
        cur_pos: dict[int, dict] = {}
        enriched: list[dict] = []

        for det in det_list:
            tid = int(det.get("track_id", -1))
            if tid < 0:
                continue
            bb = det.get("bbox", [0,0,0,0])
            cx, cy = (bb[0]+bb[2])/2, (bb[1]+bb[3])/2
            cur_pos[tid] = {"cx": cx, "cy": cy}
            p = self._prev_pos[cam_id].get(tid)
            vel = ((np.hypot(cx-p["cx"], cy-p["cy"]) * self.PIXEL_TO_METER) / dt if p else 0.0)
            pv = self._prev_vel[cam_id].get(tid, vel)
            acc = (vel - pv) / dt
            sec = 1 if p and cx < p["cx"] else 0
            prec, hw = -1, 0.0
            for oid, o in cur_pos.items():
                if oid == tid:
                    continue
                dy = cy - o["cy"]
                if dy > 0 and (hw == 0 or abs(dy) < hw):
                    hw, prec = abs(dy), oid
            enriched.append({
                "track_id": tid, "class_name": det.get("class_name", ""),
                "confidence": det.get("confidence", 0.0), "bbox": bb,
                "velocity": vel, "acceleration": acc, "section_id": sec,
                "preceding_id": prec, "space_headway": hw * self.PIXEL_TO_METER,
            })
            self._prev_vel[cam_id][tid] = vel

        self._prev_pos[cam_id] = cur_pos

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
                risk_predictor.process_frame(cam_id, self.frame_counts[cam_id], ts, enriched)
            except Exception:
                pass

        frame = self.box_annotator.annotate(scene=frame, detections=detections)
        frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        cf = self.frame_counts[cam_id]
        stale = [t for t in list(self.trails[cam_id]) if cf - self.trail_age.get(cam_id, {}).get(t, 0) > TRAIL_MAX_AGE]
        for t in stale:
            del self.trails[cam_id][t]
            self.trail_age.get(cam_id, {}).pop(t, None)
        for t in self.trails[cam_id]:
            tr = self.trails[cam_id][t]
            if len(tr) < 2:
                continue
            for j in range(1, len(tr)):
                cv2.line(frame, (int(tr[j-1][0]), int(tr[j-1][1])),
                         (int(tr[j][0]), int(tr[j][1])), (0, 255, 255), 1)

        s = self._streams.get(cam_id)
        if s is not None:
            s.set_processed_frame(frame)
