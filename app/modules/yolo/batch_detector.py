"""批量推理调度器：编排冷启动、推理、回传模块（含并发后处理）"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np

from app.modules.camera_data import CameraDataStore, BoundingBoxItem
from app.modules.lstm.predictor import risk_predictor
from app.modules.yolo.cold_start import load_model
from app.modules.yolo.inference import FrameProcessor

logger = logging.getLogger(__name__)

from app.config import settings

BATCH_SIZE = settings.YOLO_BATCH_SIZE
POST_WORKERS = settings.YOLO_POST_WORKERS
YOLO_IMSZ = settings.YOLO_IMAGE_SIZE
USE_HALF = settings.YOLO_HALF
BATCH_COLLECT_SEC = settings.YOLO_BATCH_COLLECT_MS / 1000.0


class BatchDetector:
    _instance: Optional["BatchDetector"] = None
    _instance_lock = threading.Lock()

    def __new__(cls, model_path: str = "app/modules/yolo/best.pt") -> "BatchDetector":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init(model_path)
        return cls._instance

    def _init(self, model_path: str) -> None:
        cv2.setNumThreads(settings.OPENCV_THREADS)
        self.model, self.device = load_model(model_path)
        self.class_names = self.model.names
        self._use_half = USE_HALF and self.device != "cpu"

        self._streams: dict[str, object] = {}
        self._streams_lock = threading.RLock()
        self._pending: dict[str, np.ndarray] = {}
        self._post_inflight: set[str] = set()
        self._pending_lock = threading.Lock()
        self._pending_event = threading.Event()
        self._processors: dict[str, FrameProcessor] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._post_pool: Optional[ThreadPoolExecutor] = None
        self._fps = 0.0
        self._window_total = 0
        self._lifetime_total = 0
        self._last_fps = time.perf_counter()
        self._metrics_lock = threading.Lock()

    def register_stream(self, cam_id: str, stream: object) -> None:
        with self._streams_lock:
            self._streams[cam_id] = stream

    def unregister_stream(self, cam_id: str) -> None:
        with self._streams_lock:
            self._streams.pop(cam_id, None)
        with self._pending_lock:
            self._pending.pop(cam_id, None)
            inflight = cam_id in self._post_inflight
        if not inflight:
            self._processors.pop(cam_id, None)

    def submit(self, cam_id: str, frame: np.ndarray) -> None:
        with self._pending_lock:
            self._pending[cam_id] = frame.copy()
        self._pending_event.set()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._post_pool = ThreadPoolExecutor(max_workers=POST_WORKERS, thread_name_prefix="yolo-post")
        self._thread = threading.Thread(target=self._loop, name="yolo-infer", daemon=True)
        self._thread.start()
        logger.info("BatchDetector started (batch=%d, post=%d, half=%s)", BATCH_SIZE, POST_WORKERS, self._use_half)

    def stop(self) -> None:
        self._running = False
        self._pending_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        if self._post_pool is not None:
            self._post_pool.shutdown(wait=True, cancel_futures=True)
            self._post_pool = None
        with self._pending_lock:
            self._pending.clear()
            self._post_inflight.clear()
        logger.info("BatchDetector stopped (total=%d, fps=%.1f)", self._lifetime_total, self._fps)

    def _loop(self) -> None:
        kw = {"half": True} if self._use_half else {}
        while self._running:
            self._pending_event.wait(timeout=0.02)
            if not self._running:
                break
            if BATCH_COLLECT_SEC > 0 and len(self._take_peek()) < BATCH_SIZE:
                self._pending_event.wait(timeout=BATCH_COLLECT_SEC)
            items = self._take_batch()
            if not items:
                continue
            frames = [f for _, f in items]
            try:
                results = self.model(frames, imgsz=YOLO_IMSZ, device=self.device, verbose=False, **kw)
            except Exception as e:
                logger.error("Batch infer error: %s", e)
                for cid, _ in items:
                    self._release_inflight(cid)
                continue
            for (cid, frame), result in zip(items, results):
                try:
                    if self._post_pool:
                        self._post_pool.submit(self._postprocess_job, cid, frame, result)
                    else:
                        self._postprocess_job(cid, frame, result)
                except RuntimeError:
                    self._release_inflight(cid)

    def _take_peek(self) -> list[str]:
        with self._pending_lock:
            return [c for c in self._pending if c not in self._post_inflight]

    def _take_batch(self) -> list[tuple[str, np.ndarray]]:
        selected: list[tuple[str, np.ndarray]] = []
        with self._pending_lock:
            for cid in list(self._pending):
                if cid in self._post_inflight:
                    continue
                if cid not in self._streams:
                    self._pending.pop(cid, None)
                    continue
                frame = self._pending.pop(cid)
                self._post_inflight.add(cid)
                selected.append((cid, frame))
                if len(selected) >= BATCH_SIZE:
                    break
            has_left = any(c not in self._post_inflight for c in self._pending)
            if has_left:
                self._pending_event.set()
            else:
                self._pending_event.clear()
        return selected

    def _postprocess_job(self, cam_id: str, frame: np.ndarray, result) -> None:
        try:
            self._process_single(cam_id, frame, result)
        except Exception as e:
            logger.error("Post-process error [%s]: %s", cam_id, e)
        finally:
            self._release_inflight(cam_id)
            with self._metrics_lock:
                self._window_total += 1
                self._lifetime_total += 1
                now = time.perf_counter()
                if now - self._last_fps >= 5.0:
                    self._fps = self._window_total / (now - self._last_fps)
                    self._window_total = 0
                    self._last_fps = now

    def _release_inflight(self, cam_id: str) -> None:
        with self._pending_lock:
            self._post_inflight.discard(cam_id)
            has_pending = cam_id in self._pending
        if cam_id not in self._streams:
            self._processors.pop(cam_id, None)
        if has_pending:
            self._pending_event.set()

    def _process_single(self, cam_id: str, frame: np.ndarray, result) -> None:
        if cam_id not in self._processors:
            fp = FrameProcessor(cam_id, self.class_names)
            h, w = frame.shape[:2]
            fp.init_zone(h, w)
            self._processors[cam_id] = fp
        fp = self._processors[cam_id]

        dlist, annotated, stats = fp.process(self.model, frame, self.device, result)

        with self._streams_lock:
            stream = self._streams.get(cam_id)
        if stream is not None:
            stream.set_processed_frame(annotated)

        if dlist:
            ts = int(time.time() * 1000)
            enriched = []
            for d in dlist:
                tid = d["track_id"]
                b = d["bbox"]
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                prev = fp._prev_positions.get(tid, {})
                vel = prev.get("_vel", 0.0)
                hp = 0.0
                if len(dlist) > 1:
                    ob = dlist[1]["bbox"]
                    hp = abs(cy - (ob[1] + ob[3]) / 2) * 0.05
                enriched.append({
                    "track_id": tid, "frame_id": fp.frame_count, "timestamp_ms": ts,
                    "section_id": 1 if prev.get("cx", cx) > cx else 0,
                    "velocity": prev.get("_vel_mps", vel / 3.6),
                    "acceleration": 0.0, "preceding_id": -1, "space_headway": hp,
                })
            try:
                risk_predictor.process_frame(cam_id, fp.frame_count, ts, enriched)
            except Exception:
                pass

        # 更新平均车速到CameraDataStore
        speeds = [d["velocity"] for d in enriched]
        if speeds:
            avg_kmh = float(np.mean(speeds)) * 3.6
            CameraDataStore().update_traffic_metrics(
                cam_id, avg_speed_kmh=avg_kmh, vehicle_count=len(dlist),
            )

    def get_traffic_flow(self, cam_id: str) -> dict:
        fp = self._processors.get(cam_id)
        if fp is None:
            return {"entry_count": 0, "exit_count": 0, "flow_per_min": 0.0}
        return fp.get_traffic_flow()

    def get_all_traffic_flow(self) -> dict[str, dict]:
        return {cid: self.get_traffic_flow(cid) for cid in self._processors}
