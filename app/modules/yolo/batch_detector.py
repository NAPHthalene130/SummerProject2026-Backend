"""批量推理调度器：编排冷启动、推理、回传模块"""

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np
import supervision as sv

from app.modules.camera_data import CameraDataStore
from app.modules.lstm.predictor import risk_predictor
from app.modules.yolo.cold_start import load_model
from app.modules.yolo.inference import FrameProcessor

logger = logging.getLogger(__name__)

BATCH_SIZE = 8


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
        self.model, self.device = load_model(model_path)
        self.class_names = self.model.names
        self._streams: dict[str, object] = {}
        self._pending_frames: dict[str, np.ndarray] = {}
        self._pending_lock = threading.Lock()
        self._processors: dict[str, FrameProcessor] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._fps = 0.0
        self._total_processed = 0
        self._last_fps_time = time.perf_counter()

    def register_stream(self, cam_id: str, stream: object) -> None:
        self._streams[cam_id] = stream

    def unregister_stream(self, cam_id: str) -> None:
        self._streams.pop(cam_id, None)
        with self._pending_lock:
            self._pending_frames.pop(cam_id, None)
        self._processors.pop(cam_id, None)

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

    def _loop(self) -> None:
        while self._running:
            batch_ids: list[str] = []
            batch_frames: list[np.ndarray] = []
            with self._pending_lock:
                items = list(self._pending_frames.items())
                self._pending_frames.clear()
            for cid, frame in items:
                if cid not in self._streams:
                    continue
                batch_ids.append(cid)
                batch_frames.append(frame)
                if len(batch_frames) >= BATCH_SIZE:
                    self._process_batch(batch_ids, batch_frames)
                    batch_ids, batch_frames = [], []
            if batch_frames:
                self._process_batch(batch_ids, batch_frames)
            time.sleep(0.01)

    def _process_batch(self, cam_ids: list[str], frames: list[np.ndarray]) -> None:
        try:
            results = self.model(frames, imgsz=640, device=self.device, verbose=False)
            for i, (cid, frame) in enumerate(zip(cam_ids, frames)):
                self._process_single(cid, frame, results[i])
            self._total_processed += len(frames)
            now = time.perf_counter()
            if now - self._last_fps_time >= 5.0:
                self._fps = self._total_processed / (now - self._last_fps_time)
                self._total_processed = 0
                self._last_fps_time = now
                logger.info("BatchDetector FPS: %.1f (%d cameras)", self._fps, len(self._streams))
        except Exception as e:
            logger.error("Batch error: %s", e)

    def _process_single(self, cam_id: str, frame: np.ndarray, result) -> None:
        if cam_id not in self._processors:
            fp = FrameProcessor(cam_id, self.class_names)
            h, w = frame.shape[:2]
            fp.init_zone(h, w)
            self._processors[cam_id] = fp
        fp = self._processors[cam_id]

        dlist, annotated, now, stats = fp.process(self.model, frame, self.device, result)

        stream = self._streams.get(cam_id)
        if stream is not None:
            stream.set_processed_frame(annotated)

    def get_traffic_flow(self, cam_id: str) -> dict:
        fp = self._processors.get(cam_id)
        if fp is None:
            return {"entry_count": 0, "exit_count": 0, "flow_per_min": 0.0}
        return fp._get_flow()

    def get_all_traffic_flow(self) -> dict[str, dict]:
        return {cid: self.get_traffic_flow(cid) for cid in self._processors}
