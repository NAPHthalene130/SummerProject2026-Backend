import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from app.config import settings
from app.modules.camera_data import BoundingBoxItem, CameraDataStore
from app.modules.lstm.predictor import risk_predictor
from app.modules.yolo.inference import FrameProcessor

LOST_BUFFER = 30
TRAIL_MAX_AGE = 30
BATCH_SIZE = settings.YOLO_BATCH_SIZE
# The deployment host has eight CPU threads available.  Inference remains on one
# GPU thread while tracking/drawing is pipelined through this pool.
POST_PROCESS_WORKERS = settings.YOLO_POST_WORKERS
BATCH_COLLECT_SECONDS = settings.YOLO_BATCH_COLLECT_MS / 1000.0
YOLO_IMAGE_SIZE = settings.YOLO_IMAGE_SIZE

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "best.pt"


@dataclass(slots=True)
class PendingFrame:
    frame: np.ndarray
    frame_id: int
    captured_at: float


class BatchDetector:
    _instance: Optional["BatchDetector"] = None
    _instance_lock = threading.Lock()

    def __new__(cls, model_path: str | None = None) -> "BatchDetector":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init(
                        model_path or settings.YOLO_MODEL_PATH or str(DEFAULT_MODEL_PATH)
                    )
        return cls._instance

    def _init(self, model_path: str) -> None:
        # Eight post workers already provide outer parallelism.  Prevent every
        # resize/cvtColor call from starting another full OpenCV worker team.
        cv2.setNumThreads(settings.OPENCV_THREADS)
        self.model_path = model_path
        self.model = YOLO(model_path)
        try:
            import torch
            if torch.cuda.is_available():
                self.device = 0
                torch.backends.cudnn.benchmark = True
                if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                    torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch.backends.cudnn, "allow_tf32"):
                    torch.backends.cudnn.allow_tf32 = True
                logger.info("BatchDetector running on GPU: %s", torch.cuda.get_device_name(0))
            else:
                self.device = "cpu"
                logger.warning("BatchDetector running on CPU (CUDA not available)")
        except Exception:
            self.device = "cpu"
            logger.warning("BatchDetector running on CPU (torch import failed)")
        self.model.to(self.device)
        self.class_names = self.model.names
        self._use_half = (
            self.device != "cpu"
            and settings.YOLO_HALF
        )
        try:
            from ultralytics.cfg import get_cfg

            supports_quantize = hasattr(get_cfg(), "quantize")
        except Exception:
            supports_quantize = False
        self._precision_kwargs = (
            {"quantize": 16}
            if self._use_half and supports_quantize
            else {"half": True}
            if self._use_half
            else {}
        )

        self._streams: dict[str, object] = {}
        self._streams_lock = threading.RLock()
        # 有界丢旧队列：每路仅保留最新帧，避免探测器串行后处理导致帧积压卡顿
        self._pending: dict[str, PendingFrame] = {}
        self._post_inflight: set[str] = set()
        self._pending_lock = threading.Lock()
        self._pending_event = threading.Event()
        self._stop_event = threading.Event()

        # FrameProcessor per-camera state (replaces dev2's trackers/trails/etc.)
        self._processors: dict[str, FrameProcessor] = {}

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._post_pool: Optional[ThreadPoolExecutor] = None

        self._fps = 0.0
        self._window_total = 0
        self._lifetime_total = 0
        self._last_fps = time.perf_counter()
        self._metrics_lock = threading.Lock()

    def register_stream(self, cam_id, stream):
        with self._streams_lock:
            self._streams[cam_id] = stream

    def unregister_stream(self, cam_id):
        with self._streams_lock:
            self._streams.pop(cam_id, None)
        with self._pending_lock:
            self._pending.pop(cam_id, None)
            is_inflight = cam_id in self._post_inflight
        # An in-flight post-process job still owns these per-camera structures.
        # Its finally block performs the deferred cleanup.
        if not is_inflight:
            self._clear_camera_state(cam_id)

    def submit(
        self,
        cam_id: str,
        frame: np.ndarray,
        *,
        frame_id: int = -1,
        captured_at: Optional[float] = None,
        source_stream: object | None = None,
    ) -> None:
        with self._streams_lock:
            if (
                source_stream is not None
                and self._streams.get(cam_id) is not source_stream
            ):
                return
            with self._pending_lock:
                # Per-camera latest-only queue bounds latency and memory for 30 feeds.
                self._pending[cam_id] = PendingFrame(
                    frame=frame,
                    frame_id=frame_id,
                    captured_at=time.perf_counter() if captured_at is None else captured_at,
                )
        self._pending_event.set()

    def start(self):
        if self._running:
            return
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("previous YOLO inference thread is still stopping")
        self._running = True
        self._stop_event.clear()
        with self._metrics_lock:
            self._fps = 0.0
            self._window_total = 0
            self._last_fps = time.perf_counter()
        self._post_pool = ThreadPoolExecutor(max_workers=POST_PROCESS_WORKERS,
                                            thread_name_prefix="yolo-post")
        self._thread = threading.Thread(target=self._loop, name="yolo-inference", daemon=True)
        self._thread.start()
        logger.info(
            "BatchDetector started (batch=%d, collect=%.1fms, post_workers=%d, half=%s)",
            BATCH_SIZE,
            BATCH_COLLECT_SECONDS * 1000,
            POST_PROCESS_WORKERS,
            self._use_half,
        )

    def stop(self):
        self._running = False
        self._stop_event.set()
        self._pending_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                logger.error("YOLO inference thread did not stop within 10 seconds")
            else:
                self._thread = None
        if self._post_pool is not None:
            self._post_pool.shutdown(wait=True, cancel_futures=True)
            self._post_pool = None
        with self._pending_lock:
            self._pending.clear()
            self._post_inflight.clear()
            self._pending_event.clear()
        with self._streams_lock:
            camera_ids = list(self._streams)
            self._streams.clear()
        for camera_id in camera_ids:
            self._clear_camera_state(camera_id)
        logger.info(
            "BatchDetector stopped (total=%d, fps=%.1f)",
            self._lifetime_total,
            self._fps,
        )

    def _loop(self):
        while self._running:
            self._pending_event.wait(timeout=0.05)
            if not self._running:
                break

            # Give concurrently arriving cameras a very small window to form a
            # fuller GPU batch without adding visible end-to-end latency.
            if (
                BATCH_COLLECT_SECONDS
                and self._eligible_pending_count() < BATCH_SIZE
                and self._stop_event.wait(BATCH_COLLECT_SECONDS)
            ):
                break

            items = self._take_batch()
            if not items:
                continue

            self._process_batch(items)

        with self._pending_lock:
            self._pending.clear()

    def _eligible_pending_count(self) -> int:
        with self._pending_lock:
            return sum(
                cam_id not in self._post_inflight for cam_id in self._pending
            )

    def _take_batch(self) -> list[tuple[str, PendingFrame]]:
        selected: list[tuple[str, PendingFrame]] = []
        with self._pending_lock:
            for cam_id in list(self._pending):
                if cam_id in self._post_inflight:
                    continue
                pending = self._pending.pop(cam_id)
                self._post_inflight.add(cam_id)
                selected.append((cam_id, pending))
                if len(selected) >= BATCH_SIZE:
                    break

            has_eligible_pending = any(
                cam_id not in self._post_inflight for cam_id in self._pending
            )
            if has_eligible_pending:
                self._pending_event.set()
            else:
                self._pending_event.clear()
        return selected

    def _process_batch(self, items: list[tuple[str, PendingFrame]]) -> None:
        frames = [pending.frame for _, pending in items]
        try:
            results = list(
                self.model(
                    frames,
                    device=self.device,
                    verbose=False,
                    imgsz=YOLO_IMAGE_SIZE,
                    **self._precision_kwargs,
                )
            )
        except Exception:
            logger.exception("YOLO batch inference failed")
            for cam_id, _ in items:
                self._release_inflight(cam_id)
            return

        if len(results) != len(items):
            logger.error(
                "YOLO returned %d results for a batch of %d frames",
                len(results),
                len(items),
            )
            for cam_id, _ in items[len(results):]:
                self._release_inflight(cam_id)

        completed_items = items[:len(results)]
        if not self._running:
            for cam_id, _ in completed_items:
                self._release_inflight(cam_id)
            return
        pool = self._post_pool
        if pool is None:
            for (cam_id, pending), result in zip(completed_items, results):
                self._postprocess_job(cam_id, pending, result)
            return

        # Deliberately do not wait for these futures.  The inference thread can
        # immediately launch the next GPU batch while eight CPU workers track and
        # draw the previous one.  Per-camera in-flight guards keep ByteTrack serial.
        for (cam_id, pending), result in zip(completed_items, results):
            try:
                pool.submit(self._postprocess_job, cam_id, pending, result)
            except RuntimeError:
                self._release_inflight(cam_id)

    def _postprocess_job(self, cam_id: str, pending: PendingFrame, result) -> None:
        processed = False
        try:
            self._process_single(cam_id, pending, result)
            processed = True
        except Exception:
            logger.exception("YOLO post-process failed for camera %s", cam_id)
        finally:
            if processed:
                self._record_processed_frame()
            self._release_inflight(cam_id)

    def _release_inflight(self, cam_id: str) -> None:
        with self._pending_lock:
            self._post_inflight.discard(cam_id)
            has_pending = cam_id in self._pending
        with self._streams_lock:
            is_registered = cam_id in self._streams
        if not is_registered:
            self._clear_camera_state(cam_id)
        if has_pending:
            self._pending_event.set()

    def _clear_camera_state(self, cam_id: str) -> None:
        self._processors.pop(cam_id, None)

    def _record_processed_frame(self) -> None:
        now = time.perf_counter()
        with self._metrics_lock:
            self._window_total += 1
            self._lifetime_total += 1
            elapsed = now - self._last_fps
            if elapsed >= 5.0:
                self._fps = self._window_total / elapsed
                self._window_total = 0
                self._last_fps = now
                with self._streams_lock:
                    stream_count = len(self._streams)
                logger.info(
                    "BatchDetector FPS: %.1f (%d cams)",
                    self._fps,
                    stream_count,
                )

    def performance_snapshot(self) -> dict:
        with self._metrics_lock:
            fps = self._fps
            lifetime_total = self._lifetime_total
        with self._pending_lock:
            pending = len(self._pending)
            post_inflight = len(self._post_inflight)
        with self._streams_lock:
            stream_count = len(self._streams)
        return {
            "device": str(self.device),
            "half_precision": self._use_half,
            "batch_size": BATCH_SIZE,
            "post_process_workers": POST_PROCESS_WORKERS,
            "processed_fps": round(fps, 2),
            "processed_frames": lifetime_total,
            "pending_cameras": pending,
            "post_process_inflight": post_inflight,
            "registered_streams": stream_count,
        }

    def _process_single(self, cam_id: str, pending: PendingFrame, result) -> None:
        # pending.frame 来自 CameraStream 发布的"一次性"帧：已从 _pending 字典 pop 出来，
        # 仅当前 worker 持有；CameraStream._raw_frame 独立存储，不会被并发读取。
        # supervision annotator 在 frame 上原地绘制，set_processed_frame 写入的就是这个 annotated 帧。
        # 因此可省略全帧 copy，减少 450 次/s 的 640×480×3 内存拷贝。
        frame = pending.frame

        # Get or create FrameProcessor for this camera
        if cam_id not in self._processors:
            fp = FrameProcessor(cam_id, self.class_names)
            h, w = frame.shape[:2]
            fp.init_zone(h, w)
            self._processors[cam_id] = fp
        fp = self._processors[cam_id]

        # Use FrameProcessor for tracking, speed calculation, and supervision unified drawing
        dlist, annotated, stats = fp.process(self.model, frame, self.device, result)

        # Build enriched list for LSTM (field names must stay unchanged)
        ts = int(time.time() * 1000)
        enriched: list[dict] = []
        for d in dlist:
            tid = d["track_id"]
            b = d["bbox"]
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            prev = fp._prev_positions.get(tid, {})
            vel_mps = prev.get("_vel_mps", 0.0)
            # space_headway: find nearest vehicle ahead (same y direction)
            hw = 0.0
            prec = -1
            if len(dlist) > 1:
                for od in dlist:
                    if od["track_id"] == tid:
                        continue
                    ob = od["bbox"]
                    ocy = (ob[1] + ob[3]) / 2
                    dy = cy - ocy
                    if dy > 0 and (hw == 0.0 or abs(dy) < hw):
                        hw = abs(dy)
                        prec = od["track_id"]
            enriched.append({
                "track_id": tid,
                "frame_id": fp.frame_count,
                "timestamp_ms": ts,
                "section_id": 1 if prev.get("cx", cx) > cx else 0,
                "velocity": vel_mps,
                "acceleration": 0.0,
                "preceding_id": prec,
                "space_headway": hw * 0.05,
            })

        # Update traffic metrics for risk prediction service
        if enriched:
            moving_speeds = [float(item["velocity"]) * 3.6 for item in enriched if float(item["velocity"]) > 0]
            avg_speed_kmh = float(np.mean(moving_speeds)) if moving_speeds else 0.0
            CameraDataStore().update_traffic_metrics(
                cam_id,
                avg_speed_kmh=avg_speed_kmh,
                vehicle_count=len(enriched),
            )

        # Feed enriched data to LSTM risk predictor
        if enriched:
            try:
                risk_predictor.process_frame(cam_id, fp.frame_count, ts, enriched)
            except Exception:
                pass

        # Push annotated frame to stream (preserve dev2's frame_id/captured_at signature)
        with self._streams_lock:
            s = self._streams.get(cam_id)
        if s is not None:
            s.set_processed_frame(
                annotated,
                frame_id=pending.frame_id,
                captured_at=pending.captured_at,
            )

    def get_traffic_flow(self, cam_id: str) -> dict:
        fp = self._processors.get(cam_id)
        if fp is None:
            return {"entry_count": 0, "exit_count": 0, "flow_per_min": 0.0}
        return fp.get_traffic_flow()

    def get_all_traffic_flow(self) -> dict[str, dict]:
        return {cid: self.get_traffic_flow(cid) for cid in self._processors}
