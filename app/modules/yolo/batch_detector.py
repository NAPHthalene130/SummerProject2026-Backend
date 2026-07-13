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

LOST_BUFFER = 30
TRAIL_MAX_AGE = 30
BATCH_SIZE = settings.YOLO_BATCH_SIZE
# The deployment host has eight CPU threads available.  Inference remains on one
# GPU thread while tracking/drawing is pipelined through this pool.
POST_PROCESS_WORKERS = settings.YOLO_POST_WORKERS
BATCH_COLLECT_SECONDS = settings.YOLO_BATCH_COLLECT_MS / 1000.0
YOLO_IMAGE_SIZE = settings.YOLO_IMAGE_SIZE
RENDER_ANNOTATED_FRAMES = settings.YOLO_RENDER_ANNOTATED_FRAMES

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "best.pt"


@dataclass(slots=True)
class PendingFrame:
    frame: np.ndarray
    frame_id: int
    captured_at: float
    priority: bool = False


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
        self._viewer_fps = 0.0
        self._background_fps = 0.0
        self._window_total = 0
        self._window_viewer = 0
        self._window_background = 0
        self._lifetime_total = 0
        self._last_fps = time.perf_counter()
        self._metrics_lock = threading.Lock()

        self._prev_pos: dict[str, dict[int, dict]] = {}
        self._prev_vel: dict[str, dict[int, float]] = {}
        self._prev_ts: dict[str, int] = {}
        self.PIXEL_TO_METER = 0.05

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
        priority: bool = False,
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
                    priority=priority,
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
            self._viewer_fps = 0.0
            self._background_fps = 0.0
            self._window_total = 0
            self._window_viewer = 0
            self._window_background = 0
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
            eligible = [
                (cam_id, pending)
                for cam_id, pending in self._pending.items()
                if cam_id not in self._post_inflight
            ]
            viewer_items = [item for item in eligible if item[1].priority]
            background_items = [item for item in eligible if not item[1].priority]

            # Reserve one slot for background analysis when at least one such
            # camera is ready.  With the six-tile monitor wall, the remaining
            # slots still cover every visible camera in each full batch.
            viewer_limit = BATCH_SIZE - 1 if background_items else BATCH_SIZE
            candidates = viewer_items[:viewer_limit]
            candidates.extend(background_items[: BATCH_SIZE - len(candidates)])
            if len(candidates) < BATCH_SIZE:
                candidates.extend(
                    viewer_items[viewer_limit : viewer_limit + BATCH_SIZE - len(candidates)]
                )

            for cam_id, _ in candidates:
                pending = self._pending.pop(cam_id)
                self._post_inflight.add(cam_id)
                selected.append((cam_id, pending))

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
                self._record_processed_frame(priority=pending.priority)
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
        self.trackers.pop(cam_id, None)
        self.trails.pop(cam_id, None)
        self.trail_age.pop(cam_id, None)
        self.frame_counts.pop(cam_id, None)
        self._prev_pos.pop(cam_id, None)
        self._prev_vel.pop(cam_id, None)
        self._prev_ts.pop(cam_id, None)

    def _record_processed_frame(self, *, priority: bool) -> None:
        now = time.perf_counter()
        with self._metrics_lock:
            self._window_total += 1
            if priority:
                self._window_viewer += 1
            else:
                self._window_background += 1
            self._lifetime_total += 1
            elapsed = now - self._last_fps
            if elapsed >= 5.0:
                self._fps = self._window_total / elapsed
                self._viewer_fps = self._window_viewer / elapsed
                self._background_fps = self._window_background / elapsed
                self._window_total = 0
                self._window_viewer = 0
                self._window_background = 0
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
            viewer_fps = self._viewer_fps
            background_fps = self._background_fps
            lifetime_total = self._lifetime_total
        with self._pending_lock:
            pending = len(self._pending)
            priority_pending = sum(item.priority for item in self._pending.values())
            post_inflight = len(self._post_inflight)
        with self._streams_lock:
            stream_count = len(self._streams)
        return {
            "device": str(self.device),
            "half_precision": self._use_half,
            "batch_size": BATCH_SIZE,
            "post_process_workers": POST_PROCESS_WORKERS,
            "processed_fps": round(fps, 2),
            "viewer_processed_fps": round(viewer_fps, 2),
            "background_processed_fps": round(background_fps, 2),
            "processed_frames": lifetime_total,
            "pending_cameras": pending,
            "priority_pending_cameras": priority_pending,
            "post_process_inflight": post_inflight,
            "registered_streams": stream_count,
        }

    def _process_single(self, cam_id: str, pending: PendingFrame, result) -> None:
        frame = pending.frame.copy() if RENDER_ANNOTATED_FRAMES else None
        if cam_id not in self.trackers:
            self.trackers[cam_id] = sv.ByteTrack(lost_track_buffer=LOST_BUFFER)
            if RENDER_ANNOTATED_FRAMES:
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
            if RENDER_ANNOTATED_FRAMES:
                labels.append(f"#{track_id} {class_name} {conf:.2f}")
            cx, cy = (xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2
            if RENDER_ANNOTATED_FRAMES:
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
            frame_width=int(pending.frame.shape[1]),
            frame_height=int(pending.frame.shape[0]),
            captured_at=pending.captured_at,
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

        if RENDER_ANNOTATED_FRAMES and frame is not None:
            frame = self.box_annotator.annotate(scene=frame, detections=detections)
            frame = self.label_annotator.annotate(
                scene=frame,
                detections=detections,
                labels=labels,
            )

            cf = self.frame_counts[cam_id]
            stale = [
                track_id
                for track_id in list(self.trails[cam_id])
                if cf - self.trail_age.get(cam_id, {}).get(track_id, 0) > TRAIL_MAX_AGE
            ]
            for track_id in stale:
                del self.trails[cam_id][track_id]
                self.trail_age.get(cam_id, {}).pop(track_id, None)
            for track_id in self.trails[cam_id]:
                trail = self.trails[cam_id][track_id]
                if len(trail) < 2:
                    continue
                for index in range(1, len(trail)):
                    cv2.line(
                        frame,
                        (int(trail[index - 1][0]), int(trail[index - 1][1])),
                        (int(trail[index][0]), int(trail[index][1])),
                        (0, 255, 255),
                        1,
                    )

            with self._streams_lock:
                stream = self._streams.get(cam_id)
            if stream is not None:
                stream.set_processed_frame(
                    frame,
                    frame_id=pending.frame_id,
                    captured_at=pending.captured_at,
                )
