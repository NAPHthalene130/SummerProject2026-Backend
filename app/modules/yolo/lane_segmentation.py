"""Lane segmentation worker: independent thread for periodic lane count inference.

Every `interval` seconds, serially infers lane count for all active camera streams
using a YOLO segmentation model. Results are cached in CameraDataStore._lane_count_cache,
which inference.py reads with priority over the heuristic _estimate_lanes().

When model_path is empty or the file does not exist, the worker is not started
(checked by main.py lifespan). inference.py falls back to _estimate_lanes_stable().
"""

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from app.modules.camera_data import CameraDataStore
from app.modules.stream.stream_manager import StreamManager

logger = logging.getLogger(__name__)


class LaneSegmentationWorker:
    """Independent thread: every `interval` seconds, serially infers lane count
    for all active camera streams using a YOLO segmentation model.

    Serial inference (not parallel) avoids spiking GPU usage.  At 50-100ms per
    camera, 30 cameras take 1.5-3s within the 5s cycle.
    """

    def __init__(
        self,
        model_path: str,
        interval: float = 5.0,
        image_size: int = 640,
    ) -> None:
        self._model_path = model_path
        self._interval = interval
        self._image_size = image_size
        self._model = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        from ultralytics import YOLO

        path = Path(self._model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Lane segmentation model not found: {self._model_path}")

        self._model = YOLO(str(path))
        try:
            import torch
            if torch.cuda.is_available():
                self._device = 0
                logger.info("LaneSegmentationWorker running on GPU: %s", torch.cuda.get_device_name(0))
            else:
                self._device = "cpu"
                logger.warning("LaneSegmentationWorker running on CPU (CUDA not available)")
        except ImportError:
            self._device = "cpu"
            logger.warning("LaneSegmentationWorker running on CPU (torch import failed)")
        self._model.to(self._device)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="lane-seg", daemon=True)
        self._thread.start()
        logger.info(
            "LaneSegmentationWorker started (model=%s, interval=%ss, imgsz=%d)",
            self._model_path,
            self._interval,
            self._image_size,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                logger.error("LaneSegmentationWorker did not stop within 5 seconds")
            else:
                self._thread = None
        logger.info("LaneSegmentationWorker stopped")

    def _loop(self) -> None:
        while not self._stop_event.wait(self._interval):
            if self._stop_event.is_set():
                break
            try:
                streams = StreamManager().get_all_streams()
                if not streams:
                    continue
                for cam_id, stream in streams.items():
                    if self._stop_event.is_set():
                        break
                    try:
                        frame, _ = stream.get_raw_frame()
                        if frame is None:
                            continue
                        lane_count = self._infer(frame)
                        CameraDataStore().update_lane_count(cam_id, lane_count)
                    except Exception:
                        logger.debug("Lane segmentation failed for camera %s", cam_id, exc_info=True)
            except Exception:
                logger.exception("LaneSegmentationWorker loop error")

    def _infer(self, frame: np.ndarray) -> int:
        """Run YOLO segmentation inference and extract lane count from masks."""
        if self._model is None:
            return 1
        results = self._model(
            frame,
            device=self._device,
            imgsz=self._image_size,
            verbose=False,
        )
        return self._count_lanes(results)

    def _count_lanes(self, results) -> int:
        """Extract lane count from segmentation results.

        For instance segmentation models (ultralytics YOLO-seg), each mask
        instance is treated as one lane line.  Models trained for lane
        detection typically output one mask per lane.
        """
        if not results:
            return 1
        result = results[0]
        masks = getattr(result, "masks", None)
        if masks is None or masks.data is None:
            return 1
        mask_data = masks.data
        if hasattr(mask_data, "cpu"):
            mask_data = mask_data.cpu().numpy()
        count = len(mask_data)
        return max(1, min(count, 8))
