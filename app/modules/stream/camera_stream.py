import threading
import time
from typing import Optional

import cv2
import numpy as np

from app.modules.yolo.batch_detector import BatchDetector
from app.utils.camera_manager import CameraConfig

TEST_FRAME_W = 640
TEST_FRAME_H = 480
TARGET_FPS = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS

OBJ_CENTER_X = 0


class CameraStream:
    def __init__(self, config: CameraConfig, batch_detector: BatchDetector):
        self.config = config
        self.batch_detector = batch_detector
        self._frame_id: int = 0
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._use_test_frame = False
        self._subscriber_count = 0
        self._frame_ready = threading.Event()

        self._raw_frame: Optional[np.ndarray] = None
        self._raw_lock = threading.Lock()
        self._cap_thread: Optional[threading.Thread] = None

        self._processed_frame: Optional[np.ndarray] = None
        self._processed_frame_id: int = -1
        self._processed_lock = threading.Lock()

    @property
    def camera_id(self) -> str:
        return self.config.id

    @property
    def subscriber_count(self) -> int:
        return self._subscriber_count

    def add_subscriber(self) -> None:
        self._subscriber_count += 1
        if self._subscriber_count == 1:
            self.start()

    def remove_subscriber(self) -> None:
        self._subscriber_count = max(0, self._subscriber_count - 1)
        if self._subscriber_count == 0:
            self.stop()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._frame_ready.clear()

        self._cap_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._cap_thread.start()

    def stop(self) -> None:
        self._running = False
        if self._cap_thread is not None:
            self._cap_thread.join(timeout=5)
        self._cap_thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._processed_lock:
            self._processed_frame = None
            self._processed_frame_id = -1
        with self._raw_lock:
            self._raw_frame = None
        self._frame_ready.clear()

    def _init_rtsp(self) -> bool:
        cap = cv2.VideoCapture(self.config.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return False
        self._cap = cap
        return True

    def _capture_loop(self) -> None:
        if not self._init_rtsp():
            self._use_test_frame = True

        while self._running:
            loop_start = time.perf_counter()

            if self._use_test_frame:
                frame = self._generate_test_frame()
            else:
                ret, frame = self._cap.read()
                if not ret:
                    self._cap.release()
                    self._use_test_frame = True
                    frame = self._generate_test_frame()

            with self._raw_lock:
                self._raw_frame = frame.copy()
            self._frame_id += 1

            self.batch_detector.submit(self.config.id, frame)

            elapsed = time.perf_counter() - loop_start
            sleep_time = FRAME_INTERVAL - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        if self._cap is not None:
            self._cap.release()
            self._cap = None


    def _generate_test_frame(self) -> np.ndarray:
        global OBJ_CENTER_X
        OBJ_CENTER_X = (OBJ_CENTER_X + 3) % (TEST_FRAME_W + 100)
        cx = OBJ_CENTER_X - 50

        frame = np.zeros((TEST_FRAME_H, TEST_FRAME_W, 3), dtype=np.uint8)

        for y in range(TEST_FRAME_H):
            color_val = int(40 + (y / TEST_FRAME_H) * 30)
            frame[y, :] = (color_val + 10, color_val, color_val - 10)

        cv2.rectangle(frame, (20, 20), (TEST_FRAME_W - 20, TEST_FRAME_H - 20), (60, 60, 60), 1)

        colors = [(80, 80, 200), (80, 200, 80), (200, 80, 80)]
        for i, offset in enumerate([-60, 20, 100]):
            bx = cx + offset
            if 0 < bx < TEST_FRAME_W:
                cv2.rectangle(
                    frame,
                    (bx, TEST_FRAME_H // 2 - 30),
                    (bx + 60, TEST_FRAME_H // 2 + 10),
                    colors[i % 3],
                    -1,
                )

        cv2.putText(
            frame,
            f"{self.config.name} | frame: {self._frame_id}",
            (25, TEST_FRAME_H - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        return frame

    def get_latest_frame(self) -> tuple[Optional[np.ndarray], int]:
        with self._processed_lock:
            if self._processed_frame is None:
                return None, -1
            return self._processed_frame.copy(), self._processed_frame_id

    def get_raw_frame(self) -> tuple[Optional[np.ndarray], int]:
        with self._raw_lock:
            if self._raw_frame is None:
                return None, -1
            return self._raw_frame.copy(), self._frame_id

    def set_processed_frame(self, frame: np.ndarray) -> None:
        processed_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self._processed_lock:
            self._processed_frame = processed_rgb
            self._processed_frame_id = self._frame_id
        if not self._frame_ready.is_set():
            self._frame_ready.set()
