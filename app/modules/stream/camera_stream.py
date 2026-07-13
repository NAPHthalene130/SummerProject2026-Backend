import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

from app.modules.yolo.batch_detector import BatchDetector
from app.utils.camera_manager import CameraConfig

logger = logging.getLogger(__name__)

TEST_FRAME_W = 640
TEST_FRAME_H = 480
TARGET_FPS = 15
FRAME_INTERVAL = 1.0 / TARGET_FPS

OBJ_CENTER_X = 0

# 多线程解码优化相关常量
# 每路摄像头一个 capture 线程（已存在），这里聚焦于降低单路负载与稳健性。
# 有界缓冲：原始帧与待检帧只保留最新一帧（丢旧策略），避免积压导致卡顿
MAX_PENDING_PER_CAMERA = 1
# 解码后是否缩放到较小分辨率再送检测，降低 YOLO 推理与拷贝开销（0=不缩放）
DETECT_RESIZE_WIDTH = 0

# RTSP 断连重连参数
RTSP_RECONNECT_DELAY = 1.0
RTSP_MAX_RECONNECT_DELAY = 30.0


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
        # 后台解码线程
        self._cap_thread: Optional[threading.Thread] = None

        # 原始帧缓冲：丢弃旧帧策略，避免解码线程被消费者拖慢
        self._raw_frame: Optional[np.ndarray] = None
        self._raw_lock = threading.Lock()

        # 处理后帧缓冲
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
        self._close_capture()
        with self._processed_lock:
            self._processed_frame = None
            self._processed_frame_id = -1
        with self._raw_lock:
            self._raw_frame = None
        self._frame_ready.clear()

    def _close_capture(self) -> None:
        cap = self._cap
        self._cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def _init_rtsp(self) -> bool:
        """打开 RTSP 流。成功返回 True，失败返回 False（调用方降级为测试帧）。"""
        cap = self._cap = cv2.VideoCapture(self.config.url, cv2.CAP_FFMPEG)
        return cap.isOpened()

    def _reconnect(self) -> bool:
        """断连后带指数退避重连，最多持续重试直到成功或停止。返回 True=成功，False=已停止。"""
        delay = RTSP_RECONNECT_DELAY
        self._close_capture()
        while self._running:
            logger.info("[%s] RTSP reconnect %.1fs: %s", self.config.id, delay, self.config.url)
            time.sleep(delay)
            if not self._running:
                return False
            cap = cv2.VideoCapture(self.config.url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                self._cap = cap
                return True
            delay = min(delay * 1.5, RTSP_MAX_RECONNECT_DELAY)
        return False

    def _capture_loop(self) -> None:
        if not self._init_rtsp():
            self._use_test_frame = True
            logger.warning("[%s] RTSP open failed, using test frame", self.config.id)
        self._use_test_frame = not (self._cap is not None and self._cap.isOpened())
        while self._running:
            loop_start = time.perf_counter()
            if self._use_test_frame:
                frame = self._generate_test_frame()
            else:
                ret, frame = self._cap.read()  # 阻塞解码
                if not ret:
                    logger.warning("[%s] RTSP read failed, reconnecting", self.config.id)
                    if self._reconnect():
                        self._use_test_frame = False
                    else:
                        self._use_test_frame = True
                    continue
            # 写最新帧（丢旧策略）
            with self._raw_lock:
                self._raw_frame = frame
            self._frame_id += 1
            # 送检：直接引用最新帧；探测器内部按需 copy
            self.batch_detector.submit(self.config.id, frame)
            elapsed = time.perf_counter() - loop_start
            sleep_time = FRAME_INTERVAL - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        self._close_capture()

    def _generate_test_frame(self) -> np.ndarray:
        global OBJ_CENTER_X
        OBJ_CENTER_X = (OBJ_CENTER_X + 3) % (TEST_FRAME_W + 100)
        cx = OBJ_CENTER_X - 50
        frame = np.zeros((TEST_FRAME_H, TEST_FRAME_W, 3), dtype=np.uint8)
        for y in range(TEST_FRAME_H):
            v = 40 + int((y / TEST_FRAME_H) * 30)
            frame[y, :] = (v + 10, v, v - 10)
        cv2.rectangle(frame, (20, 20), (TEST_FRAME_W - 20, TEST_FRAME_H - 20), (60, 60, 60), 1)
        for i, off in enumerate((-60, 20, 100)):
            bx = cx + off
            if 0 < bx < TEST_FRAME_W:
                cv2.rectangle(frame, (bx, TEST_FRAME_H // 2 - 30), (bx + 60, TEST_FRAME_H // 2 + 10), [(80, 80, 200), (80, 200, 80), (200, 80, 80)][i % 3], -1)
        cv2.putText(frame, f"{self.config.name} | frame: {self._frame_id}",
                    (25, TEST_FRAME_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return frame

    def get_latest_frame(self) -> tuple[Optional[np.ndarray], int]:
        """取处理后帧(已转 RGB)。为防撕裂由探测器并发覆写，返回 copy。"""
        with self._processed_lock:
            if self._processed_frame is None:
                return None, -1
            return self._processed_frame.copy(), self._processed_frame_id

    def get_raw_frame(self) -> tuple[Optional[np.ndarray], int]:
        """取最新原始帧。copy 以防被解码线程覆写撕裂。"""
        with self._raw_lock:
            if self._raw_frame is None:
                return None, -1
            return self._raw_frame.copy(), self._frame_id

    def set_processed_frame(self, frame: np.ndarray) -> None:
        processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self._processed_lock:
            self._processed_frame = processed
            self._processed_frame_id = self._frame_id
        self._frame_ready.set()

