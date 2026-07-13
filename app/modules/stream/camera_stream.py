import logging
import random
import threading
import time
import zlib
from typing import Any, Optional

import cv2
import numpy as np

from app.modules.yolo.batch_detector import BatchDetector
from app.config import settings
from app.utils.camera_manager import CameraConfig

logger = logging.getLogger(__name__)

TEST_FRAME_W = 640
TEST_FRAME_H = 480
TARGET_FPS = settings.STREAM_FPS
FRAME_INTERVAL = 1.0 / TARGET_FPS
VIEWER_DETECTION_INTERVAL = 1.0 / settings.YOLO_VIEWER_FPS
BACKGROUND_DETECTION_INTERVAL = 1.0 / settings.YOLO_BACKGROUND_FPS

OBJ_CENTER_X = 0

# 每路仅在内存中保留最新帧；真正的有界丢旧队列位于 BatchDetector。
MAX_PENDING_PER_CAMERA = 1
DETECT_RESIZE_WIDTH = settings.DETECT_RESIZE_WIDTH

# RTSP 连接参数。OpenCV/FFmpeg 的默认网络超时在不同构建中差异很大，必须显式限制。
RTSP_OPEN_TIMEOUT_MS = settings.RTSP_OPEN_TIMEOUT_MS
RTSP_READ_TIMEOUT_MS = settings.RTSP_READ_TIMEOUT_MS
RTSP_CONNECT_CONCURRENCY = settings.RTSP_CONNECT_CONCURRENCY
RTSP_RECONNECT_DELAY = settings.RTSP_RECONNECT_DELAY
RTSP_MAX_RECONNECT_DELAY = max(
    RTSP_RECONNECT_DELAY,
    settings.RTSP_MAX_RECONNECT_DELAY,
)
RTSP_STARTUP_SPREAD_SECONDS = settings.RTSP_STARTUP_SPREAD_SECONDS
RTSP_STALE_FRAME_SECONDS = settings.RTSP_STALE_FRAME_SECONDS
PROCESSED_STALE_FRAME_SECONDS = max(
    RTSP_STALE_FRAME_SECONDS,
    settings.PROCESSED_STALE_FRAME_SECONDS,
)
RTSP_ENABLE_HW_ACCELERATION = settings.RTSP_HW_ACCELERATION
_RTSP_CONNECT_SEMAPHORE = threading.BoundedSemaphore(RTSP_CONNECT_CONCURRENCY)


class CameraStream:
    """一路 RTSP 的拉流、最新帧缓存和连接健康状态。"""

    def __init__(self, config: CameraConfig, batch_detector: BatchDetector):
        self.config = config
        self.batch_detector = batch_detector
        self._frame_id = 0
        self._raw_frame_id = -1
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._subscriber_count = 0
        self._viewer_count = 0
        self._subscriber_lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._stop_event = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._cap_thread: Optional[threading.Thread] = None
        self._cap_lock = threading.Lock()

        self._raw_frame: Optional[np.ndarray] = None
        self._raw_captured_at = 0.0
        self._raw_lock = threading.Lock()

        self._processed_frame: Optional[np.ndarray] = None
        self._processed_frame_id = -1
        self._processed_captured_at = 0.0
        self._processed_updated_at = 0.0
        self._processed_lock = threading.Lock()

        self._state_lock = threading.Lock()
        self._connection_state = "stopped"
        self._last_frame_at = 0.0
        self._connected_at = 0.0
        self._reconnect_count = 0
        self._read_failure_count = 0
        self._decode_frame_count = 0
        self._published_frame_count = 0
        self._detection_submit_count = 0
        self._decode_fps = 0.0
        self._decode_window_started_at = time.perf_counter()
        self._decode_window_frames = 0

    @property
    def camera_id(self) -> str:
        return self.config.id

    @property
    def subscriber_count(self) -> int:
        with self._subscriber_lock:
            return self._subscriber_count

    @property
    def viewer_count(self) -> int:
        with self._subscriber_lock:
            return self._viewer_count

    @property
    def has_display_viewers(self) -> bool:
        return self.viewer_count > 0

    @property
    def connection_state(self) -> str:
        with self._state_lock:
            return self._connection_state

    @property
    def is_online(self) -> bool:
        with self._state_lock:
            return (
                self._connection_state == "online"
                and self._last_frame_at > 0
                and time.perf_counter() - self._last_frame_at <= RTSP_STALE_FRAME_SECONDS
            )

    def add_subscriber(self, *, viewer: bool = False) -> None:
        with self._subscriber_lock:
            self._subscriber_count += 1
            if viewer:
                self._viewer_count += 1
            should_start = self._subscriber_count == 1
        if should_start:
            self.start()

    def remove_subscriber(
        self,
        *,
        viewer: bool = False,
        stop_if_unused: bool = True,
    ) -> bool:
        with self._subscriber_lock:
            self._subscriber_count = max(0, self._subscriber_count - 1)
            if viewer:
                self._viewer_count = max(0, self._viewer_count - 1)
            should_stop = self._subscriber_count == 0
        if should_stop and stop_if_unused:
            self.stop()
        return should_stop

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._running:
                return
            self._running = True
            self._stop_event.clear()
        self._frame_ready.clear()
        self._set_connection_state("connecting")
        self._cap_thread = threading.Thread(
            target=self._capture_worker,
            name=f"rtsp-{self.config.id}",
            daemon=True,
        )
        self._cap_thread.start()

    def stop(self) -> None:
        self.request_stop()
        thread = self._cap_thread
        if thread is not None:
            timeout = max(RTSP_OPEN_TIMEOUT_MS, RTSP_READ_TIMEOUT_MS) / 1000.0 + 2.0
            thread.join(timeout=timeout)
            if thread.is_alive():
                # Releasing VideoCapture from a second thread can crash FFmpeg.
                # Leave the daemon to observe its configured timeout and let its
                # own finally block release the native handle safely.
                logger.error(
                    "[%s] RTSP capture thread is still stopping after %.1fs",
                    self.config.id,
                    timeout,
                )
            else:
                self._cap_thread = None
        with self._processed_lock:
            self._processed_frame = None
            self._processed_frame_id = -1
            self._processed_captured_at = 0.0
            self._processed_updated_at = 0.0
        with self._raw_lock:
            self._raw_frame = None
            self._raw_frame_id = -1
            self._raw_captured_at = 0.0
        self._frame_ready.clear()
        self._set_connection_state("stopped")

    def request_stop(self) -> None:
        """Signal a blocked capture to stop without waiting for its read timeout."""
        with self._lifecycle_lock:
            self._running = False
            self._stop_event.set()

    def _capture_worker(self) -> None:
        """Own the native capture handle for its full lifetime."""
        try:
            self._capture_loop()
        except Exception:
            logger.exception("[%s] RTSP capture worker crashed", self.config.id)
        finally:
            self._close_capture()
            with self._state_lock:
                if self._connection_state != "stopped":
                    self._connection_state = "offline"

    def _set_connection_state(self, state: str) -> None:
        with self._state_lock:
            self._connection_state = state

    def _set_capture(self, cap: cv2.VideoCapture) -> None:
        with self._cap_lock:
            self._cap = cap

    def _close_capture(self, expected: Optional[cv2.VideoCapture] = None) -> None:
        with self._cap_lock:
            cap = self._cap
            if expected is not None and cap is not expected:
                cap = expected
            else:
                self._cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                logger.debug("[%s] RTSP capture release failed", self.config.id, exc_info=True)

    def _capture_parameters(self) -> list[int]:
        params: list[int] = []
        open_timeout = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
        read_timeout = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
        if open_timeout is not None:
            params.extend([int(open_timeout), RTSP_OPEN_TIMEOUT_MS])
        if read_timeout is not None:
            params.extend([int(read_timeout), RTSP_READ_TIMEOUT_MS])
        if RTSP_ENABLE_HW_ACCELERATION:
            hw_property = getattr(cv2, "CAP_PROP_HW_ACCELERATION", None)
            hw_any = getattr(cv2, "VIDEO_ACCELERATION_ANY", None)
            if hw_property is not None and hw_any is not None:
                params.extend([int(hw_property), int(hw_any)])
        return params

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        while not self._stop_event.is_set():
            if _RTSP_CONNECT_SEMAPHORE.acquire(timeout=0.2):
                break
        else:
            return None

        try:
            params = self._capture_parameters()
            try:
                if params:
                    cap = cv2.VideoCapture(self.config.url, cv2.CAP_FFMPEG, params)
                else:
                    cap = cv2.VideoCapture(self.config.url, cv2.CAP_FFMPEG)
            except (TypeError, cv2.error):
                # 兼容不支持带 params 构造函数的 OpenCV 构建。
                logger.warning(
                    "[%s] OpenCV capture parameters unsupported; falling back",
                    self.config.id,
                )
                cap = cv2.VideoCapture(self.config.url, cv2.CAP_FFMPEG)
        finally:
            _RTSP_CONNECT_SEMAPHORE.release()

        if self._stop_event.is_set() or not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            return None

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _startup_delay(self) -> float:
        if RTSP_STARTUP_SPREAD_SECONDS <= 0:
            return 0.0
        bucket = zlib.crc32(self.config.id.encode("utf-8")) % 1000
        return RTSP_STARTUP_SPREAD_SECONDS * bucket / 1000.0

    def _wait_to_reconnect(self, delay: float) -> bool:
        jittered = delay * random.uniform(0.85, 1.15)
        logger.info(
            "[%s] RTSP reconnect in %.1fs (attempt=%d)",
            self.config.id,
            jittered,
            self._reconnect_count,
        )
        return self._stop_event.wait(jittered)

    def _capture_loop(self) -> None:
        if self._stop_event.wait(self._startup_delay()):
            return

        reconnect_delay = RTSP_RECONNECT_DELAY
        stable_frames = 0
        last_publish_at = 0.0
        last_detection_submit_at = 0.0

        while self._running:
            self._set_connection_state("connecting" if self._reconnect_count == 0 else "reconnecting")
            cap = self._open_capture()
            if cap is None:
                self._reconnect_count += 1
                self._set_connection_state("offline")
                if self._wait_to_reconnect(reconnect_delay):
                    break
                reconnect_delay = min(reconnect_delay * 1.5, RTSP_MAX_RECONNECT_DELAY)
                continue

            self._set_capture(cap)
            first_frame = True
            stable_frames = 0

            while self._running:
                ret, frame = cap.read()
                if not self._running:
                    break
                if not ret or frame is None:
                    self._read_failure_count += 1
                    self._reconnect_count += 1
                    self._set_connection_state("reconnecting")
                    logger.warning(
                        "[%s] RTSP read failed; reconnecting (failures=%d)",
                        self.config.id,
                        self._read_failure_count,
                    )
                    break

                now = time.perf_counter()
                if first_frame:
                    first_frame = False
                    with self._state_lock:
                        self._connected_at = now
                    self._set_connection_state("online")
                    logger.info("[%s] RTSP connected", self.config.id)

                stable_frames += 1
                if stable_frames >= TARGET_FPS * 2:
                    reconnect_delay = RTSP_RECONNECT_DELAY

                self._record_decoded_frame(now)

                # 持续读取以排空 RTSP/FFmpeg 内部缓存，只按目标帧率发布最新帧。
                if now - last_publish_at < FRAME_INTERVAL:
                    continue
                last_publish_at = now

                if DETECT_RESIZE_WIDTH and frame.shape[1] > DETECT_RESIZE_WIDTH:
                    scale = DETECT_RESIZE_WIDTH / frame.shape[1]
                    published = cv2.resize(
                        frame,
                        (DETECT_RESIZE_WIDTH, max(1, int(frame.shape[0] * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    published = frame

                # request_stop() uses the same lock.  Once it returns, this stream
                # cannot publish another frame into a newly registered camera
                # generation with the same id.
                with self._lifecycle_lock:
                    if not self._running:
                        break
                    with self._raw_lock:
                        self._frame_id += 1
                        frame_id = self._frame_id
                        self._raw_frame = published
                        self._raw_frame_id = frame_id
                        self._raw_captured_at = now
                    with self._state_lock:
                        self._last_frame_at = now
                        self._published_frame_count += 1

                    with self._subscriber_lock:
                        has_viewer = self._viewer_count > 0
                    detection_interval = (
                        VIEWER_DETECTION_INTERVAL
                        if has_viewer
                        else BACKGROUND_DETECTION_INTERVAL
                    )
                    if now - last_detection_submit_at >= detection_interval:
                        last_detection_submit_at = now
                        with self._state_lock:
                            self._detection_submit_count += 1
                        self.batch_detector.submit(
                            self.config.id,
                            published,
                            frame_id=frame_id,
                            captured_at=now,
                            source_stream=self,
                            priority=has_viewer,
                        )

            self._close_capture(expected=cap)
            if not self._running:
                break
            if self._wait_to_reconnect(reconnect_delay):
                break
            reconnect_delay = min(reconnect_delay * 1.5, RTSP_MAX_RECONNECT_DELAY)

        self._close_capture()

    def _record_decoded_frame(self, now: float) -> None:
        with self._state_lock:
            self._decode_frame_count += 1
            self._decode_window_frames += 1
            elapsed = now - self._decode_window_started_at
            if elapsed >= 5.0:
                self._decode_fps = self._decode_window_frames / elapsed
                self._decode_window_frames = 0
                self._decode_window_started_at = now

    def _generate_test_frame(self) -> np.ndarray:
        """仅保留给单元测试和显式诊断使用；RTSP 故障时不再伪装为在线画面。"""
        global OBJ_CENTER_X
        OBJ_CENTER_X = (OBJ_CENTER_X + 3) % (TEST_FRAME_W + 100)
        cx = OBJ_CENTER_X - 50
        frame = np.zeros((TEST_FRAME_H, TEST_FRAME_W, 3), dtype=np.uint8)
        for y in range(TEST_FRAME_H):
            value = 40 + int((y / TEST_FRAME_H) * 30)
            frame[y, :] = (value + 10, value, value - 10)
        cv2.putText(
            frame,
            f"{self.config.name} | diagnostic",
            (25, TEST_FRAME_H - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        return frame

    def get_latest_frame(self, *, copy: bool = True) -> tuple[Optional[np.ndarray], int]:
        """返回最新处理帧；RTSP 或 YOLO 长时间无更新时返回空帧。"""
        now = time.perf_counter()
        with self._state_lock:
            raw_is_stale = (
                self._last_frame_at <= 0
                or now - self._last_frame_at > RTSP_STALE_FRAME_SECONDS
            )
        with self._processed_lock:
            processed_is_stale = (
                self._processed_updated_at <= 0
                or now - self._processed_updated_at > PROCESSED_STALE_FRAME_SECONDS
            )
            if self._processed_frame is None or raw_is_stale or processed_is_stale:
                return None, -1
            frame = self._processed_frame.copy() if copy else self._processed_frame
            return frame, self._processed_frame_id

    def get_raw_frame(
        self,
        *,
        copy: bool = True,
    ) -> tuple[Optional[np.ndarray], int]:
        """返回最新原始帧副本；断流后不向分析模块提供陈旧画面。"""
        now = time.perf_counter()
        with self._state_lock:
            if self._last_frame_at <= 0 or now - self._last_frame_at > RTSP_STALE_FRAME_SECONDS:
                return None, -1
        with self._raw_lock:
            if self._raw_frame is None:
                return None, -1
            frame = self._raw_frame.copy() if copy else self._raw_frame
            return frame, self._raw_frame_id

    def set_processed_frame(
        self,
        frame: np.ndarray,
        *,
        frame_id: Optional[int] = None,
        captured_at: Optional[float] = None,
    ) -> None:
        processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self._processed_lock:
            self._processed_frame = processed
            self._processed_frame_id = self._frame_id if frame_id is None else frame_id
            self._processed_captured_at = captured_at or time.perf_counter()
            self._processed_updated_at = time.perf_counter()
        self._frame_ready.set()

    def health_snapshot(self) -> dict[str, Any]:
        now = time.perf_counter()
        with self._subscriber_lock:
            subscriber_count = self._subscriber_count
            viewer_count = self._viewer_count
        with self._raw_lock:
            raw_shape = self._raw_frame.shape if self._raw_frame is not None else None
        with self._processed_lock:
            processed_age = (
                None
                if self._processed_updated_at <= 0
                else now - self._processed_updated_at
            )
        with self._state_lock:
            last_frame_age = None if self._last_frame_at <= 0 else now - self._last_frame_at
            return {
                "camera_id": self.config.id,
                "state": self._connection_state,
                "online": (
                    self._connection_state == "online"
                    and last_frame_age is not None
                    and last_frame_age <= RTSP_STALE_FRAME_SECONDS
                ),
                "last_frame_age_seconds": round(last_frame_age, 3) if last_frame_age is not None else None,
                "decode_fps": round(self._decode_fps, 2),
                "decoded_frames": self._decode_frame_count,
                "published_frames": self._published_frame_count,
                "detection_submissions": self._detection_submit_count,
                "reconnect_count": self._reconnect_count,
                "read_failure_count": self._read_failure_count,
                "subscribers": subscriber_count,
                "viewers": viewer_count,
                "frame_width": int(raw_shape[1]) if raw_shape is not None else None,
                "frame_height": int(raw_shape[0]) if raw_shape is not None else None,
                "processed_frame_age_seconds": (
                    round(processed_age, 3) if processed_age is not None else None
                ),
            }
