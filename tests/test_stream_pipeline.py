import asyncio
import threading
import time
from types import MethodType
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import HTTPException

from app.api.v1 import live
from app.modules.stream.camera_stream import CameraStream
from app.modules.stream.stream_manager import StreamManager
from app.modules.yolo.batch_detector import BatchDetector, PendingFrame, POST_PROCESS_WORKERS
from app.utils.camera_manager import CameraConfig


class _RecordingDetector:
    def __init__(self) -> None:
        self.submissions: list[tuple] = []

    def submit(self, *args, **kwargs) -> None:
        self.submissions.append((args, kwargs))


def _camera_config() -> CameraConfig:
    return CameraConfig(
        id="cam-test",
        name="test camera",
        url="rtsp://example.test/stream",
        longitude=0,
        latitude=0,
    )


def test_failed_rtsp_open_releases_capture() -> None:
    detector = _RecordingDetector()
    stream = CameraStream(_camera_config(), detector)  # type: ignore[arg-type]

    class ClosedCapture:
        released = False

        @staticmethod
        def isOpened() -> bool:
            return False

        def release(self) -> None:
            self.released = True

    capture = ClosedCapture()
    with patch(
        "app.modules.stream.camera_stream.cv2.VideoCapture",
        return_value=capture,
    ):
        assert stream._open_capture() is None

    assert capture.released is True


def test_initial_rtsp_failure_retries_and_publishes_source_metadata() -> None:
    detector = _RecordingDetector()
    stream = CameraStream(_camera_config(), detector)  # type: ignore[arg-type]

    class WorkingCapture:
        def __init__(self) -> None:
            self.read_count = 0
            self.released = False

        def read(self):
            self.read_count += 1
            if self.read_count == 1:
                return True, np.zeros((32, 48, 3), dtype=np.uint8)
            stream._running = False
            return False, None

        def release(self) -> None:
            self.released = True

    capture = WorkingCapture()
    stream._running = True
    stream._stop_event.clear()
    with (
        patch.object(stream, "_startup_delay", return_value=0.0),
        patch.object(stream, "_open_capture", side_effect=[None, capture]) as opener,
        patch.object(stream, "_wait_to_reconnect", return_value=False),
    ):
        stream._capture_loop()

    assert opener.call_count == 2
    assert capture.released is True
    assert len(detector.submissions) == 1
    args, kwargs = detector.submissions[0]
    assert args[0] == "cam-test"
    assert kwargs["frame_id"] == 1
    assert kwargs["captured_at"] > 0


def test_stale_rtsp_frame_is_not_sent_to_webrtc() -> None:
    stream = CameraStream(_camera_config(), _RecordingDetector())  # type: ignore[arg-type]
    now = time.perf_counter()
    with stream._state_lock:
        stream._connection_state = "online"
        stream._last_frame_at = now
    stream.set_processed_frame(
        np.zeros((24, 24, 3), dtype=np.uint8),
        frame_id=7,
        captured_at=now,
    )

    frame, frame_id = stream.get_latest_frame(copy=False)
    assert frame is not None
    assert frame_id == 7

    with stream._state_lock:
        stream._last_frame_at = 0.0
    assert stream.get_latest_frame(copy=False) == (None, -1)


def test_stop_does_not_release_native_capture_from_another_thread() -> None:
    stream = CameraStream(_camera_config(), _RecordingDetector())  # type: ignore[arg-type]

    class Capture:
        released = False

        def release(self) -> None:
            self.released = True

    class StillBlockedThread:
        @staticmethod
        def join(timeout=None) -> None:
            return None

        @staticmethod
        def is_alive() -> bool:
            return True

    capture = Capture()
    stream._cap = capture  # type: ignore[assignment]
    stream._cap_thread = StillBlockedThread()  # type: ignore[assignment]
    stream.stop()

    assert capture.released is False
    assert stream._cap is capture


def _bare_detector_for_pipeline_test() -> BatchDetector:
    detector = object.__new__(BatchDetector)
    detector.device = "cpu"
    detector._use_half = False
    detector._precision_kwargs = {}
    detector._running = True
    detector._post_pool = None
    detector._pending = {}
    detector._post_inflight = set()
    detector._pending_lock = threading.Lock()
    detector._pending_event = threading.Event()
    detector._streams = {}
    detector._streams_lock = threading.RLock()
    detector._metrics_lock = threading.Lock()
    detector._window_total = 0
    detector._window_batches = 0
    detector._lifetime_total = 0
    detector._fps = 0.0
    detector._last_fps = time.perf_counter()
    detector._processors = {}
    return detector


class _RecordingStream:
    def __init__(self) -> None:
        self.processed_frames: list[tuple] = []

    def set_processed_frame(self, frame, *, frame_id: int, captured_at: float) -> None:
        self.processed_frames.append((frame, frame_id, captured_at))


def test_gpu_inference_does_not_wait_for_cpu_post_processing() -> None:
    detector = _bare_detector_for_pipeline_test()
    post_gate = threading.Event()
    all_processed = threading.Event()
    processed_count = 0
    count_lock = threading.Lock()
    model_kwargs: dict = {}

    def model(frames, **kwargs):
        model_kwargs.update(kwargs)
        return [object() for _ in frames]

    def fake_process_single(self, cam_id, pending, result) -> None:
        nonlocal processed_count
        assert post_gate.wait(timeout=2.0)
        with count_lock:
            processed_count += 1
            if processed_count == POST_PROCESS_WORKERS:
                all_processed.set()

    detector.model = model
    detector._process_single = MethodType(fake_process_single, detector)
    detector._post_pool = __import__(
        "concurrent.futures",
        fromlist=["ThreadPoolExecutor"],
    ).ThreadPoolExecutor(max_workers=POST_PROCESS_WORKERS)

    streams: dict[str, _RecordingStream] = {}
    items = []
    for index in range(POST_PROCESS_WORKERS):
        camera_id = f"cam-{index}"
        stream = _RecordingStream()
        streams[camera_id] = stream
        detector._streams[camera_id] = stream
        items.append(
            (
                camera_id,
                PendingFrame(
                    np.zeros((16, 16, 3), dtype=np.uint8),
                    frame_id=index,
                    captured_at=time.perf_counter(),
                ),
            )
        )

    inference_returned = threading.Event()
    caller = threading.Thread(
        target=lambda: (
            detector._process_batch(items),
            inference_returned.set(),
        )
    )
    caller.start()
    returned_before_post_processing = inference_returned.wait(timeout=0.5)

    # 后处理被gate卡住期间,推理线程必须已经返回并把原始帧推给WebRTC
    assert returned_before_post_processing is True
    for index, stream in enumerate(streams.values()):
        assert len(stream.processed_frames) == 1
        assert stream.processed_frames[0][1] == index

    post_gate.set()
    caller.join(timeout=2.0)

    assert all_processed.wait(timeout=2.0)
    assert "half" not in model_kwargs
    assert "quantize" not in model_kwargs
    detector._post_pool.shutdown(wait=True)
    assert detector._post_inflight == set()


def test_inflight_camera_is_not_eligible_for_batching() -> None:
    detector = _bare_detector_for_pipeline_test()
    detector._pending["cam-1"] = PendingFrame(
        np.zeros((4, 4, 3), dtype=np.uint8),
        frame_id=1,
        captured_at=time.perf_counter(),
    )
    detector._post_inflight.add("cam-1")
    detector._pending["cam-2"] = PendingFrame(
        np.zeros((4, 4, 3), dtype=np.uint8),
        frame_id=2,
        captured_at=time.perf_counter(),
    )

    # 仅 inflight 的摄像头不计入可凑批数量,避免同一帧被重复后处理
    assert detector._eligible_pending_count() == 1

    taken = detector._take_batch()
    assert [cam_id for cam_id, _ in taken] == ["cam-1", "cam-2"]
    assert detector._pending_event.is_set() is False

    detector._pending["cam-3"] = PendingFrame(
        np.zeros((4, 4, 3), dtype=np.uint8),
        frame_id=3,
        captured_at=time.perf_counter(),
    )
    detector._pending_event.set()
    detector._take_batch()
    assert detector._pending_event.is_set() is False


def test_stopped_stream_cannot_reinsert_a_stale_yolo_frame() -> None:
    detector = _bare_detector_for_pipeline_test()
    current_stream = object()
    stopped_stream = object()
    detector._streams["cam-1"] = current_stream

    detector.submit(
        "cam-1",
        np.zeros((4, 4, 3), dtype=np.uint8),
        source_stream=stopped_stream,
    )
    assert detector._pending == {}

    detector.submit(
        "cam-1",
        np.zeros((4, 4, 3), dtype=np.uint8),
        source_stream=current_stream,
    )
    assert list(detector._pending) == ["cam-1"]


def test_slow_rtsp_stop_does_not_block_other_stream_manager_operations() -> None:
    stop_started = threading.Event()
    allow_stop = threading.Event()

    class SlowStream:
        subscriber_count = 1

        @staticmethod
        def remove_subscriber(*, stop_if_unused: bool) -> bool:
            assert stop_if_unused is False
            return True

        @staticmethod
        def request_stop() -> None:
            return None

        @staticmethod
        def stop() -> None:
            stop_started.set()
            assert allow_stop.wait(timeout=2.0)

    class FakeDetector:
        @staticmethod
        def unregister_stream(camera_id: str) -> None:
            assert camera_id == "cam-1"

    manager = object.__new__(StreamManager)
    manager._lock = threading.RLock()
    manager._streams = {"cam-1": SlowStream()}
    manager._batch_detector = FakeDetector()

    unsubscribe_thread = threading.Thread(target=manager.unsubscribe, args=("cam-1",))
    unsubscribe_thread.start()
    assert stop_started.wait(timeout=0.5)

    manager_was_responsive = threading.Event()
    reader = threading.Thread(
        target=lambda: (manager.get_all_streams(), manager_was_responsive.set())
    )
    reader.start()
    assert manager_was_responsive.wait(timeout=0.5)

    allow_stop.set()
    unsubscribe_thread.join(timeout=2.0)
    reader.join(timeout=2.0)


def test_webrtc_negotiation_failure_releases_subscription() -> None:
    class FakeManager:
        def __init__(self) -> None:
            self.unsubscribed: list[str] = []

        @staticmethod
        def subscribe(camera_id: str):
            return object()

        def unsubscribe(self, camera_id: str) -> None:
            self.unsubscribed.append(camera_id)

    class FakePeer:
        def __init__(self) -> None:
            self.connectionState = "new"

        @staticmethod
        def addTrack(track) -> None:
            return None

        @staticmethod
        def on(event_name: str):
            return lambda callback: callback

        @staticmethod
        async def setRemoteDescription(description) -> None:
            raise ValueError("bad sdp")

        async def close(self) -> None:
            self.connectionState = "closed"

    manager = FakeManager()

    async def exercise() -> None:
        with (
            patch.object(live, "StreamManager", return_value=manager),
            patch.object(live, "RTCPeerConnection", FakePeer),
            patch.object(live, "ProcessedVideoTrack", return_value=object()),
        ):
            with pytest.raises(HTTPException) as caught:
                await live.webrtc_offer(
                    "cam-test",
                    live.OfferRequest(sdp="invalid", type="offer"),
                )
            assert caught.value.status_code == 400

    asyncio.run(exercise())
    assert manager.unsubscribed == ["cam-test"]
    assert live._peer_releases == {}
