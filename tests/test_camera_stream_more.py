"""CameraStream 扩展测试:启动延迟、重连抖动、解码FPS、健康快照、生命期管理。"""

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.modules.stream.camera_stream import CameraStream, RTSP_STALE_FRAME_SECONDS, PROCESSED_STALE_FRAME_SECONDS


def _camera_config(camera_id="cam-test"):
    return SimpleNamespace(id=camera_id, name=f"cam {camera_id}",
                          url="rtsp://example.test/stream", longitude=0, latitude=0)


class _FakeDetector:
    def submit(self, *args, **kwargs):
        pass


class RecordingDetector:
    def __init__(self):
        self.submissions = []

    def submit(self, cam_id, frame, **kwargs):
        self.submissions.append((cam_id, kwargs))


class StartupDelayTest(unittest.TestCase):
    def test_delay_is_deterministic_and_between_zero_and_spread(self) -> None:
        stream = CameraStream(_camera_config("cam-05"), _FakeDetector())

        delay = stream._startup_delay()

        self.assertGreaterEqual(delay, 0.0)
        self.assertLess(delay, 2.0)  # RTSP_STARTUP_SPREAD_SECONDS = 2.0

    def test_same_camera_returns_same_delay(self) -> None:
        a = CameraStream(_camera_config("cam-01"), _FakeDetector())._startup_delay()
        b = CameraStream(_camera_config("cam-01"), _FakeDetector())._startup_delay()

        self.assertEqual(a, b)

    def test_different_cameras_have_different_delays(self) -> None:
        a = CameraStream(_camera_config("cam-01"), _FakeDetector())._startup_delay()
        b = CameraStream(_camera_config("cam-30"), _FakeDetector())._startup_delay()

        self.assertNotEqual(a, b)


class WaitToReconnectTest(unittest.TestCase):
    def test_returns_true_when_stop_event_is_set(self) -> None:
        stream = CameraStream(_camera_config("cam-01"), _FakeDetector())

        stream._stop_event.set()
        result = stream._wait_to_reconnect(1.0)

        self.assertTrue(result)
        stream._stop_event.clear()

    def test_returns_false_when_stop_event_times_out(self) -> None:
        stream = CameraStream(_camera_config("cam-01"), _FakeDetector())

        result = stream._wait_to_reconnect(0.001)

        self.assertFalse(result)

    def test_jitter_stays_within_fifteen_percent(self) -> None:
        stream = CameraStream(_camera_config("cam-01"), _FakeDetector())
        stream._reconnect_count = 0

        with patch("random.uniform", return_value=1.0):
            with patch.object(stream._stop_event, "wait", return_value=False) as mock_wait:
                stream._wait_to_reconnect(2.0)

                mock_wait.assert_called_once_with(2.0)


class EncodeDecodeFPSTest(unittest.TestCase):
    def test_decode_fps_resets_every_five_seconds(self) -> None:
        stream = CameraStream(_camera_config("cam-01"), _FakeDetector())

        base = time.perf_counter()
        for i in range(15):
            stream._record_decoded_frame(base + i * 0.5)

        with stream._state_lock:
            self.assertEqual(stream._decode_frame_count, 15)
            self.assertGreater(stream._decode_fps, 0.0)

    def test_decode_fps_is_zero_before_first_full_window(self) -> None:
        stream = CameraStream(_camera_config("cam-01"), _FakeDetector())

        self.assertEqual(stream._decode_fps, 0.0)


class TestFrameGenerationTest(unittest.TestCase):
    def test_output_has_correct_shape(self) -> None:
        stream = CameraStream(_camera_config("cam-01"), _FakeDetector())

        frame = stream._generate_test_frame()

        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertEqual(frame.dtype, np.uint8)


class SetProcessedFrameTest(unittest.TestCase):
    def test_converts_bgr_to_rgb_and_sets_all_fields(self) -> None:
        stream = CameraStream(_camera_config("cam-01"), _FakeDetector())
        bgr_frame = np.zeros((16, 16, 3), dtype=np.uint8)
        bgr_frame[:, :, 2] = 255  # Red channel in BGR — becomes Red in RGB at index 0
        captured_at = time.perf_counter()

        stream.set_processed_frame(bgr_frame, frame_id=42, captured_at=captured_at)

        with stream._processed_lock:
            self.assertEqual(stream._processed_frame_id, 42)
            self.assertEqual(stream._processed_captured_at, captured_at)
            self.assertIsNotNone(stream._processed_frame)
            self.assertEqual(stream._processed_frame[0, 0, 0], 255)

    def test_defaults_frame_id_to_raw_count_when_none(self) -> None:
        stream = CameraStream(_camera_config("cam-01"), _FakeDetector())
        stream._frame_id = 99

        stream.set_processed_frame(np.zeros((8, 8, 3), dtype=np.uint8))

        with stream._processed_lock:
            self.assertEqual(stream._processed_frame_id, 99)

    def test_frame_ready_event_is_set(self) -> None:
        stream = CameraStream(_camera_config("cam-01"), _FakeDetector())

        self.assertFalse(stream._frame_ready.is_set())
        stream.set_processed_frame(np.zeros((8, 8, 3), dtype=np.uint8))

        self.assertTrue(stream._frame_ready.is_set())


class HealthSnapshotTest(unittest.TestCase):
    def test_returns_offline_state_for_never_started_stream(self) -> None:
        stream = CameraStream(_camera_config("cam-01"), _FakeDetector())

        snapshot = stream.health_snapshot()

        self.assertEqual(snapshot["camera_id"], "cam-01")
        self.assertEqual(snapshot["state"], "stopped")
        self.assertFalse(snapshot["online"])
        self.assertIsNone(snapshot["last_frame_age_seconds"])
        self.assertEqual(snapshot["subscribers"], 0)

    def test_online_stream_shows_stale_fields(self) -> None:
        stream = CameraStream(_camera_config("cam-01"), _FakeDetector())
        now = time.perf_counter()
        with stream._state_lock:
            stream._connection_state = "online"
            stream._last_frame_at = now - 0.5

        snapshot = stream.health_snapshot()

        self.assertEqual(snapshot["state"], "online")
        self.assertTrue(snapshot["online"])
        self.assertIsNotNone(snapshot["last_frame_age_seconds"])

    def test_connection_online_but_frame_too_old_is_not_online(self) -> None:
        stream = CameraStream(_camera_config("cam-01"), _FakeDetector())
        with stream._state_lock:
            stream._connection_state = "online"
            stream._last_frame_at = time.perf_counter() - RTSP_STALE_FRAME_SECONDS - 10

        snapshot = stream.health_snapshot()

        self.assertFalse(snapshot["online"])


class IsOnlineTest(unittest.TestCase):
    def test_online_when_connected_with_fresh_frame(self) -> None:
        stream = CameraStream(_camera_config("cam-01"), _FakeDetector())
        with stream._state_lock:
            stream._connection_state = "online"
            stream._last_frame_at = time.perf_counter()

        self.assertTrue(stream.is_online)

    def test_not_online_before_first_frame(self) -> None:
        stream = CameraStream(_camera_config("cam-01"), _FakeDetector())
        with stream._state_lock:
            stream._connection_state = "online"

        self.assertFalse(stream.is_online)

    def test_not_online_when_stale(self) -> None:
        stream = CameraStream(_camera_config("cam-01"), _FakeDetector())
        with stream._state_lock:
            stream._connection_state = "online"
            stream._last_frame_at = time.perf_counter() - RTSP_STALE_FRAME_SECONDS - 1

        self.assertFalse(stream.is_online)


class SubscriberLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.stream = CameraStream(_camera_config("cam-01"), _FakeDetector())

    def test_add_subscriber_increments_count_and_starts_on_first(self) -> None:
        self.assertEqual(self.stream.subscriber_count, 0)

        self.stream.add_subscriber()
        self.assertEqual(self.stream.subscriber_count, 1)

        self.stream.add_subscriber()
        self.assertEqual(self.stream.subscriber_count, 2)

    def test_remove_subscriber_always_signals_stop_at_zero(self) -> None:
        self.stream._subscriber_count = 1

        should_stop = self.stream.remove_subscriber()
        self.assertTrue(should_stop)
        self.assertEqual(self.stream.subscriber_count, 0)

        # 重复取消仍返回 True（count 保持在 0）
        should_stop = self.stream.remove_subscriber()
        self.assertTrue(should_stop)
        self.assertEqual(self.stream.subscriber_count, 0)

    def test_remove_subscriber_with_stop_if_unused_false(self) -> None:
        self.stream._subscriber_count = 1

        should_stop = self.stream.remove_subscriber(stop_if_unused=False)

        self.assertTrue(should_stop)
        self.assertEqual(self.stream.subscriber_count, 0)


class CaptureMethodsTest(unittest.TestCase):
    def test_set_connection_state_transitions(self) -> None:
        stream = CameraStream(_camera_config("cam-01"), _FakeDetector())

        stream._set_connection_state("connecting")
        self.assertEqual(stream.connection_state, "connecting")

        stream._set_connection_state("online")
        self.assertEqual(stream.connection_state, "online")

    def test_close_capture_with_expected_arg(self) -> None:
        stream = CameraStream(_camera_config("cam-01"), _FakeDetector())
        capture_a = SimpleNamespace(released=False, release=lambda: setattr(capture_a, "released", True))
        capture_b = SimpleNamespace(released=False, release=lambda: setattr(capture_b, "released", True))
        stream._cap = capture_a

        # expected 与当前不同:释放 expected,不修改 _cap
        stream._close_capture(expected=capture_b)

        self.assertTrue(capture_b.released)
        self.assertFalse(capture_a.released)
        self.assertIs(stream._cap, capture_a)

    def test_close_capture_without_arg_releases_and_clears(self) -> None:
        stream = CameraStream(_camera_config("cam-01"), _FakeDetector())
        capture = SimpleNamespace(released=False, release=lambda: setattr(capture, "released", True))
        stream._cap = capture

        stream._close_capture()

        self.assertTrue(capture.released)
        self.assertIsNone(stream._cap)

    def test_request_stop_signals_event(self) -> None:
        stream = CameraStream(_camera_config("cam-01"), _FakeDetector())

        stream._running = True
        stream.request_stop()

        self.assertFalse(stream._running)
        self.assertTrue(stream._stop_event.is_set())

    def test_get_raw_frame_returns_copy_when_fresh(self) -> None:
        stream = CameraStream(_camera_config("cam-01"), _FakeDetector())
        original = np.ones((4, 4, 3), dtype=np.uint8) * 100
        stream._raw_frame = original
        stream._raw_frame_id = 7
        stream._last_frame_at = time.perf_counter()

        frame, frame_id = stream.get_raw_frame()
        self.assertEqual(frame_id, 7)
        # 确认是拷贝而非引用
        frame[0, 0, 0] = 0
        self.assertEqual(original[0, 0, 0], 100)



if __name__ == "__main__":
    unittest.main()
