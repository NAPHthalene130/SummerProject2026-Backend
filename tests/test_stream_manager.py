"""StreamManager 单元测试:订阅/取消、健康快照与优雅停止(mocked 底层)。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.modules.stream.stream_manager import StreamManager


class _FakeStream:
    def __init__(self, *args, **kwargs):
        self.subscriber_count = 0
        self.started = False
        self._stopped = False

    def add_subscriber(self) -> None:
        self.subscriber_count += 1
        self.started = True

    def remove_subscriber(self, *, stop_if_unused: bool) -> bool:
        self.subscriber_count = max(0, self.subscriber_count - 1)
        return self.subscriber_count == 0

    def request_stop(self) -> None:
        self._stopped = True

    def stop(self) -> None:
        pass

    def health_snapshot(self) -> dict:
        return {"camera_id": "cam-test", "state": "online"}

    is_stopped = property(lambda self: self._stopped)


class _FakeDetector:
    def __init__(self):
        self.streams = {}
        self.started = False
        self.stopped = False
        self._perf = {"device": "cpu"}

    def register_stream(self, cam_id, stream) -> None:
        self.streams[cam_id] = stream

    def unregister_stream(self, cam_id) -> None:
        self.streams.pop(cam_id, None)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def performance_snapshot(self) -> dict:
        return self._perf


class StreamManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = StreamManager()
        self.manager._streams.clear()
        self.manager._batch_detector = None

    def test_get_stream_returns_none_for_unknown_camera(self) -> None:
        self.assertIsNone(self.manager.get_stream("cam-missing"))

    def test_get_all_streams_returns_copy(self) -> None:
        self.manager._streams["cam-01"] = _FakeStream()
        snapshot = self.manager.get_all_streams()
        snapshot.pop("cam-01")

        self.assertIn("cam-01", self.manager.get_all_streams())

    def test_subscribe_creates_stream_and_adds_subscriber(self) -> None:
        camera_config = SimpleNamespace(id="cam-01")
        detector = _FakeDetector()

        # 通过私有方法注入依赖，避免构造函数触发 RTSP 连接
        self.manager._batch_detector = detector

        with (
            patch.object(self.manager, "_resolve_camera", return_value=camera_config),
            patch("app.modules.stream.stream_manager.CameraStream", _FakeStream),
            patch.object(detector, "register_stream"),
        ):
            stream = self.manager.subscribe("cam-01")

        self.assertIsNotNone(stream)
        self.assertEqual(stream.subscriber_count, 1)

    def test_subscribe_resolves_camera_via_manager(self) -> None:
        self.manager._batch_detector = _FakeDetector()
        with (
            patch.object(self.manager, "_resolve_camera", return_value=None),
        ):
            stream = self.manager.subscribe("cam-999")

        self.assertIsNone(stream)

    def test_unsubscribe_removes_stream_when_last_subscriber_goes_away(self) -> None:
        stream = _FakeStream()
        stream.subscriber_count = 1
        self.manager._streams["cam-01"] = stream

        with patch.object(stream, "stop"):
            self.manager.unsubscribe("cam-01")

        self.assertNotIn("cam-01", self.manager._streams)

    def test_health_snapshots_aggregates_stream_health(self) -> None:
        self.manager._streams["cam-01"] = _FakeStream()
        self.manager._streams["cam-02"] = _FakeStream()

        snapshots = self.manager.health_snapshots()

        self.assertEqual(len(snapshots), 2)
        self.assertEqual(snapshots[0]["state"], "online")

    def test_pipeline_health_includes_detector(self) -> None:
        detector = _FakeDetector()
        self.manager._streams["cam-01"] = _FakeStream()
        self.manager._batch_detector = detector

        health = self.manager.pipeline_health()

        self.assertEqual(len(health["streams"]), 1)
        self.assertIsNotNone(health["detector"])
        self.assertEqual(health["detector"]["device"], "cpu")

    def test_pipeline_health_handles_missing_detector(self) -> None:
        self.manager._streams["cam-01"] = _FakeStream()
        self.manager._batch_detector = None

        health = self.manager.pipeline_health()

        self.assertIsNone(health["detector"])

    def test_stop_all_requests_stop_and_stops_streams_and_detector(self) -> None:
        stream = _FakeStream()
        detector = _FakeDetector()
        self.manager._streams["cam-01"] = stream
        self.manager._batch_detector = detector

        self.manager.stop_all()

        self.assertTrue(stream.is_stopped)
        self.assertTrue(detector.stopped)
        self.assertEqual(self.manager._streams, {})


if __name__ == "__main__":
    unittest.main()
