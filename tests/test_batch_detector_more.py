"""BatchDetector 扩展测试:提交过滤、BEV 矩阵、性能快照、流量查询。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.modules.yolo.batch_detector import BatchDetector


def _bare_detector():
    detector = object.__new__(BatchDetector)
    detector.device = "cpu"
    detector._use_half = False
    detector._precision_kwargs = {}
    detector._running = True
    detector._pending = {}
    detector._post_inflight = set()
    detector._pending_lock = __import__("threading").Lock()
    detector._pending_event = __import__("threading").Event()
    detector._streams = {}
    detector._streams_lock = __import__("threading").RLock()
    detector._metrics_lock = __import__("threading").Lock()
    detector._window_total = 0
    detector._window_batches = 0
    detector._lifetime_total = 0
    detector._fps = 0.0
    detector._last_fps = __import__("time").perf_counter()
    detector._processors = {}
    detector.model_path = "/fake/path/best.pt"
    detector._post_pool = None
    return detector


class SubmitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = _bare_detector()

    def test_stopped_stream_is_rejected(self) -> None:
        current_stream = object()
        stopped_stream = object()
        self.detector._streams["cam-1"] = current_stream

        self.detector.submit("cam-1", np.zeros((4, 4, 3), dtype=np.uint8),
                              source_stream=stopped_stream)

        self.assertNotIn("cam-1", self.detector._pending)

    def test_current_stream_is_accepted(self) -> None:
        current_stream = object()
        self.detector._streams["cam-1"] = current_stream

        self.detector.submit("cam-1", np.zeros((4, 4, 3), dtype=np.uint8),
                              source_stream=current_stream)

        self.assertIn("cam-1", self.detector._pending)
        self.assertEqual(self.detector._pending["cam-1"].frame_id, -1)

    def test_submit_without_source_stream_goes_to_registered(self) -> None:
        current_stream = object()
        self.detector._streams["cam-1"] = current_stream

        self.detector.submit("cam-1", np.zeros((4, 4, 3), dtype=np.uint8),
                              frame_id=42, captured_at=100.0)

        self.assertEqual(self.detector._pending["cam-1"].frame_id, 42)
        self.assertEqual(self.detector._pending["cam-1"].captured_at, 100.0)

    def test_submit_sets_pending_event(self) -> None:
        current_stream = object()
        self.detector._streams["cam-1"] = current_stream

        self.detector.submit("cam-1", np.zeros((4, 4, 3), dtype=np.uint8),
                              source_stream=current_stream)

        self.assertTrue(self.detector._pending_event.is_set())


class BEVMatrixTest(unittest.TestCase):
    @patch("app.modules.yolo.batch_detector.settings")
    def test_empty_calibration_returns_none(self, mock_settings):
        detector = _bare_detector()
        mock_settings.BEV_CALIBRATION = {}

        matrix = detector._build_bev_matrix("cam-01")

        self.assertIsNone(matrix)

    @patch("app.modules.yolo.batch_detector.settings")
    def test_valid_calibration_returns_matrix(self, mock_settings):
        detector = _bare_detector()
        mock_settings.BEV_CALIBRATION = {
            "cam-01": {
                "src": [[0, 0], [640, 0], [640, 480], [0, 480]],
                "dst": [[0, 0], [100, 0], [100, 50], [0, 50]],
            }
        }

        matrix = detector._build_bev_matrix("cam-01")

        self.assertIsNotNone(matrix)
        self.assertEqual(matrix.shape, (3, 3))

    @patch("app.modules.yolo.batch_detector.settings")
    def test_wrong_shape_calibration_returns_none(self, mock_settings):
        detector = _bare_detector()
        mock_settings.BEV_CALIBRATION = {
            "cam-01": {
                "src": [[0, 0], [1, 1]],
                "dst": [[0, 0], [1, 1]],
            }
        }

        matrix = detector._build_bev_matrix("cam-01")

        self.assertIsNone(matrix)


class PerformanceSnapshotTest(unittest.TestCase):
    def test_snapshot_reflects_current_state(self) -> None:
        detector = _bare_detector()

        snapshot = detector.performance_snapshot()

        self.assertIn("device", snapshot)
        self.assertEqual(snapshot["pending_cameras"], 0)
        self.assertEqual(snapshot["registered_streams"], 0)

    def test_snapshot_with_active_pending_and_inflight(self) -> None:
        detector = _bare_detector()
        from app.modules.yolo.batch_detector import PendingFrame
        detector._pending["cam-1"] = PendingFrame(np.zeros((4, 4, 3), dtype=np.uint8), 1, 100.0)
        detector._post_inflight.add("cam-2")
        detector._streams["cam-1"] = object()

        snapshot = detector.performance_snapshot()

        self.assertEqual(snapshot["pending_cameras"], 1)
        self.assertEqual(snapshot["registered_streams"], 1)
        self.assertEqual(snapshot["post_process_inflight"], 1)


class TrafficFlowTest(unittest.TestCase):
    def test_unknown_camera_returns_zeros(self) -> None:
        detector = _bare_detector()

        result = detector.get_traffic_flow("cam-nonexistent")

        self.assertEqual(result, {"entry_count": 0, "exit_count": 0, "flow_per_min": 0.0})

    def test_all_traffic_flow_aggregates(self) -> None:
        detector = _bare_detector()
        detector._processors["cam-1"] = SimpleNamespace(get_traffic_flow=lambda: {"entry_count": 5})
        detector._processors["cam-2"] = SimpleNamespace(get_traffic_flow=lambda: {"entry_count": 3})

        result = detector.get_all_traffic_flow()

        self.assertEqual(result, {"cam-1": {"entry_count": 5}, "cam-2": {"entry_count": 3}})


class UnregisterReleaseTest(unittest.TestCase):
    def test_unregister_releases_camera_state_when_not_inflight(self) -> None:
        detector = _bare_detector()
        detector._streams["cam-1"] = object()
        detector._pending["cam-1"] = SimpleNamespace()
        detector._processors["cam-1"] = object()

        detector.unregister_stream("cam-1")

        self.assertNotIn("cam-1", detector._streams)
        self.assertNotIn("cam-1", detector._pending)
        self.assertNotIn("cam-1", detector._processors)

    def test_unregister_defers_cleanup_when_inflight(self) -> None:
        detector = _bare_detector()
        detector._streams["cam-1"] = object()
        detector._processors["cam-1"] = object()
        detector._post_inflight.add("cam-1")

        detector.unregister_stream("cam-1")

        self.assertNotIn("cam-1", detector._streams)
        self.assertNotIn("cam-1", detector._pending)
        # 正在后处理中的摄像头状态延后清理
        self.assertIn("cam-1", detector._processors)


if __name__ == "__main__":
    unittest.main()
