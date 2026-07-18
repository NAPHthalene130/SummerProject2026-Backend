"""LSTM 风险预测测试:模型前向传播、特征提取、标准化与预测器完整管线。"""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from app.modules.lstm.feature_extractor import (
    MAX_WINDOWS_STORED,
    STEP_SEC,
    ACC_THRESHOLD,
    TTC_THRESHOLD,
    MacroFeatures,
    RawDetection,
    StandardScalerWrapper,
    TrafficFeatureExtractor,
)
from app.modules.lstm.model import TrafficRiskLSTM


class TrafficRiskLSTMTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = TrafficRiskLSTM(input_size=8, hidden_size=64, num_layers=2)
        self.model.eval()

    def test_forward_output_range_and_shape(self) -> None:
        batch_size = 2
        x = torch.randn(batch_size, 4, 8)

        with torch.no_grad():
            out = self.model(x)

        self.assertEqual(out.shape, (batch_size, 1))
        self.assertTrue((out >= 0).all() and (out <= 1).all())

    def test_forward_random_weights_produce_varied_outputs(self) -> None:
        x1 = torch.randn(1, 4, 8)
        x2 = torch.randn(1, 4, 8)

        with torch.no_grad():
            out1 = self.model(x1).item()
            out2 = self.model(x2).item()

        self.assertNotEqual(out1, out2)


class StandardScalerWrapperTest(unittest.TestCase):
    def test_defaults_are_identity_transform(self) -> None:
        scaler = StandardScalerWrapper(params_file=None)
        values = [15.0, 15.0, 20.0, 0.3, 50.0, 1000.0, 0.02, 0.5]

        result = scaler.transform(values)

        np.testing.assert_array_almost_equal(result, values, decimal=3)

    def test_loads_params_from_json_file(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"mean": [1, 2, 3, 4, 5, 6, 7, 8], "scale": [0.5, 1, 2, 1, 1, 1, 1, 1]}, f)
            tmp_path = f.name

        try:
            scaler = StandardScalerWrapper(params_file=tmp_path)

            self.assertTrue(scaler.is_loaded)
            result = scaler.transform([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

            np.testing.assert_array_almost_equal(result, [0.0] * 8, decimal=3)
        finally:
            os.unlink(tmp_path)

    def test_bad_json_file_falls_back_to_defaults(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write("not valid json")
            tmp_path = f.name

        try:
            scaler = StandardScalerWrapper(params_file=tmp_path)

            self.assertFalse(scaler.is_loaded)
            self.assertEqual(scaler.mean_[0], 15.0)
        finally:
            os.unlink(tmp_path)

    def test_zero_scale_is_guarded(self) -> None:
        scaler = StandardScalerWrapper(params_file=None)
        scaler.scale_ = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        scaler.mean_ = np.zeros(8, dtype=np.float64)

        result = scaler.transform([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        self.assertTrue(np.all(np.isfinite(result)))


class TrafficFeatureExtractorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scaler = StandardScalerWrapper(params_file=None)
        self.extractor = TrafficFeatureExtractor(scaler=self.scaler)
        self.camera = "cam-lstm-test"
        self.now_ms = int(time.time() * 1000)

    def _make_det(self, track_id=1, velocity=10.0, acceleration=0.0, preceding_id=-1, space_headway=50.0):
        self.now_ms += 1000
        return RawDetection(
            track_id=track_id,
            frame_id=0,
            timestamp_ms=self.now_ms,
            section_id=1,
            velocity=velocity,
            acceleration=acceleration,
            preceding_id=preceding_id,
            space_headway=space_headway,
        )

    def test_should_aggregate_throttles_by_step_seconds(self) -> None:
        self.assertTrue(self.extractor.should_aggregate(self.camera))
        self.assertFalse(self.extractor.should_aggregate(self.camera))

        # rewind and test again
        self.extractor._last_agg_time[self.camera] -= STEP_SEC + 1
        self.assertTrue(self.extractor.should_aggregate(self.camera))

    def test_aggregate_requires_at_least_three_detections(self) -> None:
        self.extractor.add_detection(self.camera, self._make_det(1))
        self.extractor.add_detection(self.camera, self._make_det(2))

        result = self.extractor.aggregate_and_normalize(self.camera, road_type=0)

        self.assertIsNone(result)

    def test_aggregate_returns_eight_float_list(self) -> None:
        for _ in range(3):
            self.extractor.add_detection(self.camera, self._make_det(1))

        result = self.extractor.aggregate_and_normalize(self.camera, road_type=1)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 8)
        self.assertEqual(result[7], 1.0)  # road_type embedded as float

    def test_window_queue_is_capped_at_max_windows(self) -> None:
        queue_map = self.extractor._window_queues
        for _ in range(5):
            for _ in range(3):
                self.extractor.add_detection(self.camera, self._make_det(1))
            self.extractor._last_agg_time[self.camera] = 0  # force should_aggregate
            self.extractor.aggregate_and_normalize(self.camera, road_type=0)

        q = queue_map[self.camera][1]
        self.assertEqual(len(q), MAX_WINDOWS_STORED)

    def test_get_lstm_input_returns_correct_shape(self) -> None:
        for _ in range(MAX_WINDOWS_STORED):
            for _ in range(3):
                self.extractor.add_detection(self.camera, self._make_det(1))
            self.extractor._last_agg_time[self.camera] = 0
            self.extractor.aggregate_and_normalize(self.camera, road_type=0)

        lstm_input = self.extractor.get_lstm_input(self.camera)

        self.assertIsNotNone(lstm_input)
        self.assertEqual(lstm_input.shape, (MAX_WINDOWS_STORED, 8))
        self.assertEqual(lstm_input.dtype, np.float32)

    def test_get_lstm_input_returns_none_with_fewer_than_max_windows(self) -> None:
        for _ in range(3):
            self.extractor.add_detection(self.camera, self._make_det(1))
        self.extractor._last_agg_time[self.camera] = 0
        self.extractor.aggregate_and_normalize(self.camera, road_type=0)

        self.assertIsNone(self.extractor.get_lstm_input(self.camera))

    def test_micro_metrics_compute_ttc_and_risk_flag(self) -> None:
        dets = [
            self._make_det(1, velocity=10.0, acceleration=-1.0),
            self._make_det(2, velocity=5.0, acceleration=0.0, preceding_id=1, space_headway=15.0),
        ]

        processed = self.extractor._calculate_micro_metrics(dets)

        follower = processed[1]
        # delta_v = 5 - 10 = -5, no ttc (delta_v <= 0)
        self.assertTrue(np.isnan(follower["ttc"]))
        self.assertTrue(np.isnan(follower["drac"]))
        self.assertEqual(follower["risk_flag"], 0)

    def test_ttc_below_threshold_triggers_risk_flag(self) -> None:
        dets = [
            self._make_det(2, velocity=20.0, acceleration=0.0),
            self._make_det(1, velocity=10.0, acceleration=0.0, preceding_id=2, space_headway=2.0),
        ]

        processed = self.extractor._calculate_micro_metrics(dets)
        follower = processed[1]
        # delta_v = 10 - 20 = -10, no ttc
        self.assertEqual(follower["risk_flag"], 0)

        # instead: rear faster than front → delta_v > 0 → ttc computed
        dets2 = [
            self._make_det(2, velocity=10.0, acceleration=0.0),
            self._make_det(1, velocity=30.0, acceleration=0.0, preceding_id=2, space_headway=5.0),
        ]
        processed2 = self.extractor._calculate_micro_metrics(dets2)
        follower2 = processed2[1]
        # delta_v = 30-10 = 20 > 0, ttc = 5/20 = 0.25 < 3 → risk_flag = 1
        self.assertEqual(follower2["risk_flag"], 1)
        self.assertAlmostEqual(follower2["ttc"], 0.25)

    def test_aggregate_window_returns_none_with_less_than_two_detections(self) -> None:
        macro = self.extractor._aggregate_window([], 1, 0)

        self.assertIsNone(macro)

    def test_volume_uses_unique_track_count(self) -> None:
        processed = [
            {"track_id": 1, "velocity": 10.0, "acceleration": 0.0, "space_headway": 50.0,
             "drac": float("nan"), "risk_flag": 0},
            {"track_id": 1, "velocity": 12.0, "acceleration": 0.0, "space_headway": 55.0,
             "drac": float("nan"), "risk_flag": 0},
            {"track_id": 2, "velocity": 8.0, "acceleration": 0.0, "space_headway": 40.0,
             "drac": float("nan"), "risk_flag": 0},
        ]

        macro = self.extractor._aggregate_window(processed, 1, 0)

        self.assertIsNotNone(macro)
        self.assertEqual(macro.volume, 2.0)

    def test_combine_sections_averages_speed_var_and_headway(self) -> None:
        macros = [
            MacroFeatures(section_id=1, time_window=0, volume=1.0, avg_speed=30.0,
                          speed_var=4.0, density=0.02, avg_space_headway=50.0,
                          space_headway_var=100.0, deceleration_freq=0.01,
                          avg_drac=2.0, total_conflicts=1.0, road_type=0),
            MacroFeatures(section_id=2, time_window=0, volume=3.0, avg_speed=60.0,
                          speed_var=16.0, density=0.06, avg_space_headway=50.0,
                          space_headway_var=400.0, deceleration_freq=0.03,
                          avg_drac=4.0, total_conflicts=2.0, road_type=0),
        ]

        combined = self.extractor._combine_sections(macros, 0)

        self.assertEqual(len(combined), 8)
        # volume-weighted avg speed
        self.assertAlmostEqual(combined[1], (30.0 * 1 + 60.0 * 3) / 4)
        # avg_speed_var
        self.assertAlmostEqual(combined[2], (4.0 + 16.0) / 2)

    def test_deceleration_frequency_uses_acceleration_threshold(self) -> None:
        processed = [
            {"track_id": 1, "velocity": 10.0, "acceleration": -10.0, "space_headway": 50.0,
             "drac": float("nan"), "risk_flag": 0},
            {"track_id": 2, "velocity": 12.0, "acceleration": -5.0, "space_headway": 55.0,
             "drac": float("nan"), "risk_flag": 0},
            {"track_id": 3, "velocity": 8.0, "acceleration": 1.0, "space_headway": 40.0,
             "drac": float("nan"), "risk_flag": 0},
        ]

        macro = self.extractor._aggregate_window(processed, 1, 0)

        # only acceleration_1 = -10.0 < ACC_THRESHOLD (-9.8)
        self.assertAlmostEqual(macro.deceleration_freq, 1.0 / 3)

    def test_detection_to_raw_validation(self) -> None:
        from app.modules.lstm.predictor import RiskPredictor

        predictor = object.__new__(RiskPredictor)

        self.assertIsNone(predictor._detection_to_raw({"track_id": -1}, 0, 0))
        self.assertIsNone(predictor._detection_to_raw({"track_id": -5}, 0, 0))

        raw = predictor._detection_to_raw(
            {"track_id": 7, "section_id": 2, "velocity": 15.5, "acceleration": -2.0,
             "preceding_id": 3, "space_headway": 40.0},
            frame_id=100,
            timestamp_ms=5000,
        )

        self.assertEqual(raw.track_id, 7)
        self.assertEqual(raw.section_id, 2)
        self.assertEqual(raw.velocity, 15.5)
        self.assertEqual(raw.acceleration, -2.0)
        self.assertEqual(raw.preceding_id, 3)
        self.assertEqual(raw.space_headway, 40.0)

    def test_uninitialized_predictor_process_frame_returns_none(self) -> None:
        from app.modules.lstm.predictor import RiskPredictor

        # reset singleton
        RiskPredictor._instance = None
        predictor = RiskPredictor()

        try:
            self.assertIsNone(predictor.process_frame("cam-1", 0, 0, []))
        finally:
            RiskPredictor._instance = None

    def test_fully_initialized_predictor_processes_frame_pipeline(self) -> None:
        from app.modules.lstm.predictor import RiskPredictor

        # build model and save weights
        model = TrafficRiskLSTM(input_size=8, hidden_size=64, num_layers=2)
        model.eval()
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            torch.save(model.state_dict(), f)
            weights_path = f.name

        try:
            RiskPredictor._instance = None
            predictor = RiskPredictor()
            predictor.initialize(
                weights_path=weights_path,
                scaler_path=None,
                device="cpu",
                camera_road_types={"cam-infer": 1},
            )

            # seeded detections: 3 per call × 4 windows = 12 detections
            timestamp_ms = int(time.time() * 1000) - 10000
            for _ in range(MAX_WINDOWS_STORED):
                for track_id in range(3):
                    timestamp_ms += 1000
                    det = {
                        "track_id": track_id,
                        "frame_id": 0,
                        "timestamp_ms": timestamp_ms,
                        "section_id": 1,
                        "velocity": 10.0 + track_id * 2,
                        "acceleration": 0.0,
                        "preceding_id": -1,
                        "space_headway": 50.0,
                    }
                    predictor.extractor.add_detection("cam-infer",
                        RawDetection(track_id=det["track_id"], frame_id=0, timestamp_ms=det["timestamp_ms"],
                                     section_id=1, velocity=det["velocity"], acceleration=det["acceleration"],
                                     preceding_id=-1, space_headway=det["space_headway"]))
                predictor.extractor._last_agg_time["cam-infer"] = 0
                predictor.extractor.aggregate_and_normalize("cam-infer", road_type=1)

            # now process_frame should have window queue full
            risk = predictor.process_frame("cam-infer", 0, timestamp_ms + 1000,
                [{"track_id": 1, "frame_id": 0, "timestamp_ms": timestamp_ms + 1000,
                  "section_id": 1, "velocity": 12.0, "acceleration": 0.0,
                  "preceding_id": -1, "space_headway": 50.0}])

            self.assertIsNotNone(risk)
            self.assertTrue(0.0 <= risk <= 1.0)
            self.assertEqual(predictor.get_camera_risk("cam-infer"), risk)
            detailed = predictor.get_detailed_risks()
            self.assertIn("cam-infer", detailed)
            self.assertIn("risk_score", detailed["cam-infer"])
            self.assertIn("window_sec", detailed["cam-infer"])
        finally:
            RiskPredictor._instance = None
            os.unlink(weights_path)


if __name__ == "__main__":
    unittest.main()
