import logging
import os
import threading
from typing import Optional

import numpy as np
import torch

from app.modules.lstm.feature_extractor import (
    MAX_WINDOWS_STORED,
    STEP_SEC,
    WINDOW_SEC,
    RawDetection,
    StandardScalerWrapper,
    TrafficFeatureExtractor,
)
from app.modules.lstm.model import TrafficRiskLSTM

logger = logging.getLogger(__name__)


class RiskPredictor:
    _instance: Optional["RiskPredictor"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "RiskPredictor":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def initialize(
        self,
        weights_path: str,
        scaler_path: Optional[str] = None,
        device: str = "cpu",
        camera_road_types: Optional[dict[str, int]] = None,
    ) -> None:
        if self._initialized:
            return

        self.device = torch.device(device)
        self.model = TrafficRiskLSTM(input_size=8, hidden_size=64, num_layers=2)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.scaler = StandardScalerWrapper(params_file=scaler_path)
        self.extractor = TrafficFeatureExtractor(scaler=self.scaler)
        self.camera_road_types = camera_road_types or {}

        self._camera_risks: dict[str, float] = {}
        self._risk_lock = threading.Lock()
        self._initialized = True

        logger.info(
            "RiskPredictor initialized: model=%s scaler=%s device=%s window=%ds/%ds",
            weights_path,
            scaler_path or "defaults",
            device,
            WINDOW_SEC,
            STEP_SEC,
        )

    def process_frame(
        self, camera_id: str, frame_id: int, timestamp_ms: int, detections: list[dict]
    ) -> Optional[float]:
        if not self._initialized:
            return None

        for det in detections:
            raw = self._detection_to_raw(det, frame_id, timestamp_ms)
            if raw:
                self.extractor.add_detection(camera_id, raw)

        if not self.extractor.should_aggregate(camera_id):
            return None

        road_type = self.camera_road_types.get(camera_id, 0)
        combined = self.extractor.aggregate_and_normalize(camera_id, road_type=road_type)
        if combined is None:
            return None

        lstm_input = self.extractor.get_lstm_input(camera_id)
        if lstm_input is None or lstm_input.shape[0] < MAX_WINDOWS_STORED:
            return None

        return self._infer(lstm_input, camera_id)

    def _detection_to_raw(self, det: dict, frame_id: int, timestamp_ms: int) -> Optional[RawDetection]:
        track_id = int(det.get("track_id", -1))
        if track_id < 0:
            return None
        return RawDetection(
            track_id=track_id,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            section_id=int(det.get("section_id", 0)),
            velocity=float(det.get("velocity", 0.0)),
            acceleration=float(det.get("acceleration", 0.0)),
            preceding_id=int(det.get("preceding_id", -1)),
            space_headway=float(det.get("space_headway", 0.0)),
        )

    def _infer(self, sequence: np.ndarray, camera_id: str) -> Optional[float]:
        tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.model(tensor)
        risk = float(prediction.item())

        with self._risk_lock:
            self._camera_risks[camera_id] = risk

        return risk

    def get_camera_risk(self, camera_id: str) -> Optional[float]:
        with self._risk_lock:
            return self._camera_risks.get(camera_id)

    def get_all_risks(self) -> dict[str, float]:
        with self._risk_lock:
            return dict(self._camera_risks)

    def get_detailed_risks(self) -> dict[str, dict]:
        with self._risk_lock:
            return {
                cam_id: {
                    "risk_score": score,
                    "window_sec": WINDOW_SEC,
                    "step_sec": STEP_SEC,
                }
                for cam_id, score in self._camera_risks.items()
            }


risk_predictor = RiskPredictor()
