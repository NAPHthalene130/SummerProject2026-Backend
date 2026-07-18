"""风险预测 API 测试:摄像头风险映射、单路查询与路段预测请求转发。"""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.api.v1.risks import (
    RiskPredictionRequest,
    get_all_risks,
    get_camera_risk,
    predict_road_risks,
)


class GetAllRisksApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_cameras_are_mapped_to_frontend_ids_in_config_order(self) -> None:
        cameras = [
            SimpleNamespace(id="cam-01"),
            SimpleNamespace(id="cam-02"),
            SimpleNamespace(id="cam-03"),
        ]
        predictor = SimpleNamespace(
            get_all_risks=lambda: {"cam-01": 0.9, "cam-03": 0.4},
            get_detailed_risks=lambda: {"cam-01": {"risk_score": 0.9}},
        )
        with (
            patch("app.api.v1.risks.risk_predictor", predictor),
            patch("app.api.v1.risks.CameraManager") as manager_cls,
        ):
            manager_cls.return_value.get_all.return_value = cameras
            result = await get_all_risks()

        # cam-02 无数据时不输出;编号从 C01 开始按配置顺序递增
        self.assertEqual(result["camera_risks"], {"C01": 0.9, "C03": 0.4})
        self.assertEqual(result["detailed"], {"cam-01": {"risk_score": 0.9}})


class GetCameraRiskApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_returns_message_without_data(self) -> None:
        predictor = SimpleNamespace(get_camera_risk=lambda cam_id: None)
        with patch("app.api.v1.risks.risk_predictor", predictor):
            result = await get_camera_risk("cam-01")

        self.assertEqual(result["risk_score"], None)
        self.assertEqual(result["message"], "no data yet")

    async def test_returns_score(self) -> None:
        predictor = SimpleNamespace(get_camera_risk=lambda cam_id: 0.73)
        with patch("app.api.v1.risks.risk_predictor", predictor):
            result = await get_camera_risk("cam-01")

        self.assertEqual(result, {"camera_id": "cam-01", "risk_score": 0.73})


class PredictRoadRisksApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_forwards_segments_and_selection_to_service(self) -> None:
        service = SimpleNamespace(predict=AsyncMock(return_value={"results": []}))
        request = RiskPredictionRequest(
            segments=[
                {
                    "segment_id": "seg-1",
                    "latitude": 39.97,
                    "longitude": 116.31,
                    "road_type": "primary",
                }
            ],
            selected_segment_id="seg-1",
        )

        with patch("app.api.v1.risks.live_risk_prediction_service", service):
            result = await predict_road_risks(request)

        self.assertEqual(result, {"results": []})
        service.predict.assert_awaited_once()
        segments_arg = service.predict.await_args.args[0]
        self.assertEqual(segments_arg[0]["segment_id"], "seg-1")
        # 默认值由模型填充
        self.assertEqual(segments_arg[0]["lane_count"], 2)
        self.assertEqual(segments_arg[0]["speed_limit"], 40)
        self.assertEqual(service.predict.await_args.kwargs["selected_segment_id"], "seg-1")

    def test_request_model_validates_coordinate_ranges(self) -> None:
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            RiskPredictionRequest(
                segments=[{"segment_id": "s", "latitude": 123.0, "longitude": 0.0}]
            )


if __name__ == "__main__":
    unittest.main()
