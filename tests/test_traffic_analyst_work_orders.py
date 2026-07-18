import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.modules.agent.traffic_analyst import TrafficAnalyst
from app.modules.camera_data import CONSECUTIVE_NORMAL_THRESHOLD, CameraDataStore


ANOMALY_RESPONSE = (
    '{"incident_detected": true, "confidence": 0.96, "incident_type": "车辆碰撞", '
    '"description": "检测到两车碰撞"}'
)
CONGESTION_RESPONSE = (
    '{"incident_detected": true, "confidence": 0.92, "incident_type": "交通拥堵", '
    '"description": "检测到车辆持续拥堵"}'
)
NORMAL_RESPONSE = (
    '{"incident_detected": false, "confidence": 0.05, "incident_type": "正常", '
    '"description": "道路通行正常"}'
)
CONFIRMED_VERIFICATION = '{"confirmed": true, "reason": "二次校验确认事故成立"}'
REJECTED_VERIFICATION = '{"confirmed": false, "reason": "画面中未观察到事故特征"}'


class _FakeStream:
    def __init__(self):
        self._frame = np.zeros((4, 4, 3), dtype=np.uint8)

    def get_raw_frame(self) -> tuple[np.ndarray, int]:
        return self._frame.copy(), 1


class TrafficAnalystWorkOrderTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.analyst = TrafficAnalyst()
        self.analyst._frame_buffer = {}
        self.analyst._next_analysis = {}
        self.stream = _FakeStream()
        self.store = CameraDataStore()
        self._reset_store()

    def tearDown(self) -> None:
        self._reset_store()

    def _reset_store(self) -> None:
        self.store._incident_data.clear()
        self.store._active_incidents.clear()
        self.store._consecutive_normal_count.clear()

    async def test_first_anomaly_creates_one_order_and_continuing_anomaly_does_not_duplicate(self) -> None:
        work_order = SimpleNamespace(work_order_id="WO-TEST-001")

        with (
            patch.object(self.analyst, "_call_multimodal", return_value=ANOMALY_RESPONSE),
            patch.object(self.analyst, "_call_verification", return_value=CONFIRMED_VERIFICATION),
            patch.object(self.analyst, "_save_incident_frame", return_value="/orderImg/test.jpg"),
            patch(
                "app.modules.agent.traffic_analyst.WorkOrderRepository.create_work_order",
                return_value=work_order,
            ) as create_work_order,
        ):
            await self.analyst._analyze_camera(self.stream, "cam-01")
            await self.analyst._analyze_camera(self.stream, "cam-01")

        self.assertTrue(self.store.is_incident_active("cam-01"))
        create_work_order.assert_called_once()

    async def test_different_incident_type_creates_a_new_higher_priority_order(self) -> None:
        work_order = SimpleNamespace(work_order_id="WO-TEST-PRIORITY")
        responses = [CONGESTION_RESPONSE] * 6 + [ANOMALY_RESPONSE]

        with (
            patch.object(self.analyst, "_call_multimodal", side_effect=responses),
            patch.object(self.analyst, "_call_verification", return_value=CONFIRMED_VERIFICATION),
            patch.object(self.analyst, "_save_incident_frame", return_value="/orderImg/test.jpg"),
            patch(
                "app.modules.agent.traffic_analyst.WorkOrderRepository.create_work_order",
                return_value=work_order,
            ) as create_work_order,
        ):
            for _ in responses:
                await self.analyst._analyze_camera(self.stream, "cam-01")

        self.assertEqual(create_work_order.call_count, 2)
        self.assertEqual(
            [call.kwargs["incident_type"] for call in create_work_order.call_args_list],
            ["交通拥堵", "车辆碰撞"],
        )
        self.assertEqual(
            [call.kwargs["rank"] for call in create_work_order.call_args_list],
            [2, 3],
        )
        self.assertEqual(
            self.store.get_active_incident_types("cam-01"),
            {"交通拥堵", "车辆碰撞"},
        )

    async def test_other_active_type_does_not_clear_or_duplicate_existing_event(self) -> None:
        work_order = SimpleNamespace(work_order_id="WO-TEST-LIFECYCLE")
        responses = (
            [CONGESTION_RESPONSE]
            + [ANOMALY_RESPONSE] * CONSECUTIVE_NORMAL_THRESHOLD
            + [CONGESTION_RESPONSE]
        )

        with (
            patch.object(self.analyst, "_call_multimodal", side_effect=responses),
            patch.object(self.analyst, "_call_verification", return_value=CONFIRMED_VERIFICATION),
            patch.object(self.analyst, "_save_incident_frame", return_value="/orderImg/test.jpg"),
            patch(
                "app.modules.agent.traffic_analyst.WorkOrderRepository.create_work_order",
                return_value=work_order,
            ) as create_work_order,
        ):
            for _ in responses:
                await self.analyst._analyze_camera(self.stream, "cam-01")

        self.assertEqual(create_work_order.call_count, 2)
        self.assertEqual(
            [call.kwargs["incident_type"] for call in create_work_order.call_args_list],
            ["交通拥堵", "车辆碰撞"],
        )
        self.assertEqual(
            self.store.get_active_incident_types("cam-01"),
            {"交通拥堵", "车辆碰撞"},
        )

    async def test_ten_normal_results_clear_all_types_before_they_can_reappear(self) -> None:
        work_order = SimpleNamespace(work_order_id="WO-TEST-RESET")
        responses = (
            [CONGESTION_RESPONSE, ANOMALY_RESPONSE]
            + [NORMAL_RESPONSE] * CONSECUTIVE_NORMAL_THRESHOLD
            + [CONGESTION_RESPONSE, ANOMALY_RESPONSE]
        )

        with (
            patch.object(self.analyst, "_call_multimodal", side_effect=responses),
            patch.object(self.analyst, "_call_verification", return_value=CONFIRMED_VERIFICATION),
            patch.object(self.analyst, "_save_incident_frame", return_value="/orderImg/test.jpg"),
            patch(
                "app.modules.agent.traffic_analyst.WorkOrderRepository.create_work_order",
                return_value=work_order,
            ) as create_work_order,
        ):
            for _ in responses:
                await self.analyst._analyze_camera(self.stream, "cam-01")

        self.assertEqual(create_work_order.call_count, 4)
        self.assertEqual(
            [call.kwargs["incident_type"] for call in create_work_order.call_args_list],
            ["交通拥堵", "车辆碰撞", "交通拥堵", "车辆碰撞"],
        )
        self.assertEqual(
            self.store.get_active_incident_types("cam-01"),
            {"交通拥堵", "车辆碰撞"},
        )

    async def test_failed_database_insert_is_retried_on_next_anomaly(self) -> None:
        work_order = SimpleNamespace(work_order_id="WO-TEST-002")

        with (
            patch.object(self.analyst, "_call_multimodal", return_value=ANOMALY_RESPONSE),
            patch.object(self.analyst, "_call_verification", return_value=CONFIRMED_VERIFICATION),
            patch.object(self.analyst, "_save_incident_frame", return_value="/orderImg/test.jpg"),
            patch(
                "app.modules.agent.traffic_analyst.WorkOrderRepository.create_work_order",
                side_effect=[None, work_order],
            ) as create_work_order,
        ):
            await self.analyst._analyze_camera(self.stream, "cam-01")
            self.assertFalse(self.store.is_incident_active("cam-01"))

            await self.analyst._analyze_camera(self.stream, "cam-01")

        self.assertTrue(self.store.is_incident_active("cam-01"))
        self.assertEqual(create_work_order.call_count, 2)

    async def test_new_order_is_created_after_incident_clears_and_reappears(self) -> None:
        work_order = SimpleNamespace(work_order_id="WO-TEST-003")
        responses = [ANOMALY_RESPONSE] + [NORMAL_RESPONSE] * CONSECUTIVE_NORMAL_THRESHOLD + [ANOMALY_RESPONSE]

        with (
            patch.object(self.analyst, "_call_multimodal", side_effect=responses),
            patch.object(self.analyst, "_call_verification", return_value=CONFIRMED_VERIFICATION),
            patch.object(self.analyst, "_save_incident_frame", return_value="/orderImg/test.jpg"),
            patch(
                "app.modules.agent.traffic_analyst.WorkOrderRepository.create_work_order",
                return_value=work_order,
            ) as create_work_order,
        ):
            await self.analyst._analyze_camera(self.stream, "cam-01")
            for _ in range(CONSECUTIVE_NORMAL_THRESHOLD - 1):
                await self.analyst._analyze_camera(self.stream, "cam-01")
            self.assertTrue(self.store.is_incident_active("cam-01"))

            await self.analyst._analyze_camera(self.stream, "cam-01")
            self.assertFalse(self.store.is_incident_active("cam-01"))

            await self.analyst._analyze_camera(self.stream, "cam-01")

        self.assertTrue(self.store.is_incident_active("cam-01"))
        self.assertEqual(create_work_order.call_count, 2)

    def test_string_false_does_not_create_an_anomaly(self) -> None:
        incident = self.analyst._parse_response(
            "cam-01",
            '{"incident_detected": "false", "incident_type": "车辆碰撞", "description": ""}',
        )

        self.assertFalse(incident.incident_detected)
        self.assertEqual(incident.incident_type, "正常")

    def test_detected_anomaly_with_normal_type_is_normalized(self) -> None:
        incident = self.analyst._parse_response(
            "cam-01",
            '{"incident_detected": true, "confidence": 0.95, "incident_type": "正常", "description": "异常"}',
        )

        self.assertTrue(incident.incident_detected)
        self.assertEqual(incident.incident_type, "其他事故")

    def test_low_confidence_anomaly_is_downgraded_to_normal(self) -> None:
        incident = self.analyst._parse_response(
            "cam-01",
            '{"incident_detected": true, "confidence": 0.84, "incident_type": "车辆碰撞", "description": "疑似碰撞"}',
        )

        self.assertFalse(incident.incident_detected)
        self.assertEqual(incident.incident_type, "正常")
        self.assertEqual(incident.confidence, 0.84)

    async def test_low_confidence_anomaly_does_not_create_work_order(self) -> None:
        low_confidence_response = (
            '{"incident_detected": true, "confidence": 0.84, '
            '"incident_type": "车辆碰撞", "description": "疑似碰撞"}'
        )

        with (
            patch.object(self.analyst, "_call_multimodal", return_value=low_confidence_response),
            patch.object(self.analyst, "_save_incident_frame") as save_incident_frame,
            patch(
                "app.modules.agent.traffic_analyst.WorkOrderRepository.create_work_order",
            ) as create_work_order,
        ):
            await self.analyst._analyze_camera(self.stream, "cam-01")

        save_incident_frame.assert_not_called()
        create_work_order.assert_not_called()
        self.assertFalse(self.store.is_incident_active("cam-01"))

    def test_missing_confidence_cannot_create_an_anomaly(self) -> None:
        incident = self.analyst._parse_response(
            "cam-01",
            '{"incident_detected": true, "incident_type": "车辆碰撞", "description": "碰撞"}',
        )

        self.assertFalse(incident.incident_detected)
        self.assertEqual(incident.incident_type, "正常")
        self.assertEqual(incident.confidence, 0.0)

    def test_confidence_threshold_is_inclusive(self) -> None:
        incident = self.analyst._parse_response(
            "cam-01",
            '{"incident_detected": true, "confidence": 0.85, "incident_type": "车辆碰撞", "description": "明确碰撞"}',
        )

        self.assertTrue(incident.incident_detected)
        self.assertEqual(incident.incident_type, "车辆碰撞")

    async def test_rejected_verification_creates_no_work_order_and_does_not_activate(self) -> None:
        with (
            patch.object(self.analyst, "_call_multimodal", return_value=ANOMALY_RESPONSE),
            patch.object(self.analyst, "_call_verification", return_value=REJECTED_VERIFICATION),
            patch.object(self.analyst, "_save_incident_frame") as save_incident_frame,
            patch(
                "app.modules.agent.traffic_analyst.WorkOrderRepository.create_work_order",
            ) as create_work_order,
        ):
            await self.analyst._analyze_camera(self.stream, "cam-01")

        save_incident_frame.assert_not_called()
        create_work_order.assert_not_called()
        self.assertFalse(self.store.is_incident_active("cam-01"))
        # 误报被驳回后清空帧缓冲,避免旧帧污染后续检测
        self.assertNotIn("cam-01", self.analyst._frame_buffer)

    async def test_rejected_incident_is_reverified_and_confirmed_on_next_anomaly(self) -> None:
        work_order = SimpleNamespace(work_order_id="WO-TEST-VERIFY")

        with (
            patch.object(self.analyst, "_call_multimodal", return_value=ANOMALY_RESPONSE),
            patch.object(
                self.analyst,
                "_call_verification",
                side_effect=[REJECTED_VERIFICATION, CONFIRMED_VERIFICATION],
            ) as call_verification,
            patch.object(self.analyst, "_save_incident_frame", return_value="/orderImg/test.jpg"),
            patch(
                "app.modules.agent.traffic_analyst.WorkOrderRepository.create_work_order",
                return_value=work_order,
            ) as create_work_order,
        ):
            await self.analyst._analyze_camera(self.stream, "cam-01")
            self.assertFalse(self.store.is_incident_active("cam-01"))

            await self.analyst._analyze_camera(self.stream, "cam-01")

        self.assertEqual(call_verification.call_count, 2)
        create_work_order.assert_called_once()
        self.assertTrue(self.store.is_incident_active("cam-01"))

    async def test_verification_exception_is_treated_as_rejection(self) -> None:
        with (
            patch.object(self.analyst, "_call_multimodal", return_value=ANOMALY_RESPONSE),
            patch.object(self.analyst, "_call_verification", side_effect=RuntimeError("vlm down")),
            patch(
                "app.modules.agent.traffic_analyst.WorkOrderRepository.create_work_order",
            ) as create_work_order,
        ):
            await self.analyst._analyze_camera(self.stream, "cam-01")

        create_work_order.assert_not_called()
        self.assertFalse(self.store.is_incident_active("cam-01"))

    async def test_active_incident_skips_verification_and_work_order(self) -> None:
        work_order = SimpleNamespace(work_order_id="WO-TEST-ACTIVE")

        with (
            patch.object(self.analyst, "_call_multimodal", return_value=ANOMALY_RESPONSE),
            patch.object(self.analyst, "_call_verification", return_value=CONFIRMED_VERIFICATION) as call_verification,
            patch.object(self.analyst, "_save_incident_frame", return_value="/orderImg/test.jpg"),
            patch(
                "app.modules.agent.traffic_analyst.WorkOrderRepository.create_work_order",
                return_value=work_order,
            ),
        ):
            await self.analyst._analyze_camera(self.stream, "cam-01")
            await self.analyst._analyze_camera(self.stream, "cam-01")
            await self.analyst._analyze_camera(self.stream, "cam-01")

        # 事故持续期间不再重复二次校验,也不再重复建单
        call_verification.assert_called_once()

    def test_parse_verification_response_variants(self) -> None:
        confirmed, _ = self.analyst._parse_verification_response(
            "cam-01", '```json\n{"confirmed": true, "reason": "确认"}\n```'
        )
        self.assertTrue(confirmed)

        confirmed, reason = self.analyst._parse_verification_response(
            "cam-01", '{"confirmed": "true", "reason": ""}'
        )
        self.assertTrue(confirmed)
        self.assertEqual(reason, "校验模型未提供裁定理由")

        confirmed, _ = self.analyst._parse_verification_response("cam-01", "not-json")
        self.assertFalse(confirmed)

        confirmed, reason = self.analyst._parse_verification_response("cam-01", '["confirmed"]')
        self.assertFalse(confirmed)
        self.assertIn("非JSON对象", reason)

    def test_scheduler_delay_is_bounded(self) -> None:
        self.analyst._next_analysis = {}
        self.assertEqual(self.analyst._scheduler_delay(now=0.0), 0.5)

        self.analyst._next_analysis = {"cam-01": 10.0}
        self.assertAlmostEqual(self.analyst._scheduler_delay(now=9.8), 0.2)
        self.assertEqual(self.analyst._scheduler_delay(now=0.0), 0.5)
        self.assertEqual(self.analyst._scheduler_delay(now=10.0), 0.0)

    async def test_frame_buffer_is_capped_at_buffer_size(self) -> None:
        with patch.object(self.analyst, "_call_multimodal", return_value=NORMAL_RESPONSE):
            for _ in range(8):
                await self.analyst._analyze_camera(self.stream, "cam-01")

        self.assertEqual(len(self.analyst._frame_buffer["cam-01"]), 5)

    async def test_missing_frame_returns_before_model_call(self) -> None:
        class EmptyStream:
            @staticmethod
            def get_raw_frame() -> tuple[None, int]:
                return None, -1

        with patch.object(self.analyst, "_call_multimodal") as call_multimodal:
            await self.analyst._analyze_camera(EmptyStream(), "cam-01")

        call_multimodal.assert_not_called()
        self.assertNotIn("cam-01", self.analyst._frame_buffer)


if __name__ == "__main__":
    unittest.main()
