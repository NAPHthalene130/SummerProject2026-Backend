import unittest
from datetime import datetime

from pydantic import ValidationError

from app.models.work_order import WorkOrderStatus, WorkOrderStatusUpdateRequest
from app.repository.work_order_repository import WorkOrderRepository


def make_row(stage: str, status: int) -> dict[str, object]:
    return {
        "work_order_id": 7,
        "event_id": "evt-test-7",
        "camera_id": "cam-07",
        "camera_name": "测试摄像头",
        "segment_id": "segment-07",
        "segment_name": "测试路段",
        "monitor_address": "测试地址",
        "work_order_type": "测试事件",
        "work_order_describe": "测试描述",
        "work_order_img_url": "",
        "work_order_rank": 2,
        "work_order_time": datetime(2026, 7, 10, 12, 0, 0),
        "work_order_stage": stage,
        "work_order_status": status,
        "ai_suggestion": "测试建议",
        "scene_info": "测试现场信息",
    }


class WorkOrderStatusTest(unittest.TestCase):
    def test_response_exposes_unresolved_numeric_status(self) -> None:
        order = WorkOrderRepository._row_to_response(make_row("processing", 0))

        self.assertEqual(order.status, "processing")
        self.assertEqual(order.work_order_status, WorkOrderStatus.UNRESOLVED)
        self.assertEqual(int(order.work_order_status), 0)

    def test_ignored_numeric_status_wins_over_stale_stage(self) -> None:
        order = WorkOrderRepository._row_to_response(make_row("processing", 2))

        self.assertEqual(order.status, "ignored")
        self.assertEqual(order.work_order_status, WorkOrderStatus.IGNORED)
        self.assertEqual(int(order.work_order_status), 2)

    def test_completed_numeric_status_wins_over_stale_stage(self) -> None:
        order = WorkOrderRepository._row_to_response(make_row("pending", 1))

        self.assertEqual(order.status, "completed")
        self.assertEqual(order.work_order_status, WorkOrderStatus.RESOLVED)

    def test_status_update_accepts_ignored_and_rejects_unknown_stage(self) -> None:
        request = WorkOrderStatusUpdateRequest(status="ignored")
        self.assertEqual(request.status, "ignored")

        with self.assertRaises(ValidationError):
            WorkOrderStatusUpdateRequest(status="false_alarm")


if __name__ == "__main__":
    unittest.main()
