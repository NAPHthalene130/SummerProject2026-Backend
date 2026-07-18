"""工单仓储层纯函数测试:行序列化、工单编号格式化、图片分割、等级转换等。"""

import unittest
from datetime import datetime

from app.models.work_order import WorkOrderStatus
from app.repository.work_order_repository import (
    WorkOrderRepository,
    format_work_order_code,
    parse_work_order_id,
    rank_to_level,
    split_images,
)


class ParseWorkOrderIdTest(unittest.TestCase):
    def test_digits_only(self) -> None:
        self.assertEqual(parse_work_order_id("144"), 144)
        self.assertEqual(parse_work_order_id("0007"), 7)

    def test_prefixed_ids(self) -> None:
        self.assertEqual(parse_work_order_id("WO-20260709-001"), 1)
        self.assertEqual(parse_work_order_id("WO-20260712-999"), 999)

    def test_invalid_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_work_order_id("not-a-number")


class FormatWorkOrderCodeTest(unittest.TestCase):
    def test_with_datetime(self) -> None:
        result = format_work_order_code(1, datetime(2026, 7, 12, 10, 30))

        self.assertEqual(result, "WO-20260712-001")

    def test_with_non_datetime_fallback(self) -> None:
        result = format_work_order_code(42, "not-a-datetime")

        self.assertEqual(result, "WO-20260709-042")


class RankToLevelTest(unittest.TestCase):
    def test_boundaries(self) -> None:
        self.assertEqual(rank_to_level(3), "high")
        self.assertEqual(rank_to_level(5), "high")
        self.assertEqual(rank_to_level(2), "medium")
        self.assertEqual(rank_to_level(1), "low")
        self.assertEqual(rank_to_level(0), "low")


class SplitImagesTest(unittest.TestCase):
    def test_empty_and_whitespace(self) -> None:
        self.assertEqual(split_images(""), [])
        self.assertEqual(split_images(None), [])
        self.assertEqual(split_images("  "), [])

    def test_single_and_multiple(self) -> None:
        self.assertEqual(split_images("/orderImg/a.jpg"), ["/orderImg/a.jpg"])
        self.assertEqual(
            split_images("/orderImg/a.jpg,/orderImg/b.png"),
            ["/orderImg/a.jpg", "/orderImg/b.png"],
        )

    def test_trims_whitespace_around_entries(self) -> None:
        self.assertEqual(
            split_images(" /a.jpg , /b.png "),
            ["/a.jpg", "/b.png"],
        )


class RowToResponseTest(unittest.TestCase):
    def _row(self, work_order_id=7, stage="pending", status=0, **overrides):
        values = {
            "work_order_id": work_order_id,
            "event_id": "evt-test-7",
            "camera_id": "cam-07",
            "camera_name": "测试摄像头",
            "segment_id": "seg-sz-07",
            "segment_name": "苏州街-海淀南路",
            "monitor_address": "海淀南路东口",
            "work_order_type": "车辆碰撞",
            "work_order_describe": "两车追尾",
            "work_order_img_url": "/orderImg/scene.jpg,/orderImg/scene2.jpg",
            "work_order_rank": 3,
            "work_order_time": datetime(2026, 7, 10, 12, 0, 0),
            "work_order_stage": stage,
            "work_order_status": status,
            "ai_suggestion": "先设警戒区域",
            "scene_info": "早高峰期间",
            "required_category": "traffic_police",
            "assignee": None,
            "assignee_user_id": None,
            "work_order_reply_msg": None,
            "work_order_reply_img_url": None,
            "work_order_reply_time": None,
            "work_order_reply_status": None,
            "requested_status": None,
            "review_message": None,
            "reviewed_at": None,
            "completed_at": None,
        }
        values.update(overrides)
        return values

    def test_basic_mapping(self) -> None:
        response = WorkOrderRepository._row_to_response(self._row())

        self.assertEqual(response.work_order_id, "WO-20260710-007")
        self.assertEqual(response.event_id, "evt-test-7")
        self.assertEqual(response.camera_id, "cam-07")
        self.assertEqual(response.camera_name, "测试摄像头")
        self.assertEqual(response.segment_id, "seg-sz-07")
        self.assertEqual(response.monitor_address, "海淀南路东口")
        self.assertEqual(response.accident_info, "车辆碰撞")
        self.assertEqual(response.event_time, "2026-07-10 12:00:00")
        self.assertEqual(response.event_level, "high")  # rank 3
        self.assertEqual(response.status, "pending")
        self.assertEqual(response.work_order_status, WorkOrderStatus.UNRESOLVED)
        self.assertIsNone(response.assignee)
        self.assertEqual(response.description, "两车追尾")
        self.assertEqual(response.ai_suggestion, "先设警戒区域")
        self.assertEqual(response.scene_images, ["/orderImg/scene.jpg", "/orderImg/scene2.jpg"])
        self.assertEqual(response.scene_info, "早高峰期间")
        self.assertIsNone(response.process_message)
        self.assertIsNone(response.completed_at)

    def test_resolved_status_wins_over_stale_stage(self) -> None:
        response = WorkOrderRepository._row_to_response(self._row(stage="pending", status=1))

        self.assertEqual(response.status, "completed")
        self.assertEqual(response.work_order_status, WorkOrderStatus.RESOLVED)

    def test_ignored_status_wins_over_stale_stage(self) -> None:
        response = WorkOrderRepository._row_to_response(self._row(stage="processing", status=2))

        self.assertEqual(response.status, "ignored")
        self.assertEqual(response.work_order_status, WorkOrderStatus.IGNORED)

    def test_unknown_stage_falls_back_to_unassigned(self) -> None:
        response = WorkOrderRepository._row_to_response(self._row(stage="unknown", status=0))

        self.assertEqual(response.status, "unassigned")

    def test_reply_img_splitting(self) -> None:
        response = WorkOrderRepository._row_to_response(
            self._row(work_order_reply_img_url="/r1.jpg,/r2.png")
        )

        self.assertEqual(response.process_images, ["/r1.jpg", "/r2.png"])

    def test_feedback_review_status_pending(self) -> None:
        response = WorkOrderRepository._row_to_response(
            self._row(work_order_reply_status=0, work_order_reply_msg="待审核")
        )

        self.assertEqual(response.feedback_review_status, "pending")

    def test_feedback_review_status_approved(self) -> None:
        response = WorkOrderRepository._row_to_response(
            self._row(work_order_reply_status=1, requested_status="completed",
                      work_order_reply_msg="已处理", work_order_reply_time=datetime(2026, 7, 10, 14, 0),
                      work_order_status=1)
        )

        self.assertEqual(response.feedback_review_status, "approved")
        self.assertEqual(response.process_message, "已处理")
        self.assertEqual(response.completed_at, "2026-07-10 14:00:00")

    def test_feedback_review_status_rejected(self) -> None:
        response = WorkOrderRepository._row_to_response(
            self._row(work_order_reply_status=2, review_message="材料不全")
        )

        self.assertEqual(response.feedback_review_status, "rejected")
        self.assertEqual(response.feedback_review_message, "材料不全")

    def test_no_reply_status_maps_to_none(self) -> None:
        response = WorkOrderRepository._row_to_response(self._row(work_order_reply_status=None))

        self.assertEqual(response.feedback_review_status, "none")


if __name__ == "__main__":
    unittest.main()
