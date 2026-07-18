"""工单/人员 API 层测试:仓储层全部 mock,校验路由函数的分组、排序与错误映射。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from app.api.v1.work_orders import (
    dispatch_work_order,
    get_work_order_detail,
    list_staff,
    list_unprocessed,
    list_work_orders,
    review_mobile_feedback,
    submit_mobile_feedback,
    update_work_order_status,
)
from app.models.work_order import (
    FeedbackReviewRequest,
    MobileFeedbackRequest,
    WorkOrderDispatchRequest,
    WorkOrderItemResponse,
    WorkOrderStatus,
    WorkOrderStatusUpdateRequest,
)


def _order(**overrides):
    values = {
        "work_order_id": "WO-20260712-001",
        "event_id": "EV-1",
        "camera_id": "cam-01",
        "camera_name": "摄像头01",
        "segment_id": "seg-01",
        "segment_name": "路段01",
        "monitor_address": "测试路口",
        "accident_info": "车辆碰撞",
        "event_time": "2026-07-12 10:00:00",
        "event_level": "high",
        "status": "pending",
        "work_order_status": WorkOrderStatus.UNRESOLVED,
        "description": "测试描述",
        "ai_suggestion": "测试建议",
        "scene_images": [],
        "scene_info": "测试现场",
    }
    values.update(overrides)
    return WorkOrderItemResponse(**values)


class UnprocessedGroupingTest(unittest.IsolatedAsyncioTestCase):
    async def test_groups_by_camera_excludes_terminal_and_sorts(self) -> None:
        orders = [
            _order(work_order_id="WO-1", camera_id="cam-02", event_time="2026-07-12 10:00:00"),
            _order(work_order_id="WO-2", camera_id="cam-01", event_time="2026-07-12 09:00:00"),
            _order(work_order_id="WO-3", camera_id="cam-01", event_time="2026-07-12 11:00:00"),
            _order(work_order_id="WO-4", camera_id="cam-03", status="completed"),
            _order(work_order_id="WO-5", camera_id="cam-03", status="ignored"),
        ]
        with patch(
            "app.api.v1.work_orders.WorkOrderRepository.list_work_orders", return_value=orders
        ):
            response = await list_unprocessed()

        self.assertEqual([cam.camera_id for cam in response.cameras], ["cam-01", "cam-02"])
        cam01 = response.cameras[0]
        self.assertEqual(cam01.camera_name, "摄像头01")
        # 事件按时间倒序
        self.assertEqual(
            [event.work_order_id for event in cam01.events],
            ["WO-3", "WO-2"],
        )

    async def test_missing_camera_fields_fall_back_to_defaults(self) -> None:
        orders = [_order(camera_id="", camera_name="")]
        with patch(
            "app.api.v1.work_orders.WorkOrderRepository.list_work_orders", return_value=orders
        ):
            response = await list_unprocessed()

        self.assertEqual(response.cameras[0].camera_id, "unknown")
        self.assertEqual(response.cameras[0].camera_name, "未知摄像头")


class WorkOrderDetailApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_detail_returns_order(self) -> None:
        order = _order()
        with patch(
            "app.api.v1.work_orders.WorkOrderRepository.get_work_order", return_value=order
        ):
            result = await get_work_order_detail("WO-20260712-001")

        self.assertIs(result, order)

    async def test_detail_missing_raises_404(self) -> None:
        with patch("app.api.v1.work_orders.WorkOrderRepository.get_work_order", return_value=None):
            with self.assertRaises(HTTPException) as caught:
                await get_work_order_detail("WO-x")

        self.assertEqual(caught.exception.status_code, 404)


class ListWorkOrdersApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_forwards_user_id_filter(self) -> None:
        with patch(
            "app.api.v1.work_orders.WorkOrderRepository.list_work_orders", return_value=[]
        ) as list_orders:
            await list_work_orders(user_id=3)
            await list_work_orders()

        self.assertEqual(list_orders.call_args_list[0].args[0], 3)
        self.assertEqual(list_orders.call_args_list[1].args[0], None)


class DispatchApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_success_returns_order(self) -> None:
        order = _order(status="pending")
        with patch(
            "app.api.v1.work_orders.WorkOrderRepository.dispatch_work_order", return_value=order
        ):
            result = await dispatch_work_order("WO-1", WorkOrderDispatchRequest(user_id=2))

        self.assertIs(result, order)

    async def test_missing_order_raises_404(self) -> None:
        with patch(
            "app.api.v1.work_orders.WorkOrderRepository.dispatch_work_order", return_value=None
        ):
            with self.assertRaises(HTTPException) as caught:
                await dispatch_work_order("WO-x", WorkOrderDispatchRequest(user_id=2))

        self.assertEqual(caught.exception.status_code, 404)

    async def test_stage_conflict_raises_409(self) -> None:
        with patch(
            "app.api.v1.work_orders.WorkOrderRepository.dispatch_work_order",
            side_effect=ValueError("仅未派发(unassigned)工单可派发"),
        ):
            with self.assertRaises(HTTPException) as caught:
                await dispatch_work_order("WO-1", WorkOrderDispatchRequest(user_id=2))

        self.assertEqual(caught.exception.status_code, 409)
        self.assertIn("仅未派发", caught.exception.detail)


class StatusUpdateApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_success_forwards_all_fields(self) -> None:
        order = _order(status="completed")
        request = WorkOrderStatusUpdateRequest(
            status="completed", process_message="已清障", process_image_url="/orderImg/r.jpg"
        )
        with patch(
            "app.api.v1.work_orders.WorkOrderRepository.update_work_order_status", return_value=order
        ) as update:
            result = await update_work_order_status("WO-1", request)

        self.assertIs(result, order)
        update.assert_called_once_with(
            work_order_id="WO-1",
            status="completed",
            process_message="已清障",
            process_image_url="/orderImg/r.jpg",
        )

    async def test_missing_order_raises_404(self) -> None:
        with patch(
            "app.api.v1.work_orders.WorkOrderRepository.update_work_order_status", return_value=None
        ):
            with self.assertRaises(HTTPException) as caught:
                await update_work_order_status("WO-x", WorkOrderStatusUpdateRequest(status="ignored"))

        self.assertEqual(caught.exception.status_code, 404)


class MobileFeedbackApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_success(self) -> None:
        order = _order(status="processing")
        request = MobileFeedbackRequest(user_id=5, status="completed", process_message="处置完成")
        with patch(
            "app.api.v1.work_orders.WorkOrderRepository.submit_mobile_feedback", return_value=order
        ):
            result = await submit_mobile_feedback("WO-1", request)

        self.assertIs(result, order)

    async def test_missing_raises_404(self) -> None:
        request = MobileFeedbackRequest(user_id=5, status="completed", process_message="x")
        with patch(
            "app.api.v1.work_orders.WorkOrderRepository.submit_mobile_feedback", return_value=None
        ):
            with self.assertRaises(HTTPException) as caught:
                await submit_mobile_feedback("WO-x", request)

        self.assertEqual(caught.exception.status_code, 404)

    async def test_conflict_raises_409(self) -> None:
        request = MobileFeedbackRequest(user_id=5, status="completed", process_message="x")
        with patch(
            "app.api.v1.work_orders.WorkOrderRepository.submit_mobile_feedback",
            side_effect=ValueError("已有处置结果等待电脑端审核"),
        ):
            with self.assertRaises(HTTPException) as caught:
                await submit_mobile_feedback("WO-1", request)

        self.assertEqual(caught.exception.status_code, 409)


class FeedbackReviewApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_success(self) -> None:
        order = _order(status="completed")
        request = FeedbackReviewRequest(decision="approve", review_message="材料完整")
        with patch(
            "app.api.v1.work_orders.WorkOrderRepository.review_mobile_feedback", return_value=order
        ) as review:
            result = await review_mobile_feedback("WO-1", request)

        self.assertIs(result, order)
        review.assert_called_once_with("WO-1", "approve", "材料完整")

    async def test_nothing_to_review_raises_404(self) -> None:
        request = FeedbackReviewRequest(decision="reject")
        with patch(
            "app.api.v1.work_orders.WorkOrderRepository.review_mobile_feedback", return_value=None
        ):
            with self.assertRaises(HTTPException) as caught:
                await review_mobile_feedback("WO-1", request)

        self.assertEqual(caught.exception.status_code, 404)


class StaffApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_list_staff_passthrough(self) -> None:
        staff = [SimpleNamespace(id="1")]
        with patch("app.api.v1.work_orders.WorkOrderRepository.list_staff", return_value=staff):
            result = await list_staff()

        self.assertEqual(result, staff)


if __name__ == "__main__":
    unittest.main()
