import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.modules.agent import tool


def _order(order_id: str, *, category: str = "traffic_police", status: str = "unassigned", level: str = "medium"):
    return SimpleNamespace(
        work_order_id=order_id,
        accident_info=f"事件{order_id}",
        event_level=level,
        status=status,
        required_category=category,
    )


def _staff(staff_id: str, *, category: str = "traffic_police", count: int = 0, distance: float = 1.0):
    return SimpleNamespace(
        id=staff_id,
        name=f"人员{staff_id}",
        role="处置员",
        work_order_count=count,
        distance_km=distance,
        personnel_category=category,
        user_work_describe=None,
    )


def _dispatched(order, staff_id: int):
    return SimpleNamespace(
        work_order_id=order.work_order_id,
        accident_info=order.accident_info,
        event_level=order.event_level,
        status="pending",
        assignee=f"人员{staff_id}",
    )


class AgentBatchToolTests(unittest.TestCase):
    def test_batch_dispatch_balances_workload_and_reports_unhandled_orders(self) -> None:
        orders = [
            _order("WO-20260714-300", category="traffic_police", level="high"),
            _order("WO-20260714-301", category="traffic_police", level="medium"),
            _order("WO-20260714-302", category="road_maintenance", level="low"),
            _order("WO-20260714-303", category="emergency_fire", level="high"),
            _order("WO-20260714-304", category="traffic_police", status="pending"),
        ]
        staff = [
            _staff("1", category="traffic_police", count=0, distance=2.0),
            _staff("2", category="traffic_police", count=0, distance=0.5),
            _staff("3", category="road_maintenance", count=1, distance=0.2),
        ]

        def dispatch(work_order_id: str, user_id: int):
            order = next(item for item in orders if item.work_order_id == work_order_id)
            return _dispatched(order, user_id)

        with (
            patch.object(tool.WorkOrderRepository, "list_work_orders", return_value=orders),
            patch.object(tool.WorkOrderRepository, "list_staff", return_value=staff),
            patch.object(tool.WorkOrderRepository, "dispatch_work_order", side_effect=dispatch) as dispatch_mock,
        ):
            payload = json.loads(tool.batch_dispatch_unassigned())

        self.assertEqual(payload["total"], 4)
        self.assertEqual(payload["success"], 3)
        self.assertEqual(payload["skipped"], 1)
        self.assertTrue(payload["needs_manual_fallback"])
        self.assertEqual(
            [(call.args[0], call.args[1]) for call in dispatch_mock.call_args_list],
            [
                ("WO-20260714-300", 2),
                ("WO-20260714-301", 1),
                ("WO-20260714-302", 3),
            ],
        )

    def test_dispatch_range_only_dispatches_unassigned_orders_inside_range(self) -> None:
        orders = [
            _order("WO-20260714-300"),
            _order("WO-20260714-301", status="pending"),
            _order("WO-20260714-302"),
            _order("WO-20260714-305"),
        ]
        staff = [_staff("1")]

        def dispatch(work_order_id: str, user_id: int):
            order = next(item for item in orders if item.work_order_id == work_order_id)
            return _dispatched(order, user_id)

        with (
            patch.object(tool.WorkOrderRepository, "list_work_orders", return_value=orders),
            patch.object(tool.WorkOrderRepository, "list_staff", return_value=staff),
            patch.object(tool.WorkOrderRepository, "dispatch_work_order", side_effect=dispatch) as dispatch_mock,
        ):
            payload = json.loads(tool.dispatch_work_order_range(300, 302))

        self.assertEqual(payload["range"], [300, 302])
        self.assertEqual(payload["total"], 2)
        self.assertEqual(payload["success"], 2)
        self.assertEqual([call.args[0] for call in dispatch_mock.call_args_list], ["WO-20260714-300", "WO-20260714-302"])

    def test_dry_run_batch_dispatch_plans_without_writing_database(self) -> None:
        orders = [
            _order("WO-20260714-300", category="traffic_police"),
            _order("WO-20260714-301", category="road_maintenance"),
            _order("WO-20260714-302", category="emergency_fire"),
            _order("WO-20260714-303", category="traffic_police", status="pending"),
        ]
        staff = [
            _staff("1", category="traffic_police", count=0),
            _staff("2", category="road_maintenance", count=0),
        ]

        with (
            patch.object(tool.WorkOrderRepository, "list_work_orders", return_value=orders),
            patch.object(tool.WorkOrderRepository, "list_staff", return_value=staff),
            patch.object(tool.WorkOrderRepository, "dispatch_work_order") as dispatch_mock,
        ):
            payload = json.loads(tool.dry_run_batch_dispatch(start_id=300, end_id=302))

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["tool"], "dry_run_batch_dispatch")
        self.assertEqual(payload["mode"], "dry_run")
        self.assertFalse(payload["would_modify_database"])
        self.assertEqual(payload["total"], 3)
        self.assertEqual(payload["success"], 2)
        self.assertEqual(payload["skipped"], 1)
        self.assertEqual(payload["affected_work_order_ids"], [])
        self.assertEqual(len(payload["fallback_candidates"]), 1)
        dispatch_mock.assert_not_called()

    def test_batch_ignore_marks_filtered_orders_as_ignored_with_reason(self) -> None:
        orders = [
            _order("WO-20260714-300", category="traffic_police", status="unassigned"),
            _order("WO-20260714-301", category="traffic_police", status="pending"),
            _order("WO-20260714-302", category="road_maintenance", status="unassigned"),
            _order("WO-20260714-303", category="traffic_police", status="unassigned"),
        ]

        def update(work_order_id: str, status: str, process_message: str = "", process_image_url=None):
            order = next(item for item in orders if item.work_order_id == work_order_id)
            return SimpleNamespace(
                work_order_id=order.work_order_id,
                accident_info=order.accident_info,
                event_level=order.event_level,
                status=status,
            )

        with (
            patch.object(tool.WorkOrderRepository, "list_work_orders", return_value=orders),
            patch.object(tool.WorkOrderRepository, "update_work_order_status", side_effect=update) as update_mock,
        ):
            payload = json.loads(tool.batch_ignore_work_orders(
                start_id=300,
                end_id=303,
                stage="unassigned",
                required_category="traffic_police",
                reason="无对应处置资源,暂不处理。",
            ))

        self.assertEqual(payload["total"], 2)
        self.assertEqual(payload["ignored"], 2)
        self.assertEqual(payload["failed"], 0)
        self.assertEqual(
            [(call.args[0], call.args[1], call.kwargs["process_message"]) for call in update_mock.call_args_list],
            [
                ("WO-20260714-300", "ignored", "无对应处置资源,暂不处理。"),
                ("WO-20260714-303", "ignored", "无对应处置资源,暂不处理。"),
            ],
        )

    def test_batch_dispatch_and_ignore_scale_to_500_orders(self) -> None:
        supported_categories = ["traffic_police", "road_maintenance", "municipal_facilities"]
        unsupported_categories = ["emergency_fire", "vehicle_rescue"]
        all_categories = supported_categories + unsupported_categories
        orders = [
            _order(
                f"WO-20260714-{index:03d}",
                category=all_categories[index % len(all_categories)],
                status="pending" if index % 10 == 0 else "unassigned",
                level=("high" if index % 7 == 0 else "medium" if index % 3 == 0 else "low"),
            )
            for index in range(1, 501)
        ]
        staff = [
            _staff("1", category="traffic_police", count=0, distance=0.5),
            _staff("2", category="traffic_police", count=0, distance=1.0),
            _staff("3", category="road_maintenance", count=1, distance=0.7),
            _staff("4", category="road_maintenance", count=1, distance=1.2),
            _staff("5", category="municipal_facilities", count=2, distance=0.6),
            _staff("6", category="municipal_facilities", count=2, distance=1.1),
        ]

        def dispatch(work_order_id: str, user_id: int):
            order = next(item for item in orders if item.work_order_id == work_order_id)
            order.status = "pending"
            return _dispatched(order, user_id)

        def update(work_order_id: str, status: str, process_message: str = "", process_image_url=None):
            order = next(item for item in orders if item.work_order_id == work_order_id)
            order.status = status
            return SimpleNamespace(
                work_order_id=order.work_order_id,
                accident_info=order.accident_info,
                event_level=order.event_level,
                status=status,
            )

        with (
            patch.object(tool.WorkOrderRepository, "list_work_orders", return_value=orders),
            patch.object(tool.WorkOrderRepository, "list_staff", return_value=staff),
            patch.object(tool.WorkOrderRepository, "dispatch_work_order", side_effect=dispatch) as dispatch_mock,
            patch.object(tool.WorkOrderRepository, "update_work_order_status", side_effect=update) as update_mock,
        ):
            dispatch_payload = json.loads(tool.batch_dispatch_unassigned())
            ignore_fire_payload = json.loads(tool.batch_ignore_work_orders(
                stage="unassigned",
                required_category="emergency_fire",
                reason="规模测试: emergency_fire 暂无可用人员。",
            ))
            ignore_rescue_payload = json.loads(tool.batch_ignore_work_orders(
                stage="unassigned",
                required_category="vehicle_rescue",
                reason="规模测试: vehicle_rescue 暂无可用人员。",
            ))

        self.assertEqual(len(orders), 500)
        self.assertEqual(dispatch_payload["total"], 450)
        self.assertEqual(dispatch_payload["success"], 250)
        self.assertEqual(dispatch_payload["skipped"], 200)
        self.assertEqual(dispatch_payload["failed"], 0)
        self.assertTrue(dispatch_payload["needs_manual_fallback"])
        self.assertEqual(dispatch_mock.call_count, 250)
        self.assertEqual(ignore_fire_payload["ignored"], 100)
        self.assertEqual(ignore_rescue_payload["ignored"], 100)
        self.assertEqual(update_mock.call_count, 200)

        final_status_counts = {}
        for order in orders:
            final_status_counts[order.status] = final_status_counts.get(order.status, 0) + 1
        self.assertEqual(final_status_counts, {"pending": 300, "ignored": 200})

    def test_batch_tools_keep_linear_performance_on_5000_orders(self) -> None:
        categories = [
            "traffic_police",
            "road_maintenance",
            "municipal_facilities",
            "emergency_fire",
            "vehicle_rescue",
        ]
        orders = [
            _order(
                f"WO-20260714-{index:05d}",
                category=categories[index % len(categories)],
                status="unassigned",
                level="medium",
            )
            for index in range(1, 5001)
        ]
        order_by_id = {order.work_order_id: order for order in orders}
        staff = [
            _staff("1", category="traffic_police", count=0),
            _staff("2", category="traffic_police", count=0),
            _staff("3", category="road_maintenance", count=0),
            _staff("4", category="road_maintenance", count=0),
            _staff("5", category="municipal_facilities", count=0),
            _staff("6", category="municipal_facilities", count=0),
        ]

        def dispatch(work_order_id: str, user_id: int):
            order = order_by_id[work_order_id]
            order.status = "pending"
            return _dispatched(order, user_id)

        def update(work_order_id: str, status: str, process_message: str = "", process_image_url=None):
            order = order_by_id[work_order_id]
            order.status = status
            return SimpleNamespace(
                work_order_id=order.work_order_id,
                accident_info=order.accident_info,
                event_level=order.event_level,
                status=status,
            )

        with (
            patch.object(tool.WorkOrderRepository, "list_work_orders", return_value=orders),
            patch.object(tool.WorkOrderRepository, "list_staff", return_value=staff),
            patch.object(tool.WorkOrderRepository, "dispatch_work_order", side_effect=dispatch),
            patch.object(tool.WorkOrderRepository, "update_work_order_status", side_effect=update),
        ):
            started = time.perf_counter()
            dry_payload = json.loads(tool.dry_run_batch_dispatch())
            dry_elapsed_ms = (time.perf_counter() - started) * 1000

            started = time.perf_counter()
            dispatch_payload = json.loads(tool.batch_dispatch_unassigned())
            dispatch_elapsed_ms = (time.perf_counter() - started) * 1000

            started = time.perf_counter()
            ignore_payload = json.loads(tool.batch_ignore_work_orders(
                stage="unassigned",
                required_category="emergency_fire",
                reason="性能测试: 暂无可用人员。",
            ))
            ignore_elapsed_ms = (time.perf_counter() - started) * 1000

        self.assertEqual(dry_payload["total"], 5000)
        self.assertEqual(dry_payload["success"], 3000)
        self.assertEqual(dry_payload["skipped"], 2000)
        self.assertEqual(dispatch_payload["success"], 3000)
        self.assertEqual(dispatch_payload["skipped"], 2000)
        self.assertEqual(ignore_payload["ignored"], 1000)
        self.assertLess(dry_elapsed_ms, 1500)
        self.assertLess(dispatch_elapsed_ms, 2500)
        self.assertLess(ignore_elapsed_ms, 1500)


if __name__ == "__main__":
    unittest.main()
