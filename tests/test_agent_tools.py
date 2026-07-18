"""app.modules.agent.tool 工具层测试:所有数据库访问与RAG检索均以 mock 替代。"""

import json
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

from app.modules.agent import tool


def _work_order(**overrides):
    values = {
        "work_order_id": "WO-20260712-144",
        "event_id": "EV-144",
        "camera_id": "live30",
        "camera_name": "摄像头 live30",
        "monitor_address": "测试路口",
        "accident_info": "车辆碰撞",
        "event_time": "2026-07-12 16:00:51",
        "event_level": "high",
        "status": "unassigned",
        "assignee": None,
        "assignee_user_id": None,
        "description": "两车发生碰撞。",
        "ai_suggestion": "先设置警戒区。",
        "scene_info": "晚高峰，路面湿滑。",
        "process_message": None,
        "completed_at": None,
        "required_category": "traffic_police",
        "scene_images": [],
        "process_images": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _staff(staff_id: int, count: int, category: str = "traffic_police", distance: float = 1.0):
    return SimpleNamespace(
        id=str(staff_id),
        name=f"人员{staff_id}",
        role="交警执法",
        work_order_count=count,
        distance_km=distance,
        personnel_category=category,
        user_work_describe=f"职责{staff_id}",
    )


class ParseWorkOrderIdTest(unittest.TestCase):
    def test_numeric_and_prefixed_ids(self) -> None:
        self.assertEqual(tool._parse_work_order_id("144"), 144)
        self.assertEqual(tool._parse_work_order_id("WO-20260709-001"), 1)
        self.assertEqual(tool._parse_work_order_id("wo-007"), 7)
        self.assertEqual(tool._parse_work_order_id(" 12 "), 12)

    def test_invalid_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            tool._parse_work_order_id("abc")


class SerializeDatetimeTest(unittest.TestCase):
    def test_none_datetime_and_string(self) -> None:
        self.assertIsNone(tool._serialize_datetime(None))
        self.assertEqual(
            tool._serialize_datetime(datetime(2026, 7, 12, 16, 0, 51)),
            "2026-07-12 16:00:51",
        )
        self.assertEqual(tool._serialize_datetime("2026-07-12"), "2026-07-12")


class GetWorkOrderDetailToolTest(unittest.TestCase):
    def test_returns_full_detail_with_flags(self) -> None:
        work_order = _work_order(
            status="completed",
            process_message="已清障",
            process_images=["/orderImg/r.jpg"],
            completed_at="2026-07-12 17:00:00",
        )
        with patch.object(tool.WorkOrderRepository, "get_work_order", return_value=work_order):
            payload = json.loads(tool.get_work_order_detail("144"))

        self.assertEqual(payload["work_order_id"], "WO-20260712-144")
        self.assertEqual(payload["accident_info"], "车辆碰撞")
        self.assertEqual(payload["process_image_count"], 1)
        self.assertTrue(payload["has_process_message"])
        self.assertTrue(payload["has_completed_at"])

    def test_missing_work_order_returns_error_envelope(self) -> None:
        with patch.object(tool.WorkOrderRepository, "get_work_order", return_value=None):
            payload = json.loads(tool.get_work_order_detail("999"))

        self.assertIn("未找到工单", payload["error"])
        self.assertIsNone(payload["data"])

    def test_repository_failure_returns_error_envelope(self) -> None:
        with patch.object(
            tool.WorkOrderRepository, "get_work_order", side_effect=RuntimeError("db down")
        ):
            payload = json.loads(tool.get_work_order_detail("144"))

        self.assertIn("数据库查询异常", payload["error"])


class QueryWorkOrdersToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.orders = [
            _work_order(work_order_id="WO-1", status="unassigned", event_level="high"),
            _work_order(work_order_id="WO-2", status="pending", event_level="low"),
            _work_order(work_order_id="WO-3", status="completed", event_level="high"),
            _work_order(work_order_id="WO-4", status="ignored", event_level="medium"),
        ]

    def _run(self, **kwargs):
        with patch.object(
            tool.WorkOrderRepository, "list_work_orders", return_value=list(self.orders)
        ) as list_orders:
            payload = json.loads(tool.query_work_orders(**kwargs))
        return payload, list_orders

    def test_default_filter_excludes_terminal_orders(self) -> None:
        payload, _ = self._run()

        ids = [item["work_order_id"] for item in payload["work_orders"]]
        self.assertEqual(ids, ["WO-1", "WO-2"])
        self.assertEqual(payload["stats"]["total"], 2)
        self.assertEqual(payload["stats"]["by_stage"], {"unassigned": 1, "pending": 1})
        self.assertEqual(payload["stats"]["by_level"], {"high": 1, "low": 1})

    def test_stage_and_level_filters(self) -> None:
        payload, _ = self._run(stage="completed")
        self.assertEqual([i["work_order_id"] for i in payload["work_orders"]], ["WO-3"])

        payload, _ = self._run(event_level="high")
        self.assertEqual([i["work_order_id"] for i in payload["work_orders"]], ["WO-1"])

    def test_limit_truncates_results(self) -> None:
        payload, _ = self._run(limit=1)
        self.assertEqual(len(payload["work_orders"]), 1)

    def test_user_id_is_forwarded_to_repository(self) -> None:
        _, list_orders = self._run(user_id=5)
        list_orders.assert_called_once_with(user_id=5)

    def test_repository_failure_returns_error_envelope(self) -> None:
        with patch.object(
            tool.WorkOrderRepository, "list_work_orders", side_effect=RuntimeError("db down")
        ):
            payload = json.loads(tool.query_work_orders())

        self.assertIn("查询工单列表失败", payload["error"])


class QueryStaffToolTest(unittest.TestCase):
    def test_staff_sorted_by_workload_with_average(self) -> None:
        staff = [_staff(1, 4), _staff(2, 1), _staff(3, 2)]
        with patch.object(tool.WorkOrderRepository, "list_staff", return_value=staff):
            payload = json.loads(tool.query_staff())

        self.assertEqual([s["id"] for s in payload["staff"]], ["2", "3", "1"])
        self.assertEqual(payload["total"], 3)
        self.assertAlmostEqual(payload["avg_workload"], round(7 / 3, 1))

    def test_empty_staff_list(self) -> None:
        with patch.object(tool.WorkOrderRepository, "list_staff", return_value=[]):
            payload = json.loads(tool.query_staff())

        self.assertEqual(payload["staff"], [])
        self.assertEqual(payload["avg_workload"], 0)


class QueryWorkOrderStatsToolTest(unittest.TestCase):
    def test_aggregates_and_high_priority_shortlist(self) -> None:
        orders = [
            _work_order(work_order_id=f"WO-{i}", status="pending", event_level="high")
            for i in range(7)
        ] + [
            _work_order(work_order_id="WO-c", status="completed", event_level="high"),
            _work_order(work_order_id="WO-l", status="unassigned", event_level="low"),
        ]
        with patch.object(tool.WorkOrderRepository, "list_work_orders", return_value=orders):
            payload = json.loads(tool.query_work_order_stats())

        self.assertEqual(payload["total"], 9)
        self.assertEqual(payload["by_stage"]["pending"], 7)
        self.assertEqual(payload["by_level"]["high"], 8)
        # 已完成的高等级工单不计入未解决清单,且清单最多5条
        self.assertEqual(len(payload["high_priority_unresolved"]), 5)
        self.assertTrue(all(item["stage"] == "pending" for item in payload["high_priority_unresolved"]))


class SuggestHandlingToolTest(unittest.TestCase):
    def test_stage_guidance_and_rag_references(self) -> None:
        work_order = _work_order(status="processing", scene_images=[])
        rag_result = {
            "中华人民共和国道路交通安全法": [
                {"chapter": f"第{i}条", "describe": f"条文{i}"} for i in range(5)
            ]
        }
        with (
            patch.object(tool.WorkOrderRepository, "get_work_order", return_value=work_order),
            patch.object(tool, "_search_regulations", return_value=rag_result),
        ):
            payload = json.loads(tool.suggest_handling("144"))

        self.assertEqual(payload["current_stage"], "processing")
        self.assertIn("升级条件", payload["stage_guidance"])
        # 每部法规最多取3条
        self.assertEqual(len(payload["rag_references"]), 3)
        self.assertEqual(payload["image_analysis"], "")

    def test_rag_failure_is_tolerated(self) -> None:
        with (
            patch.object(tool.WorkOrderRepository, "get_work_order", return_value=_work_order()),
            patch.object(tool, "_search_regulations", side_effect=RuntimeError("rag down")),
        ):
            payload = json.loads(tool.suggest_handling("144"))

        self.assertEqual(payload["rag_references"], [])
        self.assertEqual(payload["work_order_id"], "WO-20260712-144")

    def test_scene_images_trigger_vlm_analysis(self) -> None:
        work_order = _work_order(scene_images=["/orderImg/scene-x.jpg"])
        with (
            patch.object(tool.WorkOrderRepository, "get_work_order", return_value=work_order),
            patch.object(tool, "_search_regulations", return_value={}),
            patch.object(tool, "_load_images_base64", return_value=[{"data": "data:image/jpeg;base64,AA==", "filename": "scene-x.jpg"}]),
            patch.object(tool, "_call_vlm_with_images", return_value="画面显示两车追尾。") as call_vlm,
        ):
            payload = json.loads(tool.suggest_handling("144"))

        call_vlm.assert_called_once()
        self.assertEqual(payload["image_analysis"], "画面显示两车追尾。")
        self.assertEqual(payload["scene_image_count"], 1)

    def test_missing_work_order(self) -> None:
        with patch.object(tool.WorkOrderRepository, "get_work_order", return_value=None):
            payload = json.loads(tool.suggest_handling("999"))

        self.assertIn("未找到工单", payload["error"])


class ImageLoadingTest(unittest.TestCase):
    def test_missing_image_file_produces_error_entry(self) -> None:
        images = tool._load_images_base64(["/orderImg/definitely-missing-000.jpg"])

        self.assertEqual(len(images), 1)
        self.assertIn("error", images[0])
        self.assertIn("不存在", images[0]["error"])

    def test_existing_image_is_encoded_as_data_uri(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "scene-real.jpg"
            image_path.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
            with patch.object(tool, "_IMAGE_DIR", Path(tmp_dir)):
                images = tool._load_images_base64(["/orderImg/scene-real.jpg"])

        self.assertEqual(images[0]["filename"], "scene-real.jpg")
        self.assertTrue(images[0]["data"].startswith("data:image/jpg;base64,"))


class DispatchWorkOrderToolTest(unittest.TestCase):
    def test_missing_work_order(self) -> None:
        with patch.object(tool.WorkOrderRepository, "get_work_order", return_value=None):
            payload = json.loads(tool.dispatch_work_order("999", 1))

        self.assertIn("工单不存在", payload["error"])

    def test_non_unassigned_stage_is_rejected_before_dispatch(self) -> None:
        with (
            patch.object(
                tool.WorkOrderRepository, "get_work_order", return_value=_work_order(status="pending")
            ),
            patch.object(tool.WorkOrderRepository, "dispatch_work_order") as dispatch,
        ):
            payload = json.loads(tool.dispatch_work_order("144", 1))

        dispatch.assert_not_called()
        self.assertIn("仅未派发", payload["error"])
        self.assertEqual(payload["data"]["status"], "pending")

    def test_successful_dispatch_returns_confirmation(self) -> None:
        dispatched = _work_order(status="pending", assignee="人员2")
        with (
            patch.object(tool.WorkOrderRepository, "get_work_order", return_value=_work_order()),
            patch.object(tool.WorkOrderRepository, "dispatch_work_order", return_value=dispatched),
        ):
            payload = json.loads(tool.dispatch_work_order("144", 2))

        self.assertEqual(payload["status"], "pending")
        self.assertIn("已成功派发", payload["message"])

    def test_repository_value_error_is_surfaced(self) -> None:
        with (
            patch.object(tool.WorkOrderRepository, "get_work_order", return_value=_work_order()),
            patch.object(
                tool.WorkOrderRepository, "dispatch_work_order", side_effect=ValueError("状态已变更")
            ),
        ):
            payload = json.loads(tool.dispatch_work_order("144", 2))

        self.assertIn("状态已变更", payload["error"])

    def test_none_result_reports_failure(self) -> None:
        with (
            patch.object(tool.WorkOrderRepository, "get_work_order", return_value=_work_order()),
            patch.object(tool.WorkOrderRepository, "dispatch_work_order", return_value=None),
        ):
            payload = json.loads(tool.dispatch_work_order("144", 2))

        self.assertIn("派发失败", payload["error"])


class SelectBestStaffTest(unittest.TestCase):
    def test_filters_category_and_balances_in_batch_workload(self) -> None:
        staff = [
            _staff(1, 0, category="road_maintenance"),
            _staff(2, 1),
            _staff(3, 0, distance=0.5),
        ]

        best = tool._select_best_staff(staff, "traffic_police", {})
        self.assertEqual(best.id, "3")

        # 同批次已派一单后总负载相同(1:1),按距离决胜仍是人员3
        best = tool._select_best_staff(staff, "traffic_police", {3: 1})
        self.assertEqual(best.id, "3")

        # 已派两单后人员3总负载为2,负载最低者变为人员2
        best = tool._select_best_staff(staff, "traffic_police", {3: 2})
        self.assertEqual(best.id, "2")

    def test_returns_none_without_matching_category(self) -> None:
        self.assertIsNone(tool._select_best_staff([_staff(1, 0)], "emergency_fire", {}))


class StructuredBatchPayloadTest(unittest.TestCase):
    def test_payload_shape_and_fallback_candidates(self) -> None:
        details = [
            {"work_order_id": "WO-1", "result": "success"},
            {"work_order_id": "WO-2", "result": "skipped", "reason": "无可用人员", "type": "车辆抛锚", "level": "low"},
            {"work_order_id": "WO-3", "result": "failed", "reason": "db error"},
        ]
        payload = tool._structured_batch_payload(
            tool_name="t",
            mode="write",
            action="a",
            total=3,
            success=1,
            skipped=1,
            failed=1,
            details=details,
            presentation_hint="hint",
        )

        self.assertFalse(payload["ok"])
        self.assertEqual(payload["affected_work_order_ids"], ["WO-1"])
        self.assertEqual(len(payload["fallback_candidates"]), 2)
        self.assertTrue(payload["needs_manual_fallback"])
        self.assertEqual(payload["next_actions"][0]["tool"], "batch_ignore_work_orders")

        empty = tool._structured_batch_payload(
            tool_name="t", mode="write", action="a", total=0, details=[], presentation_hint="h"
        )
        self.assertTrue(empty["ok"])
        self.assertEqual(empty["next_actions"], [])
        self.assertEqual(empty["fallback_hint"], "")


class DryRunBatchDispatchToolTest(unittest.TestCase):
    def test_rejects_unsupported_stage(self) -> None:
        payload = json.loads(tool.dry_run_batch_dispatch(stage="completed"))

        self.assertFalse(payload["ok"])
        self.assertIn("不支持预演阶段", payload["error"])

    def test_rejects_half_open_range(self) -> None:
        payload = json.loads(tool.dry_run_batch_dispatch(start_id=10))

        self.assertFalse(payload["ok"])
        self.assertIn("必须同时提供", payload["error"])

    def test_plans_assignments_without_touching_database(self) -> None:
        orders = [
            _work_order(work_order_id="WO-20260712-001"),
            _work_order(work_order_id="WO-20260712-002"),
            _work_order(work_order_id="WO-20260712-003", required_category="emergency_fire"),
            _work_order(work_order_id="WO-20260712-004", status="pending"),
        ]
        staff = [_staff(1, 2), _staff(2, 0)]
        with (
            patch.object(tool.WorkOrderRepository, "list_work_orders", return_value=orders),
            patch.object(tool.WorkOrderRepository, "list_staff", return_value=staff),
            patch.object(tool.WorkOrderRepository, "dispatch_work_order") as dispatch,
        ):
            payload = json.loads(tool.dry_run_batch_dispatch())

        dispatch.assert_not_called()
        self.assertFalse(payload["would_modify_database"])
        self.assertEqual(payload["summary"]["total"], 3)
        self.assertEqual(payload["summary"]["success"], 2)
        self.assertEqual(payload["summary"]["skipped"], 1)
        # pending工单不参与预演;无应急消防类别人员的工单被跳过
        planned_ids = [d["work_order_id"] for d in payload["details"] if d["result"] == "planned"]
        self.assertEqual(planned_ids, ["WO-20260712-001", "WO-20260712-002"])
        skipped = payload["details"][2]
        self.assertEqual(skipped["result"], "skipped")
        self.assertIn("emergency_fire", skipped["reason"])

    def test_range_filter_selects_only_in_range_orders(self) -> None:
        orders = [
            _work_order(work_order_id="WO-20260712-001"),
            _work_order(work_order_id="WO-20260712-050"),
            _work_order(work_order_id="WO-20260712-099"),
        ]
        with (
            patch.object(tool.WorkOrderRepository, "list_work_orders", return_value=orders),
            patch.object(tool.WorkOrderRepository, "list_staff", return_value=[_staff(1, 0)]),
        ):
            payload = json.loads(tool.dry_run_batch_dispatch(start_id=10, end_id=60))

        self.assertEqual(payload["range"], [10, 60])
        self.assertEqual(payload["summary"]["total"], 1)
        self.assertEqual(payload["details"][0]["work_order_id"], "WO-20260712-050")


class BatchDispatchUnassignedToolTest(unittest.TestCase):
    def test_no_unassigned_orders(self) -> None:
        with patch.object(
            tool.WorkOrderRepository,
            "list_work_orders",
            return_value=[_work_order(status="completed")],
        ):
            payload = json.loads(tool.batch_dispatch_unassigned())

        self.assertIn("没有未派发工单", payload["message"])
        self.assertEqual(payload["total"], 0)

    def test_no_available_staff(self) -> None:
        with (
            patch.object(tool.WorkOrderRepository, "list_work_orders", return_value=[_work_order()]),
            patch.object(tool.WorkOrderRepository, "list_staff", return_value=[]),
        ):
            payload = json.loads(tool.batch_dispatch_unassigned())

        self.assertIn("无可派发人员", payload["error"])

    def test_dispatches_all_unassigned_with_load_balancing(self) -> None:
        orders = [_work_order(work_order_id="WO-20260712-001"), _work_order(work_order_id="WO-20260712-002")]
        staff = [_staff(1, 0), _staff(2, 0, distance=0.5)]

        dispatched = []
        def fake_dispatch(work_order_id, user_id):
            dispatched.append((work_order_id, user_id))
            return _work_order(work_order_id=work_order_id, status="pending", assignee=f"人员{user_id}")

        with (
            patch.object(tool.WorkOrderRepository, "list_work_orders", return_value=orders),
            patch.object(tool.WorkOrderRepository, "list_staff", return_value=staff),
            patch.object(tool.WorkOrderRepository, "dispatch_work_order", side_effect=fake_dispatch),
        ):
            payload = json.loads(tool.batch_dispatch_unassigned())

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["summary"], {"total": 2, "success": 2, "skipped": 0, "failed": 0})
        # 负载均衡:两单派给不同人员
        self.assertEqual({user_id for _, user_id in dispatched}, {1, 2})

    def test_failed_dispatch_is_reported_per_order(self) -> None:
        orders = [_work_order(work_order_id="WO-20260712-001")]
        with (
            patch.object(tool.WorkOrderRepository, "list_work_orders", return_value=orders),
            patch.object(tool.WorkOrderRepository, "list_staff", return_value=[_staff(1, 0)]),
            patch.object(tool.WorkOrderRepository, "dispatch_work_order", side_effect=ValueError("并发冲突")),
        ):
            payload = json.loads(tool.batch_dispatch_unassigned())

        self.assertFalse(payload["ok"])
        self.assertEqual(payload["summary"]["failed"], 1)
        self.assertEqual(payload["failed_items"][0]["reason"], "并发冲突")


class DispatchWorkOrderRangeToolTest(unittest.TestCase):
    def test_empty_range_returns_message(self) -> None:
        with patch.object(tool.WorkOrderRepository, "list_work_orders", return_value=[]):
            payload = json.loads(tool.dispatch_work_order_range(300, 330))

        self.assertIn("没有未派发工单", payload["message"])
        self.assertEqual(payload["range"], [300, 330])

    def test_range_is_normalized_and_dispatched(self) -> None:
        orders = [
            _work_order(work_order_id="WO-20260712-299"),
            _work_order(work_order_id="WO-20260712-300"),
        ]
        with (
            patch.object(tool.WorkOrderRepository, "list_work_orders", return_value=orders),
            patch.object(tool.WorkOrderRepository, "list_staff", return_value=[_staff(1, 0)]),
            patch.object(
                tool.WorkOrderRepository,
                "dispatch_work_order",
                side_effect=lambda wid, uid: _work_order(work_order_id=wid, status="pending", assignee="人员1"),
            ) as dispatch,
        ):
            # 区间倒置时自动交换
            payload = json.loads(tool.dispatch_work_order_range(330, 300))

        self.assertEqual(payload["range"], [300, 330])
        self.assertEqual(payload["summary"]["success"], 1)
        dispatch.assert_called_once_with("WO-20260712-300", 1)


class BatchIgnoreWorkOrdersToolTest(unittest.TestCase):
    def test_rejects_unsupported_stage(self) -> None:
        payload = json.loads(tool.batch_ignore_work_orders(stage="completed"))

        self.assertIn("不允许批量忽略阶段", payload["error"])

    def test_rejects_half_open_range(self) -> None:
        payload = json.loads(tool.batch_ignore_work_orders(end_id=5))

        self.assertIn("必须同时提供", payload["error"])

    def test_ignores_matching_orders_with_reason(self) -> None:
        orders = [
            _work_order(work_order_id="WO-20260712-001"),
            _work_order(work_order_id="WO-20260712-002", event_level="low"),
            _work_order(work_order_id="WO-20260712-003", status="pending"),
        ]
        updated = []

        def fake_update(work_order_id, status, process_message=None):
            updated.append((work_order_id, status, process_message))
            return _work_order(work_order_id=work_order_id, status="ignored")

        with (
            patch.object(tool.WorkOrderRepository, "list_work_orders", return_value=orders),
            patch.object(tool.WorkOrderRepository, "update_work_order_status", side_effect=fake_update),
        ):
            payload = json.loads(tool.batch_ignore_work_orders(event_level="high", reason="  无可用人员  "))

        # 仅高等级且unassigned的工单被忽略;pending被阶段过滤
        self.assertEqual([u[0] for u in updated], ["WO-20260712-001"])
        self.assertEqual(updated[0][1], "ignored")
        self.assertEqual(updated[0][2], "无可用人员")
        self.assertEqual(payload["summary"]["success"], 1)
        self.assertEqual(payload["ignored"], 1)

    def test_blank_reason_falls_back_to_default(self) -> None:
        orders = [_work_order(work_order_id="WO-20260712-001")]
        with (
            patch.object(tool.WorkOrderRepository, "list_work_orders", return_value=orders),
            patch.object(
                tool.WorkOrderRepository,
                "update_work_order_status",
                side_effect=lambda wid, status, process_message=None: _work_order(work_order_id=wid),
            ) as update,
        ):
            json.loads(tool.batch_ignore_work_orders(reason="   "))

        self.assertEqual(update.call_args.kwargs["process_message"], "无法处理,批量忽略。")

    def test_unparseable_id_is_reported_as_skipped(self) -> None:
        orders = [_work_order(work_order_id="BROKEN-ID")]
        with patch.object(tool.WorkOrderRepository, "list_work_orders", return_value=orders):
            payload = json.loads(tool.batch_ignore_work_orders())

        self.assertEqual(payload["summary"]["skipped"], 1)
        self.assertEqual(payload["skipped_items"][0]["reason"], "工单编号无法解析")

    def test_failed_update_is_counted(self) -> None:
        orders = [_work_order(work_order_id="WO-20260712-001")]
        with (
            patch.object(tool.WorkOrderRepository, "list_work_orders", return_value=orders),
            patch.object(tool.WorkOrderRepository, "update_work_order_status", return_value=None),
        ):
            payload = json.loads(tool.batch_ignore_work_orders())

        self.assertEqual(payload["summary"]["failed"], 1)
        self.assertFalse(payload["ok"])


class AnswerGeneralQuestionToolTest(unittest.TestCase):
    def test_references_are_truncated_to_150_chars(self) -> None:
        long_text = "条" * 200
        with patch.object(
            tool,
            "_search_regulations",
            return_value={"法规A": [{"chapter": "第一条", "describe": long_text}]},
        ):
            payload = json.loads(tool.answer_general_question("问题"))

        self.assertEqual(len(payload["rag_references"][0]["excerpt"]), 150)
        self.assertEqual(payload["question"], "问题")

    def test_rag_failure_returns_empty_references_with_note(self) -> None:
        with patch.object(tool, "_search_regulations", side_effect=RuntimeError("rag down")):
            payload = json.loads(tool.answer_general_question("问题"))

        self.assertEqual(payload["rag_references"], [])
        self.assertIn("rag_references", payload["note"])


class SearchWorkOrdersToolTest(unittest.TestCase):
    class _FakeCursor:
        def __init__(self, row):
            self._row = row
            self.executed = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, sql, args=None):
            self.executed.append((sql, args))

        def fetchone(self):
            return self._row

    class _FakeConnection:
        def __init__(self, cursor):
            self._cursor = cursor

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def cursor(self):
            return self._cursor

    def _run_with_row(self, row, work_order_id="144"):
        cursor = self._FakeCursor(row)
        connection = self._FakeConnection(cursor)
        with patch.object(tool, "mysql_connection", return_value=connection):
            payload = json.loads(tool.search_work_orders(work_order_id))
        return payload, cursor

    def test_invalid_id_returns_error(self) -> None:
        payload = json.loads(tool.search_work_orders("not-a-number"))
        self.assertIn("无效的工单编号格式", payload["error"])

    def test_found_row_is_serialized(self) -> None:
        row = {
            "work_order_id": 144,
            "event_id": "EV-144",
            "camera_id": "cam-01",
            "camera_name": "摄像头",
            "monitor_address": "路口",
            "work_order_type": "车辆碰撞",
            "work_order_describe": "描述",
            "work_order_img_url": "/a.jpg",
            "work_order_rank": 3,
            "work_order_time": datetime(2026, 7, 12, 16, 0, 51),
            "work_order_stage": "pending",
            "work_order_status": 0,
            "ai_suggestion": "建议",
            "scene_info": "现场",
            "completed_at": None,
            "assignee_name": "张三",
            "assignee_user_id": 7,
            "assignee_type": "交警",
        }
        payload, cursor = self._run_with_row(row, "WO-20260712-144")

        self.assertEqual(payload["work_order_id"], 144)
        self.assertEqual(payload["work_order_time"], "2026-07-12 16:00:51")
        self.assertEqual(payload["assignee_name"], "张三")
        # WO-前缀编号被解析为数字主键查询
        self.assertEqual(cursor.executed[0][1], (144,))

    def test_missing_row_returns_error(self) -> None:
        payload, _ = self._run_with_row(None)
        self.assertIn("未找到工单", payload["error"])


class ToolRegistryTest(unittest.TestCase):
    def test_all_twelve_tools_are_registered(self) -> None:
        self.assertEqual(
            [fn.__name__ for fn in tool.TOOLS],
            [
                "query_work_orders",
                "get_work_order_detail",
                "search_work_orders",
                "query_staff",
                "query_work_order_stats",
                "suggest_handling",
                "dispatch_work_order",
                "dry_run_batch_dispatch",
                "batch_dispatch_unassigned",
                "dispatch_work_order_range",
                "batch_ignore_work_orders",
                "answer_general_question",
            ],
        )


if __name__ == "__main__":
    unittest.main()
