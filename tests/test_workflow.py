"""AgentWorkflow 编排器测试:诊断、态势简报、派发推荐及行动计划。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.modules.agent.workflow import AgentWorkflow


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
        "description": "两车发生碰撞。",
        "ai_suggestion": "先设置警戒区。",
        "scene_info": "晚高峰。",
        "process_message": None,
        "completed_at": None,
        "required_category": "traffic_police",
        "scene_images": ["/orderImg/scene.jpg"],
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
        user_work_describe=None,
    )


class DiagnoseWorkflowTest(unittest.TestCase):
    def test_missing_work_order(self) -> None:
        with patch(
            "app.modules.agent.workflow.WorkOrderRepository.get_work_order", return_value=None
        ):
            result = AgentWorkflow.diagnose("999")

        self.assertFalse(result.success)
        self.assertIn("工单不存在", result.error)

    def test_full_report_with_matching_staff_and_evidence_boundary(self) -> None:
        order = _work_order()
        staff = [_staff(1, 3), _staff(2, 1), _staff(3, 0, category="road_maintenance")]
        with (
            patch(
                "app.modules.agent.workflow.WorkOrderRepository.get_work_order", return_value=order
            ),
            patch(
                "app.modules.agent.workflow.WorkOrderRepository.list_staff", return_value=staff
            ),
        ):
            result = AgentWorkflow.diagnose("144")

        self.assertTrue(result.success)
        report = result.data
        self.assertEqual(report["event_level_label"], "紧急")
        self.assertEqual(report["current_stage_label"], "未派发")
        # 只推荐同类别人员,按负载升序
        self.assertEqual(
            [s["id"] for s in report["available_staff"]["matching_staff"]],
            ["2", "1"],
        )
        self.assertEqual(report["available_staff"]["total_matching"], 2)
        # 证据边界:数据库字段已核验,图片文件内容未核验
        self.assertEqual(report["evidence"]["source_type"], "database")
        self.assertFalse(report["file_evidence"]["content_verified"])
        self.assertFalse(report["file_evidence"]["existence_verified"])
        self.assertFalse(report["recommended_actions_basis"]["verified_against_regulation"])

    def test_repository_failure_returns_failure_result(self) -> None:
        with patch(
            "app.modules.agent.workflow.WorkOrderRepository.get_work_order",
            side_effect=RuntimeError("db down"),
        ):
            result = AgentWorkflow.diagnose("144")

        self.assertFalse(result.success)
        self.assertIn("db down", result.error)


class BriefWorkflowTest(unittest.TestCase):
    def test_summary_aggregates(self) -> None:
        orders = [
            _work_order(work_order_id="WO-1", status="pending", event_level="high"),
            _work_order(work_order_id="WO-2", status="processing", event_level="medium"),
            _work_order(work_order_id="WO-3", status="completed", event_level="high"),
            _work_order(work_order_id="WO-4", status="ignored", event_level="low"),
            _work_order(work_order_id="WO-5", status="unassigned", event_level="medium"),
        ]
        staff = [_staff(1, 2), _staff(2, 4)]
        with (
            patch(
                "app.modules.agent.workflow.WorkOrderRepository.list_work_orders",
                return_value=orders,
            ),
            patch(
                "app.modules.agent.workflow.WorkOrderRepository.list_staff", return_value=staff
            ),
        ):
            result = AgentWorkflow.brief()

        self.assertTrue(result.success)
        data = result.data
        self.assertEqual(data["summary"]["total_work_orders"], 5)
        self.assertEqual(data["summary"]["unresolved_total"], 3)
        self.assertEqual(data["summary"]["resolved_total"], 2)
        self.assertEqual(data["summary"]["staff_max_workload"], 4)
        self.assertEqual(data["summary"]["staff_avg_workload"], 3.0)
        # 高优未解决仅含pending那单;中优未解决截断至10条以内
        self.assertEqual([i["id"] for i in data["high_priority_unresolved"]], ["WO-1"])
        self.assertEqual(len(data["medium_priority_unresolved"]), 2)
        self.assertEqual(data["by_stage"]["pending"], 1)

    def test_empty_dataset(self) -> None:
        with (
            patch(
                "app.modules.agent.workflow.WorkOrderRepository.list_work_orders", return_value=[]
            ),
            patch("app.modules.agent.workflow.WorkOrderRepository.list_staff", return_value=[]),
        ):
            result = AgentWorkflow.brief()

        self.assertTrue(result.success)
        self.assertEqual(result.data["summary"]["staff_avg_workload"], 0)
        self.assertEqual(result.data["summary"]["staff_max_workload"], 0)


class DispatchRecommendWorkflowTest(unittest.TestCase):
    def test_missing_work_order(self) -> None:
        with patch(
            "app.modules.agent.workflow.WorkOrderRepository.get_work_order", return_value=None
        ):
            result = AgentWorkflow.dispatch_recommend("999")

        self.assertFalse(result.success)

    def test_non_unassigned_order_is_rejected(self) -> None:
        with patch(
            "app.modules.agent.workflow.WorkOrderRepository.get_work_order",
            return_value=_work_order(status="pending"),
        ):
            result = AgentWorkflow.dispatch_recommend("144")

        self.assertFalse(result.success)
        self.assertIn("仅未派发", result.error)

    def test_top_k_ranked_by_workload_then_distance(self) -> None:
        staff = [
            _staff(1, 0, distance=2.0),
            _staff(2, 0, distance=0.5),
            _staff(3, 1),
            _staff(4, 0, category="road_maintenance", distance=0.1),
        ]
        with (
            patch(
                "app.modules.agent.workflow.WorkOrderRepository.get_work_order",
                return_value=_work_order(),
            ),
            patch(
                "app.modules.agent.workflow.WorkOrderRepository.list_staff", return_value=staff
            ),
        ):
            result = AgentWorkflow.dispatch_recommend("144", top_k=2)

        self.assertTrue(result.success)
        data = result.data
        # 仅同类别;负载相同按距离升序
        self.assertEqual([r["user_id"] for r in data["recommendations"]], [2, 1])
        self.assertEqual(data["total_candidates"], 3)
        self.assertEqual(data["recommendations"][0]["rank"], 1)


class ActionPlanTest(unittest.TestCase):
    def test_stage_and_level_specific_plans(self) -> None:
        high_unassigned = AgentWorkflow._build_action_plan("unassigned", "high")
        self.assertTrue(any("立即派发" in action for action in high_unassigned))

        low_completed = AgentWorkflow._build_action_plan("completed", "low")
        self.assertEqual(low_completed, ["归档工单"])

    def test_unknown_stage_falls_back_to_default(self) -> None:
        self.assertEqual(
            AgentWorkflow._build_action_plan("archived", "high"),
            ["参考标准处置流程", "根据现场情况灵活调整"],
        )

    def test_unknown_level_falls_back_to_medium(self) -> None:
        self.assertEqual(
            AgentWorkflow._build_action_plan("unassigned", "unknown"),
            AgentWorkflow._build_action_plan("unassigned", "medium"),
        )


if __name__ == "__main__":
    unittest.main()
