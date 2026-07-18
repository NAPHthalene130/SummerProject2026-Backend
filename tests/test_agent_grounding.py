import json
import unittest
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
        "status": "completed",
        "assignee": "张三",
        "assignee_user_id": 7,
        "description": "两车发生碰撞。",
        "ai_suggestion": "先设置警戒区。",
        "scene_info": "晚高峰，路面湿滑。",
        "process_message": "现场已完成清障。",
        "completed_at": "2026-07-12 16:24:00",
        "required_category": "traffic_police",
        "scene_images": ["/orderImg/scene-144.jpg"],
        "process_images": ["/orderImg/result-144.jpg"],
        "feedback_review_status": "approved",
        "feedback_review_message": "材料完整。",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_work_order_detail_exposes_file_and_result_flags() -> None:
    """详情工具必须暴露图片数量与处置结果存在性标记,供Agent区分数据库记录与未核验内容。"""
    work_order = _work_order()
    with patch.object(tool.WorkOrderRepository, "get_work_order", return_value=work_order):
        payload = json.loads(tool.get_work_order_detail(work_order.work_order_id))

    # 数据库记录字段原样返回
    assert payload["work_order_id"] == "WO-20260712-144"
    assert payload["description"] == "两车发生碰撞。"
    assert payload["process_message"] == "现场已完成清障。"
    # 文件/结果存在性标记:Agent 据此判断哪些内容有数据库记录支撑
    assert payload["scene_image_count"] == 1
    assert payload["process_image_count"] == 1
    assert payload["has_process_message"] is True
    assert payload["has_completed_at"] is True
    # 工具只返回图片计数,不返回图片内容本身
    assert "data:image" not in json.dumps(payload)


def test_work_order_detail_does_not_claim_missing_result_evidence() -> None:
    """未完成工单没有处置记录,工具不得返回暗示已有处置结果的字段值。"""
    work_order = _work_order(
        status="pending",
        process_message=None,
        process_images=None,
        completed_at=None,
        feedback_review_status="none",
        feedback_review_message=None,
    )
    with patch.object(tool.WorkOrderRepository, "get_work_order", return_value=work_order):
        payload = json.loads(tool.get_work_order_detail(work_order.work_order_id))

    assert payload["process_message"] is None
    assert payload["process_image_count"] == 0
    assert payload["has_process_message"] is False
    assert payload["has_completed_at"] is False


def test_regulation_tool_returns_structured_citations() -> None:
    search_result = {
        "中华人民共和国道路交通安全法": [
            {"chapter": "第五章第七十条", "describe": "发生交通事故后应当立即停车。"}
        ]
    }
    with patch.object(tool, "_search_regulations", return_value=search_result):
        payload = json.loads(tool.answer_general_question("发生事故后如何处理"))

    assert payload["rag_references"] == [
        {
            "law": "中华人民共和国道路交通安全法",
            "article": "第五章第七十条",
            "excerpt": "发生交通事故后应当立即停车。",
        }
    ]
    # 工具必须提示Agent:仅可引用实际检索到的条款
    assert "rag_references" in payload["note"]


def test_suggest_handling_keeps_database_and_rag_evidence_separate() -> None:
    work_order = _work_order(status="pending")
    with (
        patch.object(tool.WorkOrderRepository, "get_work_order", return_value=work_order),
        patch.object(
            tool,
            "_search_regulations",
            return_value={
                "中华人民共和国道路交通安全法实施条例": [
                    {"chapter": "第一章第一条", "describe": "依据道路交通安全法制定本条例。"}
                ]
            },
        ),
    ):
        payload = json.loads(tool.suggest_handling(work_order.work_order_id))

    # 数据库来源字段与RAG检索依据分属不同键,不得混排
    assert payload["work_order_id"] == "WO-20260712-144"
    assert payload["accident_type"] == "车辆碰撞"
    assert payload["current_stage"] == "pending"
    assert payload["rag_references"] == [
        {
            "law": "中华人民共和国道路交通安全法实施条例",
            "article": "第一章第一条",
            "excerpt": "依据道路交通安全法制定本条例。",
        }
    ]
    # 现场照片文件不存在于磁盘时,不得编造图像分析结论
    assert payload["image_analysis"] == ""
    assert payload["stage_guidance"]


class AgentGroundingTests(unittest.TestCase):
    def test_detail_file_metadata_boundary(self) -> None:
        test_work_order_detail_exposes_file_and_result_flags()

    def test_missing_result_evidence(self) -> None:
        test_work_order_detail_does_not_claim_missing_result_evidence()

    def test_structured_regulation_citations(self) -> None:
        test_regulation_tool_returns_structured_citations()

    def test_database_and_rag_evidence_are_separate(self) -> None:
        test_suggest_handling_keeps_database_and_rag_evidence_separate()


if __name__ == "__main__":
    unittest.main()
