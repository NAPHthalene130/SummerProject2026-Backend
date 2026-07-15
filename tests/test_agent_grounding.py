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


def test_work_order_detail_marks_file_metadata_as_unverified_content() -> None:
    work_order = _work_order()
    with patch.object(tool.WorkOrderRepository, "get_work_order", return_value=work_order):
        payload = json.loads(tool.get_work_order_detail(work_order.work_order_id))

    assert payload["evidence"]["entity"] == "work_order_detail"
    assert payload["result_evidence"]["result_is_database_recorded"] is True
    assert payload["result_evidence"]["result_content_independently_verified"] is False
    assert payload["file_evidence"]["process_image_urls"] == ["/orderImg/result-144.jpg"]
    assert payload["file_evidence"]["content_verified"] is False
    assert payload["file_evidence"]["existence_verified"] is False


def test_work_order_detail_does_not_claim_missing_result_evidence() -> None:
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

    assert payload["result_evidence"]["result_is_database_recorded"] is False
    assert payload["file_evidence"]["process_image_count"] == 0
    assert payload["file_evidence"]["process_image_urls"] == []


def test_regulation_tool_returns_structured_citations() -> None:
    search_result = {
        "中华人民共和国道路交通安全法": [
            {"chapter": "第五章第七十条", "describe": "发生交通事故后应当立即停车。"}
        ]
    }
    with patch.object(tool, "_search_regulations", return_value=search_result):
        payload = json.loads(tool.answer_general_question("发生事故后如何处理"))

    assert payload["evidence"]["matched_reference_count"] == 1
    assert payload["rag_references"] == [
        {
            "source_type": "regulation_knowledge_base",
            "title": "中华人民共和国道路交通安全法",
            "article": "第五章第七十条",
            "excerpt": "发生交通事故后应当立即停车。",
        }
    ]


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

    assert payload["evidence"]["source_type"] == "database"
    assert payload["rag_references"][0]["source_type"] == "regulation_knowledge_base"
    assert payload["file_evidence"]["content_verified"] is False


class AgentGroundingTests(unittest.TestCase):
    def test_detail_file_metadata_boundary(self) -> None:
        test_work_order_detail_marks_file_metadata_as_unverified_content()

    def test_missing_result_evidence(self) -> None:
        test_work_order_detail_does_not_claim_missing_result_evidence()

    def test_structured_regulation_citations(self) -> None:
        test_regulation_tool_returns_structured_citations()

    def test_database_and_rag_evidence_are_separate(self) -> None:
        test_suggest_handling_keeps_database_and_rag_evidence_separate()
