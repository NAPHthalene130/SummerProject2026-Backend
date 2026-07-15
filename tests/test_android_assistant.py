import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.api.v1.agent import ChatRequest, android_chat
from app.modules.agent import android_assistant


def _work_order() -> SimpleNamespace:
    return SimpleNamespace(
        work_order_id="WO-20260712-144",
        event_id="EV-144",
        camera_id="live30",
        camera_name="摄像头 live30",
        segment_id="SEG-30",
        segment_name="测试路段",
        monitor_address="测试路口",
        accident_info="车辆碰撞",
        description="两车发生碰撞并占用一条车道。",
        event_time="2026-07-12 16:00:51",
        event_level="high",
        status="pending",
        assignee="张三",
        required_category="traffic_police",
        ai_suggestion="先设置警戒区域。",
        scene_info="晚高峰，路面湿滑。",
        scene_images=["/orderImg/scene-144.jpg"],
        process_message=None,
        process_images=["/orderImg/result-144.jpg"],
        completed_at=None,
    )


def test_android_work_order_tool_returns_description_and_image_urls() -> None:
    with patch.object(
        android_assistant.WorkOrderRepository,
        "get_work_order",
        return_value=_work_order(),
    ):
        payload = json.loads(android_assistant.query_android_work_order("144"))

    assert payload["source_type"] == "work_order_database"
    assert payload["data"]["event_description"] == "两车发生碰撞并占用一条车道。"
    assert payload["data"]["scene_image_urls"] == ["/orderImg/scene-144.jpg"]
    assert payload["data"]["process_image_urls"] == ["/orderImg/result-144.jpg"]
    assert "未读取或核验" in payload["image_notice"]


def test_android_work_order_tool_reports_missing_order() -> None:
    with patch.object(
        android_assistant.WorkOrderRepository,
        "get_work_order",
        return_value=None,
    ):
        payload = json.loads(android_assistant.query_android_work_order("999"))

    assert payload["data"] is None
    assert "未找到工单" in payload["error"]


def test_regulation_tool_returns_rag_references() -> None:
    manager = SimpleNamespace(
        search_rag=lambda _: {
            "中华人民共和国道路交通安全法": [
                {
                    "chapter": "第五章第七十条",
                    "describe": "发生交通事故后应当立即停车。",
                }
            ]
        }
    )
    with patch.object(android_assistant, "_rag_manager", manager):
        payload = json.loads(android_assistant.search_traffic_regulations("事故后如何处理"))

    assert payload["references"] == [
        {
            "source_type": "regulation_knowledge_base",
            "title": "中华人民共和国道路交通安全法",
            "article": "第五章第七十条",
            "excerpt": "发生交通事故后应当立即停车。",
        }
    ]


def test_android_assistant_exposes_exactly_two_read_only_tools() -> None:
    assert [tool.__name__ for tool in android_assistant.ANDROID_ASSISTANT_TOOLS] == [
        "query_android_work_order",
        "search_traffic_regulations",
    ]


class AndroidAssistantToolTests(unittest.TestCase):
    def test_work_order_context(self) -> None:
        test_android_work_order_tool_returns_description_and_image_urls()

    def test_missing_work_order(self) -> None:
        test_android_work_order_tool_reports_missing_order()

    def test_regulation_references(self) -> None:
        test_regulation_tool_returns_rag_references()

    def test_exact_tool_boundary(self) -> None:
        test_android_assistant_exposes_exactly_two_read_only_tools()


class AndroidAssistantApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_android_chat_returns_reply_and_thread_id(self) -> None:
        assistant = SimpleNamespace(chat=AsyncMock(return_value="请先设置现场警戒。"))
        with patch("app.api.v1.agent.AndroidAssistant", return_value=assistant):
            response = await android_chat(
                ChatRequest(message="工单144如何处理", thread_id="android-thread-1")
            )

        self.assertEqual(response.reply, "请先设置现场警戒。")
        self.assertEqual(response.thread_id, "android-thread-1")
        assistant.chat.assert_awaited_once_with(
            "工单144如何处理",
            thread_id="android-thread-1",
        )


if __name__ == "__main__":
    unittest.main()
