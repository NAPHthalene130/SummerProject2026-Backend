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
    # 工具只返回图片URL,不返回任何图片内容,助手不得声称已看过画面
    assert "data:image" not in json.dumps(payload)


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
    assert android_assistant.ANDROID_TOOL_NAMES == [
        "query_android_work_order",
        "search_traffic_regulations",
    ]
    assert android_assistant.ANDROID_TOOLS["query_android_work_order"] is android_assistant.query_android_work_order
    assert android_assistant.ANDROID_TOOLS["search_traffic_regulations"] is android_assistant.search_traffic_regulations
    assert set(android_assistant.ANDROID_TOOL_DESCRIPTIONS) == set(android_assistant.ANDROID_TOOL_NAMES)


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


class ParseToolCallTests(unittest.TestCase):
    def test_parses_well_formed_tool_call(self) -> None:
        text = '<tool_call>\n{"name": "query_android_work_order", "arguments": {"work_order_id": "144"}}\n</tool_call>'
        parsed = android_assistant._parse_tool_call(text)

        self.assertEqual(parsed["name"], "query_android_work_order")
        self.assertEqual(parsed["arguments"], {"work_order_id": "144"})

    def test_returns_none_without_tool_call_tags(self) -> None:
        self.assertIsNone(android_assistant._parse_tool_call("直接回答即可"))

    def test_returns_none_for_invalid_json(self) -> None:
        self.assertIsNone(android_assistant._parse_tool_call("<tool_call>{invalid}</tool_call>"))

    def test_truncate_result_only_when_over_limit(self) -> None:
        self.assertEqual(android_assistant._truncate_result("short"), "short")

        long_text = "x" * 4000
        truncated = android_assistant._truncate_result(long_text)
        self.assertTrue(truncated.endswith("[结果过长已截断]"))
        self.assertLess(len(truncated), len(long_text))


class AndroidConversationTests(unittest.TestCase):
    def test_build_prompt_contains_roles_and_tool_list(self) -> None:
        conversation = android_assistant.AndroidConversation()
        conversation.add_user("工单144怎么处理")
        conversation.add_assistant("正在查询")
        conversation.add_tool_result('{"data": {}}')

        prompt = conversation.build_prompt()

        self.assertIn("query_android_work_order", prompt)
        self.assertIn("用户: 工单144怎么处理", prompt)
        self.assertIn("助手: 正在查询", prompt)
        self.assertIn("工具返回: ", prompt)
        self.assertTrue(prompt.endswith("助手: "))

    def test_trim_keeps_only_recent_messages(self) -> None:
        conversation = android_assistant.AndroidConversation()
        for index in range(20):
            conversation.add_user(f"消息{index}")

        conversation.trim(keep=16)

        self.assertEqual(len(conversation.messages), 16)
        self.assertEqual(conversation.messages[0]["content"], "消息4")


class AndroidAssistantChatTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        android_assistant.AndroidAssistant._instance = None
        self.assistant = android_assistant.AndroidAssistant()
        self.thread_id = "pytest-android-chat"

    def tearDown(self) -> None:
        self.assistant.clear_memory(self.thread_id)

    @staticmethod
    def _llm_response(text: str) -> SimpleNamespace:
        return SimpleNamespace(content=text)

    async def test_chat_executes_tool_call_then_returns_final_answer(self) -> None:
        tool_call_text = (
            '<tool_call>{"name": "query_android_work_order", '
            '"arguments": {"work_order_id": "144"}}</tool_call>'
        )
        final_text = "1. 先设置警戒区域。2. 联系交警。"
        llm = SimpleNamespace(
            ainvoke=AsyncMock(side_effect=[self._llm_response(tool_call_text), self._llm_response(final_text)])
        )

        with (
            patch.object(self.assistant, "_llm", llm),
            patch.object(
                android_assistant.WorkOrderRepository,
                "get_work_order",
                return_value=_work_order(),
            ) as get_work_order,
        ):
            reply = await self.assistant.chat("工单144如何处理", thread_id=self.thread_id)

        self.assertEqual(reply, final_text)
        get_work_order.assert_called_once_with("144")
        self.assertEqual(llm.ainvoke.await_count, 2)
        conversation = self.assistant._sessions[self.thread_id]
        self.assertTrue(any(msg["role"] == "tool" for msg in conversation.messages))

    async def test_chat_reports_unknown_tool_and_continues(self) -> None:
        tool_call_text = '<tool_call>{"name": "delete_work_order", "arguments": {}}</tool_call>'
        final_text = "我只能查询工单和法规,无法执行删除。"
        llm = SimpleNamespace(
            ainvoke=AsyncMock(side_effect=[self._llm_response(tool_call_text), self._llm_response(final_text)])
        )

        with patch.object(self.assistant, "_llm", llm):
            reply = await self.assistant.chat("删除工单144", thread_id=self.thread_id)

        self.assertEqual(reply, final_text)
        conversation = self.assistant._sessions[self.thread_id]
        tool_messages = [msg["content"] for msg in conversation.messages if msg["role"] == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("未知工具", tool_messages[0])

    async def test_chat_returns_limit_message_after_max_react_turns(self) -> None:
        tool_call_text = '<tool_call>{"name": "search_traffic_regulations", "arguments": {"question": "x"}}</tool_call>'
        llm = SimpleNamespace(ainvoke=AsyncMock(return_value=self._llm_response(tool_call_text)))

        with (
            patch.object(self.assistant, "_llm", llm),
            patch.object(android_assistant, "_rag_manager", SimpleNamespace(search_rag=lambda _: {})),
        ):
            reply = await self.assistant.chat("一直检索", thread_id=self.thread_id)

        self.assertEqual(reply, "Android助手处理达到最大轮次限制,请简化问题后重试。")
        self.assertEqual(llm.ainvoke.await_count, android_assistant.MAX_REACT_TURNS)

    async def test_chat_wraps_llm_failure_in_runtime_error(self) -> None:
        llm = SimpleNamespace(ainvoke=AsyncMock(side_effect=ConnectionError("llm unreachable")))

        with patch.object(self.assistant, "_llm", llm):
            with self.assertRaises(RuntimeError):
                await self.assistant.chat("你好", thread_id=self.thread_id)

    async def test_astream_yields_tool_events_and_tokens(self) -> None:
        tool_call_text = (
            '<tool_call>{"name": "search_traffic_regulations", '
            '"arguments": {"question": "事故处理"}}</tool_call>'
        )
        final_text = "应当立即停车保护现场。"
        llm = SimpleNamespace(
            ainvoke=AsyncMock(side_effect=[self._llm_response(tool_call_text), self._llm_response(final_text)])
        )

        events: list[dict] = []
        with (
            patch.object(self.assistant, "_llm", llm),
            patch.object(android_assistant, "_rag_manager", SimpleNamespace(search_rag=lambda _: {})),
        ):
            async for event in self.assistant.astream("事故后怎么办", thread_id=self.thread_id):
                events.append(event)

        event_types = [event["type"] for event in events]
        self.assertEqual(event_types[0], "tool_start")
        self.assertEqual(event_types[1], "tool_end")
        self.assertEqual(events[0]["name"], "search_traffic_regulations")
        token_text = "".join(event["content"] for event in events if event["type"] == "token")
        self.assertEqual(token_text, final_text)

    async def test_astream_llm_failure_yields_error_event(self) -> None:
        llm = SimpleNamespace(ainvoke=AsyncMock(side_effect=ConnectionError("down")))

        events: list[dict] = []
        with patch.object(self.assistant, "_llm", llm):
            async for event in self.assistant.astream("你好", thread_id=self.thread_id):
                events.append(event)

        self.assertEqual(events, [{"type": "error", "content": "Android助手暂时不可用，请稍后重试。"}])

    def test_clear_memory_drops_session(self) -> None:
        self.assistant._sessions[self.thread_id] = android_assistant.AndroidConversation()

        self.assistant.clear_memory(self.thread_id)

        self.assertNotIn(self.thread_id, self.assistant._sessions)


class RegulationToolTests(unittest.TestCase):
    def test_empty_question_returns_error_envelope(self) -> None:
        payload = json.loads(android_assistant.search_traffic_regulations("   "))

        self.assertIn("error", payload)
        self.assertEqual(payload["references"], [])

    def test_rag_failure_returns_error_envelope(self) -> None:
        manager = SimpleNamespace(search_rag=lambda _: (_ for _ in ()).throw(RuntimeError("chroma down")))
        with patch.object(android_assistant, "_rag_manager", manager):
            payload = json.loads(android_assistant.search_traffic_regulations("酒驾处罚"))

        self.assertIn("检索失败", payload["error"])
        self.assertEqual(payload["references"], [])

    def test_work_order_tool_wraps_repository_failure(self) -> None:
        with patch.object(
            android_assistant.WorkOrderRepository,
            "get_work_order",
            side_effect=RuntimeError("db down"),
        ):
            payload = json.loads(android_assistant.query_android_work_order("144"))

        self.assertIn("查询失败", payload["error"])
        self.assertIsNone(payload["data"])


if __name__ == "__main__":
    unittest.main()
