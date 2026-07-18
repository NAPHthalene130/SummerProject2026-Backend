"""Agent 核心模块测试:ReAct 对话、工具解析、探活与同步执行。"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.modules.agent.agent import (
    TOOLS,
    Agent,
    ReActConversation,
    _build_tool_list_text,
    _execute_tool_sync,
    _parse_tool_call,
    _truncate_tool_result,
)


class ParseToolCallTest(unittest.TestCase):
    def test_parses_json_within_tags(self) -> None:
        text = '<tool_call>\n{"name": "query_work_orders", "arguments": {"limit": 10}}\n</tool_call>'
        parsed = _parse_tool_call(text)

        self.assertEqual(parsed["name"], "query_work_orders")
        self.assertEqual(parsed["arguments"], {"limit": 10})

    def test_no_tags_returns_none(self) -> None:
        self.assertIsNone(_parse_tool_call("直接回答"))

    def test_invalid_json_returns_none(self) -> None:
        self.assertIsNone(_parse_tool_call("<tool_call>{broken}</tool_call>"))

    def test_multiline_json_with_whitespace(self) -> None:
        text = '<tool_call>  \n {  "name" : "t" , "arguments" : {} } \n  </tool_call>'
        parsed = _parse_tool_call(text)
        self.assertEqual(parsed["name"], "t")


class TruncateToolResultTest(unittest.TestCase):
    def test_truncation(self) -> None:
        self.assertEqual(_truncate_tool_result("short"), "short")
        long_text = "x" * 4000
        truncated = _truncate_tool_result(long_text)
        self.assertTrue(truncated.endswith("[结果过长已截断]"))
        self.assertLess(len(truncated), len(long_text))


class BuildToolListTextTest(unittest.TestCase):
    def test_includes_all_tool_descriptions(self) -> None:
        text = _build_tool_list_text()

        for name in [t.__name__ for t in TOOLS]:
            self.assertIn(name, text)


class ExecuteToolSyncTest(unittest.TestCase):
    def test_unknown_tool_returns_error_envelope(self) -> None:
        result = _execute_tool_sync("no_such_tool", {})

        self.assertIn("未知工具", result)


class ReActConversationTest(unittest.TestCase):
    def test_build_prompt_contains_roles_and_last_twenty_messages(self) -> None:
        conv = ReActConversation()
        conv.add_user("派发工单")
        conv.add_assistant("正在查询")
        conv.add_tool_result(json.dumps({"staff": []}))

        prompt = conv.build_prompt()

        self.assertIn("用户: 派发工单", prompt)
        self.assertIn("助手: 正在查询", prompt)
        self.assertIn("工具返回: ", prompt)
        self.assertTrue(prompt.endswith("助手: "))

    def test_trim_drops_old_messages(self) -> None:
        conv = ReActConversation()
        for index in range(25):
            conv.add_user(f"msg-{index}")

        conv.trim(keep=20)

        self.assertEqual(len(conv.messages), 20)
        self.assertEqual(conv.messages[0]["content"], "msg-5")


class AgentProbeTest(unittest.TestCase):
    def test_probe_degraded_with_no_api_key(self) -> None:
        from app.config import llm_settings

        saved_instance = Agent._instance
        saved_key = llm_settings.api_key
        Agent._instance = None

        try:
            llm_settings.api_key = "your_api_key_here"
            agent = Agent()
            result = agent.probe()
            self.assertFalse(result["ok"])
            self.assertIn("API key is not configured", result["detail"])
        finally:
            llm_settings.api_key = saved_key
            Agent._instance = saved_instance


class AgentChatSyncTest(unittest.TestCase):
    def test_chat_sync_with_no_llm_returns_message(self) -> None:
        agent = Agent()
        agent._llm = None
        result = agent.chat_sync("你好")
        self.assertIn("未初始化", result)

    def test_chat_sync_executes_tool_call_and_returns_final_answer(self) -> None:
        agent = Agent()
        agent._sessions.clear()

        query_result = json.dumps({"staff": [{"id": "1", "name": "张三"}]}, ensure_ascii=False)
        llm = SimpleNamespace(
            invoke=lambda prompt: SimpleNamespace(
                content='<tool_call>{"name": "query_staff", "arguments": {}}</tool_call>'
                if "query_staff" in prompt
                else SimpleNamespace(content="可用人员：张三。")
            )
        )

        with (
            patch.object(agent, "_llm", llm),
            patch("app.modules.agent.agent._execute_tool_sync", return_value=query_result),
        ):
            result = agent.chat_sync("查询人员", thread_id="sync-thread-1")

        self.assertEqual(result, str(llm.invoke("").content))
        agent.clear_memory("sync-thread-1")


class AgentChatMaxTurnsTest(unittest.IsolatedAsyncioTestCase):
    async def test_max_turns_limit_is_reached(self) -> None:
        agent = Agent()
        agent._sessions.clear()
        agent._llm = SimpleNamespace(
            ainvoke=AsyncMock(return_value=SimpleNamespace(
                content='<tool_call>{"name": "query_staff", "arguments": {}}</tool_call>'
            ))
        )

        with patch("app.modules.agent.agent._execute_tool", return_value='{"staff": []}'):
            result = await agent.chat("查询", thread_id="turn-test")

        self.assertIn("最大轮次限制", result)
        agent.clear_memory("turn-test")


class AgentStreamTest(unittest.IsolatedAsyncioTestCase):
    async def test_astream_no_llm_yields_error(self) -> None:
        agent = Agent()
        agent._llm = None

        events = []
        async for evt in agent.astream("你好"):
            events.append(evt)

        self.assertEqual(events[0]["type"], "error")
        self.assertIn("未初始化", events[0]["content"])

    async def test_astream_yields_tool_events_and_tokens(self) -> None:
        agent = Agent()
        agent._sessions.clear()

        llm = SimpleNamespace(
            ainvoke=AsyncMock(side_effect=[
                SimpleNamespace(content='<tool_call>{"name": "query_staff", "arguments": {}}</tool_call>'),
                SimpleNamespace(content="可用3名人员。"),
            ])
        )

        events = []
        with (
            patch.object(agent, "_llm", llm),
            patch("app.modules.agent.agent._execute_tool", return_value='{"staff": []}'),
        ):
            async for evt in agent.astream("查询", thread_id="stream-test"):
                events.append(evt)

        types = [evt["type"] for evt in events]
        self.assertIn("tool_start", types)
        self.assertIn("tool_end", types)
        self.assertIn("token", types)

        agent.clear_memory("stream-test")


if __name__ == "__main__":
    unittest.main()
