"""Agent API 层测试:对话、流式、工具列表、工作流端点错误映射。"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from app.api.v1.agent import (
    ChatRequest,
    android_chat,
    chat,
    clear_android_session,
    clear_session,
    health,
    list_tools,
    workflow_brief,
    workflow_diagnose,
    workflow_dispatch_recommend,
)


class ChatApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_chat_returns_reply_and_auto_thread_id(self) -> None:
        agent = SimpleNamespace(chat=AsyncMock(return_value="已派发。"))
        with patch("app.api.v1.agent.Agent", return_value=agent):
            response = await chat(ChatRequest(message="派发", thread_id=None))

        self.assertEqual(response.reply, "已派发。")
        self.assertTrue(response.thread_id.startswith("0000") or len(response.thread_id) > 10)
        agent.chat.assert_awaited_once()

    async def test_runtime_error_maps_to_503(self) -> None:
        agent = SimpleNamespace(chat=AsyncMock(side_effect=RuntimeError("llm overload")))
        with patch("app.api.v1.agent.Agent", return_value=agent):
            with self.assertRaises(HTTPException) as caught:
                await chat(ChatRequest(message="x"))

        self.assertEqual(caught.exception.status_code, 503)

    async def test_generic_exception_maps_to_500(self) -> None:
        agent = SimpleNamespace(chat=AsyncMock(side_effect=ValueError("unexpected")))
        with patch("app.api.v1.agent.Agent", return_value=agent):
            with self.assertRaises(HTTPException) as caught:
                await chat(ChatRequest(message="x"))

        self.assertEqual(caught.exception.status_code, 500)

    async def test_android_chat_error_mapping(self) -> None:
        assistant = SimpleNamespace(chat=AsyncMock(side_effect=RuntimeError("model down")))
        with patch("app.api.v1.agent.AndroidAssistant", return_value=assistant):
            with self.assertRaises(HTTPException) as caught:
                await android_chat(ChatRequest(message="工单144"))

        self.assertEqual(caught.exception.status_code, 503)


class HealthEndpointTest(unittest.IsolatedAsyncioTestCase):
    async def test_healthy(self) -> None:
        agent = SimpleNamespace(probe=lambda: {"ok": True, "model": "gpt4"}, tools=["t1"])
        with patch("app.api.v1.agent.Agent", return_value=agent):
            result = await health()

        self.assertEqual(result["status"], "healthy")
        self.assertEqual(result["model"], "gpt4")
        self.assertEqual(result["tools"], ["t1"])

    async def test_degraded_without_api_key(self) -> None:
        agent = SimpleNamespace(probe=lambda: {"ok": False, "detail": "API key not configured"}, tools=[])
        with patch("app.api.v1.agent.Agent", return_value=agent):
            result = await health()

        self.assertEqual(result["status"], "degraded")

    async def test_unhealthy_on_exception(self) -> None:
        with patch("app.api.v1.agent.Agent", side_effect=RuntimeError("init fail")):
            result = await health()

        self.assertEqual(result["status"], "unhealthy")
        self.assertIn("init fail", result["error"])


class SessionClearTest(unittest.IsolatedAsyncioTestCase):
    async def test_clear_session_ok(self) -> None:
        agent = SimpleNamespace(clear_memory=lambda tid: None)
        with patch("app.api.v1.agent.Agent", return_value=agent):
            result = await clear_session("thread-1")

        self.assertEqual(result, {"status": "ok", "thread_id": "thread-1"})

    async def test_clear_session_exception_maps_to_500(self) -> None:
        agent = SimpleNamespace(clear_memory=lambda tid: (_ for _ in ()).throw(RuntimeError("oops")))
        with patch("app.api.v1.agent.Agent", return_value=agent):
            with self.assertRaises(HTTPException) as caught:
                await clear_session("thread-1")

        self.assertEqual(caught.exception.status_code, 500)

    async def test_clear_android_session_ok(self) -> None:
        assistant = SimpleNamespace(clear_memory=lambda tid: None)
        with patch("app.api.v1.agent.AndroidAssistant", return_value=assistant):
            result = await clear_android_session("android-thread-1")

        self.assertEqual(result["thread_id"], "android-thread-1")


class ToolsEndpointTest(unittest.IsolatedAsyncioTestCase):
    async def test_lists_tools_from_agent(self) -> None:
        agent = SimpleNamespace(tools=["query_work_orders", "query_staff"])
        with patch("app.api.v1.agent.Agent", return_value=agent):
            result = await list_tools()

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "query_work_orders")


class WorkflowApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_diagnose_failure_returns_404(self) -> None:
        result = SimpleNamespace(success=False, error="工单不存在: 999")
        with patch("app.api.v1.agent.AgentWorkflow.diagnose", return_value=result):
            with self.assertRaises(HTTPException) as caught:
                await workflow_diagnose("999")

        self.assertEqual(caught.exception.status_code, 404)

    async def test_diagnose_success_returns_data(self) -> None:
        result = SimpleNamespace(success=True, data={"ok": True})
        with patch("app.api.v1.agent.AgentWorkflow.diagnose", return_value=result):
            response = await workflow_diagnose("144")

        self.assertEqual(response, {"ok": True})

    async def test_brief_failure_returns_500(self) -> None:
        result = SimpleNamespace(success=False, error="db down")
        with patch("app.api.v1.agent.AgentWorkflow.brief", return_value=result):
            with self.assertRaises(HTTPException) as caught:
                await workflow_brief()

        self.assertEqual(caught.exception.status_code, 500)

    async def test_dispatch_recommend_failure_returns_400(self) -> None:
        result = SimpleNamespace(success=False, error="工单状态非未派发")
        with patch("app.api.v1.agent.AgentWorkflow.dispatch_recommend", return_value=result):
            with self.assertRaises(HTTPException) as caught:
                await workflow_dispatch_recommend("144")

        self.assertEqual(caught.exception.status_code, 400)

    async def test_dispatch_recommend_success_forwards_top_k(self) -> None:
        result = SimpleNamespace(success=True, data={"recommendations": []})
        with patch("app.api.v1.agent.AgentWorkflow.dispatch_recommend", return_value=result) as wf:
            await workflow_dispatch_recommend("144", top_k=5)

        wf.assert_called_once_with("144", top_k=5)


if __name__ == "__main__":
    unittest.main()
