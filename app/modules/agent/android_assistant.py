import json
import logging
from typing import Any, AsyncIterator, Optional

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from app.config import llm_settings
from app.modules.agent.base import BaseAgent
from app.modules.agent.rag.rag_manager import RagManager
from app.repository.work_order_repository import WorkOrderRepository

logger = logging.getLogger(__name__)

_rag_manager: Optional[RagManager] = None


def query_android_work_order(work_order_id: str) -> str:
    """按工单ID查询数据库中的工单辅助信息。

    work_order_id: 工单编号，例如 WO-20260712-144、144。返回事件描述、地点、
    等级、状态、现场和处理图片URL、已有AI建议与处理记录；本工具只读，不修改工单。
    """
    try:
        work_order = WorkOrderRepository.get_work_order(work_order_id)
    except (TypeError, ValueError):
        return json.dumps(
            {"error": f"无效的工单ID: {work_order_id}", "data": None},
            ensure_ascii=False,
        )
    except Exception as exc:
        logger.exception("Android助手查询工单失败: work_order_id=%s", work_order_id)
        return json.dumps(
            {"error": f"工单数据库查询失败: {exc}", "data": None},
            ensure_ascii=False,
        )

    if work_order is None:
        return json.dumps(
            {"error": f"未找到工单: {work_order_id}", "data": None},
            ensure_ascii=False,
        )

    data = {
        "work_order_id": work_order.work_order_id,
        "event_id": work_order.event_id,
        "camera_id": work_order.camera_id,
        "camera_name": work_order.camera_name,
        "segment_id": work_order.segment_id,
        "segment_name": work_order.segment_name,
        "monitor_address": work_order.monitor_address,
        "event_type": work_order.accident_info,
        "event_description": work_order.description,
        "event_time": work_order.event_time,
        "event_level": work_order.event_level,
        "status": work_order.status,
        "assignee": work_order.assignee,
        "required_category": work_order.required_category,
        "ai_suggestion": work_order.ai_suggestion,
        "scene_info": work_order.scene_info,
        "scene_image_urls": list(work_order.scene_images),
        "process_message": work_order.process_message,
        "process_image_urls": list(work_order.process_images or []),
        "completed_at": work_order.completed_at,
    }
    return json.dumps(
        {
            "source_type": "work_order_database",
            "data": data,
            "image_notice": "图片字段仅为数据库记录的URL，本工具未读取或核验图片内容。",
        },
        ensure_ascii=False,
    )


def search_traffic_regulations(question: str) -> str:
    """从RAG交通规定知识库检索与问题相关的法条。

    question: 用户的交通法规或现场处置问题。返回法规名称、条款编号和法条摘要，
    仅供生成建议时引用，不执行工单操作。
    """
    global _rag_manager

    query = question.strip()
    if not query:
        return json.dumps(
            {"error": "法规检索问题不能为空", "references": []},
            ensure_ascii=False,
        )

    try:
        if _rag_manager is None:
            _rag_manager = RagManager()
        search_result = _rag_manager.search_rag(query)
    except Exception as exc:
        logger.exception("Android助手检索交通法规失败")
        return json.dumps(
            {"error": f"交通规定知识库检索失败: {exc}", "references": []},
            ensure_ascii=False,
        )

    references = [
        {
            "source_type": "regulation_knowledge_base",
            "title": title,
            "article": item.get("chapter", ""),
            "excerpt": item.get("describe", ""),
        }
        for title, items in search_result.items()
        for item in items
    ]
    return json.dumps(
        {"query": query, "references": references},
        ensure_ascii=False,
    )


ANDROID_ASSISTANT_TOOLS = [
    query_android_work_order,
    search_traffic_regulations,
]

SYSTEM_PROMPT = """你是 Android 端的智能交通助手，为交通巡检和现场处置人员提供简洁、可执行的中文建议。

你只有两个只读工具：
1. query_android_work_order：用户提到具体工单ID时，必须先查询该工单，不得猜测工单内容。
2. search_traffic_regulations：用户询问法规、责任、处罚、通行规定或处置方法时，应检索交通规定知识库；针对工单生成处置建议时，也应按需检索法规作为补充依据。

回答规则：
- 可以同时使用工单数据库信息和法规知识库信息辅助回答，但必须区分两类来源。
- 工单工具返回的图片只是一组URL；你没有看到图片内容，不得声称已根据图片判断现场情况。
- 工单不存在、工具失败或法规未命中时，明确说明信息不足，并给出安全的通用建议，禁止编造数据、法条或处理结果。
- 建议优先覆盖人员安全、现场警戒、交通疏导、证据留存、部门联动和后续反馈；紧急情况提示立即联系交警、消防或急救部门。
- 不执行派单、结案、处罚等操作，也不得声称已经执行。
- 回答通常控制在 3 至 6 条建议。引用法规时写明法规名称和条款编号；不要展示内部JSON键名。
"""


class AndroidAssistant(BaseAgent):
    """供 Android 客户端调用的只读智能交通助手。"""

    _instance: Optional["AndroidAssistant"] = None

    def __new__(cls) -> "AndroidAssistant":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        self._llm: Optional[ChatOpenAI] = ChatOpenAI(
            base_url=llm_settings.url,
            api_key=llm_settings.api_key,  # type: ignore[arg-type]
            model=llm_settings.model_name,
            temperature=0.2,
            max_tokens=1536,
            timeout=30,
            max_retries=2,
        )
        self._memory: Optional[MemorySaver] = MemorySaver()
        self._graph: Optional[CompiledStateGraph] = create_agent(
            model=self._llm,
            tools=ANDROID_ASSISTANT_TOOLS,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=self._memory,
        )
        logger.info("AndroidAssistant初始化完成")

    @staticmethod
    def _build_config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    async def chat(self, user_message: str, thread_id: str = "default") -> str:
        if not self._graph:
            return "Android助手未初始化，请检查LLM配置。"
        try:
            result = await self._graph.ainvoke(
                {"messages": [HumanMessage(content=user_message)]},
                config=self._build_config(thread_id),
            )
            messages: list[Any] = result.get("messages", [])
            if messages:
                return str(messages[-1].content)
            return "Android助手未返回有效结果。"
        except Exception as exc:
            logger.exception("Android助手对话异常")
            raise RuntimeError("Android助手暂时不可用，请检查模型和检索服务。") from exc

    async def astream(
        self,
        user_message: str,
        thread_id: str = "default",
    ) -> AsyncIterator[dict[str, Any]]:
        if not self._graph:
            yield {"type": "error", "content": "Android助手未初始化，请检查LLM配置。"}
            return

        try:
            async for event in self._graph.astream_events(
                {"messages": [HumanMessage(content=user_message)]},
                config=self._build_config(thread_id),
                version="v2",
            ):
                kind = event.get("event", "")
                data = event.get("data") or {}
                if kind == "on_tool_start":
                    yield {"type": "tool_start", "name": event.get("name", "")}
                elif kind == "on_tool_end":
                    yield {"type": "tool_end", "name": event.get("name", "")}
                elif kind == "on_chat_model_stream":
                    content = getattr(data.get("chunk"), "content", "")
                    if isinstance(content, str) and content:
                        yield {"type": "token", "content": content}
        except Exception:
            logger.exception("Android助手流式对话异常")
            yield {"type": "error", "content": "Android助手暂时不可用，请稍后重试。"}

    def clear_memory(self, thread_id: str = "default") -> None:
        if not self._memory or not thread_id:
            return
        try:
            self._memory.delete_thread(thread_id)
        except Exception:
            logger.exception("清除Android助手会话失败: thread_id=%s", thread_id)

    @property
    def tools(self) -> list[str]:
        return [tool.__name__ for tool in ANDROID_ASSISTANT_TOOLS]

    def probe(self) -> dict[str, Any]:
        if llm_settings.api_key in {"", "your_api_key_here"}:
            return {
                "ok": False,
                "model": llm_settings.model_name,
                "detail": "LLM API key is not configured",
            }
        if not self._llm:
            return {"ok": False, "detail": "LLM未初始化"}
        try:
            self._llm.invoke("ping", config={"tags": ["android-assistant-health-probe"]})
            return {"ok": True, "model": llm_settings.model_name}
        except Exception as exc:
            return {"ok": False, "model": llm_settings.model_name, "detail": str(exc)}


# 兼容需求描述中的 android_assisant 拼写，同时对外推荐使用 AndroidAssistant。
AndroidAssisant = AndroidAssistant
android_assisant = AndroidAssistant
