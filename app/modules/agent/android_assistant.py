import json
import logging
import re
from typing import Any, AsyncIterator, Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from app.config import llm_settings
from app.modules.agent.base import BaseAgent
from app.modules.agent.rag.rag_manager import RagManager
from app.repository.work_order_repository import WorkOrderRepository

logger = logging.getLogger(__name__)

MAX_REACT_TURNS = 8

_rag_manager: Optional[RagManager] = None


def query_android_work_order(work_order_id: str) -> str:
    """按工单ID查询数据库中的工单辅助信息。"""
    try:
        work_order = WorkOrderRepository.get_work_order(work_order_id)
    except (TypeError, ValueError):
        return json.dumps({"error": f"无效的工单ID: {work_order_id}", "data": None}, ensure_ascii=False)
    except Exception as exc:
        logger.exception("Android助手查询工单失败: work_order_id=%s", work_order_id)
        return json.dumps({"error": f"工单数据库查询失败: {exc}", "data": None}, ensure_ascii=False)

    if work_order is None:
        return json.dumps({"error": f"未找到工单: {work_order_id}", "data": None}, ensure_ascii=False)

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
    return json.dumps({"source_type": "work_order_database", "data": data}, ensure_ascii=False)


def search_traffic_regulations(question: str) -> str:
    """从RAG交通规定知识库检索与问题相关的法条。"""
    global _rag_manager
    query = question.strip()
    if not query:
        return json.dumps({"error": "法规检索问题不能为空", "references": []}, ensure_ascii=False)

    try:
        if _rag_manager is None:
            _rag_manager = RagManager()
        search_result = _rag_manager.search_rag(query)
    except Exception as exc:
        logger.exception("Android助手检索交通法规失败")
        return json.dumps({"error": f"交通规定知识库检索失败: {exc}", "references": []}, ensure_ascii=False)

    references = [
        {"source_type": "regulation_knowledge_base", "title": title, "article": item.get("chapter", ""), "excerpt": item.get("describe", "")}
        for title, items in search_result.items()
        for item in items
    ]
    return json.dumps({"query": query, "references": references}, ensure_ascii=False)


ANDROID_TOOLS = {
    "query_android_work_order": query_android_work_order,
    "search_traffic_regulations": search_traffic_regulations,
}

ANDROID_TOOL_NAMES = list(ANDROID_TOOLS.keys())

ANDROID_TOOL_DESCRIPTIONS = {
    "query_android_work_order": "查询工单详情,参数: work_order_id(str) - 工单ID",
    "search_traffic_regulations": "检索交通法规知识库,参数: question(str) - 法规问题",
}

SYSTEM_PROMPT = """你是Android端的智能交通助手，为交通巡检和现场处置人员提供简洁、可执行的中文建议。

## 可用工具
{tool_list}

## 工具调用格式
当需要调用工具时,严格按以下JSON格式输出(不要加任何其他文字):

<tool_call>
{{"name": "工具名", "arguments": {{"参数名": "参数值"}}}}
</tool_call>

工具执行后你会收到结果,然后可以继续调用其他工具或给出最终回答。

## 回答规则
- 你只有两个只读工具: query_android_work_order(按ID查工单) 和 search_traffic_regulations(检索交通法规)
- 用户提到具体工单ID时,必须先查询该工单,不得猜测工单内容
- 用户询问法规、责任、处罚、通行规定或处置方法时,应检索交通规定知识库
- 可以同时使用工单数据库信息和法规知识库信息辅助回答,但必须区分两类来源
- 工单工具返回的图片只是一组URL;你没有看到图片内容,不得声称已根据图片判断现场情况
- 工单不存在、工具失败或法规未命中时,明确说明信息不足,并给出安全的通用建议,禁止编造数据、法条或处理结果
- 建议优先覆盖人员安全、现场警戒、交通疏导、证据留存、部门联动和后续反馈;紧急情况提示立即联系交警、消防或急救部门
- 不执行派单、结案、处罚等操作,也不得声称已经执行
- 回答通常控制在3至6条建议。引用法规时写明法规名称和条款编号;不要展示内部JSON键名"""


def _parse_tool_call(text: str) -> Optional[dict[str, Any]]:
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        logger.warning("Android assistant failed to parse tool call JSON: %s", m.group(1)[:200])
        return None


def _truncate_result(result: str, max_chars: int = 3000) -> str:
    if len(result) <= max_chars:
        return result
    return result[:max_chars] + "\n…[结果过长已截断]"


class AndroidConversation:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_result(self, content: str) -> None:
        self.messages.append({"role": "tool", "content": content})

    def build_prompt(self) -> str:
        lines = [f"- {name}: {desc}" for name, desc in ANDROID_TOOL_DESCRIPTIONS.items()]
        tool_list = "\n".join(lines)
        header = SYSTEM_PROMPT.format(tool_list=tool_list) + "\n\n## 对话历史\n"
        parts: list[str] = [header]
        for msg in self.messages[-16:]:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                parts.append(f"用户: {content}")
            elif role == "assistant":
                parts.append(f"助手: {content}")
            elif role == "tool":
                parts.append(f"工具返回: {content}")
        parts.append("助手: ")
        return "\n\n".join(parts)

    def trim(self, keep: int = 16) -> None:
        if len(self.messages) > keep:
            self.messages = self.messages[-keep:]


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

        self._llm: Optional[ChatOpenAI] = None
        self._sessions: dict[str, AndroidConversation] = {}
        self._init_llm()
        logger.info("AndroidAssistant(ReAct)初始化完成")

    def _init_llm(self) -> None:
        self._llm = ChatOpenAI(
            base_url=llm_settings.url,
            api_key=llm_settings.api_key,  # type: ignore[arg-type]
            model=llm_settings.model_name,
            temperature=0.2,
            max_tokens=1536,
            timeout=60,
            max_retries=2,
        )

    def _get_conversation(self, thread_id: str) -> AndroidConversation:
        if thread_id not in self._sessions:
            self._sessions[thread_id] = AndroidConversation()
        return self._sessions[thread_id]

    async def chat(self, user_message: str, thread_id: str = "default") -> str:
        if not self._llm:
            return "Android助手未初始化，请检查LLM配置。"

        conv = self._get_conversation(thread_id)
        conv.add_user(user_message)

        try:
            for turn in range(MAX_REACT_TURNS):
                prompt = conv.build_prompt()
                response = await self._llm.ainvoke([HumanMessage(content=prompt)])
                text = str(response.content).strip() if response.content else ""

                if not text:
                    return "Android助手未返回有效结果。"

                tool_call = _parse_tool_call(text)
                if tool_call:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("arguments", {})
                    if not isinstance(tool_args, dict):
                        tool_args = {}
                    clean_args = {}
                    if isinstance(tool_args, dict):
                        for k, v in tool_args.items():
                            clean_args[str(k)] = v

                    conv.add_assistant(text)
                    tool_fn = ANDROID_TOOLS.get(tool_name)
                    if tool_fn is None:
                        conv.add_tool_result(json.dumps({"error": f"未知工具: {tool_name}"}, ensure_ascii=False))
                    else:
                        try:
                            result = tool_fn(**clean_args) if clean_args else tool_fn()
                            result = _truncate_result(result) if isinstance(result, str) else str(result)
                        except Exception as exc:
                            logger.exception("Android tool %s failed", tool_name)
                            result = json.dumps({"error": str(exc)}, ensure_ascii=False)
                        conv.add_tool_result(result)
                else:
                    conv.add_assistant(text)
                    conv.trim(keep=24)
                    return text

            return "Android助手处理达到最大轮次限制,请简化问题后重试。"
        except Exception as exc:
            logger.exception("Android助手对话异常")
            raise RuntimeError("Android助手暂时不可用，请检查模型和检索服务。") from exc

    async def astream(self, user_message: str, thread_id: str = "default") -> AsyncIterator[dict[str, Any]]:
        if not self._llm:
            yield {"type": "error", "content": "Android助手未初始化，请检查LLM配置。"}
            return

        conv = self._get_conversation(thread_id)
        conv.add_user(user_message)

        try:
            for turn in range(MAX_REACT_TURNS):
                prompt = conv.build_prompt()
                response = await self._llm.ainvoke([HumanMessage(content=prompt)])
                text = str(response.content).strip() if response.content else ""

                if not text:
                    yield {"type": "error", "content": "Android助手未返回有效结果。"}
                    return

                tool_call = _parse_tool_call(text)
                if tool_call:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("arguments", {})
                    if not isinstance(tool_args, dict):
                        tool_args = {}
                    clean_args = {}
                    if isinstance(tool_args, dict):
                        for k, v in tool_args.items():
                            clean_args[str(k)] = v

                    yield {"type": "tool_start", "name": tool_name}
                    conv.add_assistant(text)
                    tool_fn = ANDROID_TOOLS.get(tool_name)
                    if tool_fn is None:
                        conv.add_tool_result(json.dumps({"error": f"未知工具: {tool_name}"}, ensure_ascii=False))
                        yield {"type": "tool_end", "name": tool_name, "preview": f"未知工具: {tool_name}"}
                    else:
                        try:
                            result = tool_fn(**clean_args) if clean_args else tool_fn()
                            result = _truncate_result(result) if isinstance(result, str) else str(result)
                        except Exception as exc:
                            result = json.dumps({"error": str(exc)}, ensure_ascii=False)
                        conv.add_tool_result(result)
                        preview = result[:200] + "…" if len(result) > 200 else result
                        yield {"type": "tool_end", "name": tool_name, "preview": preview}
                else:
                    conv.add_assistant(text)
                    conv.trim(keep=24)
                    for i in range(0, len(text), 20):
                        yield {"type": "token", "content": text[i:i + 20]}
                    return

            yield {"type": "error", "content": "Android助手处理达到最大轮次限制"}
        except Exception:
            logger.exception("Android助手流式对话异常")
            yield {"type": "error", "content": "Android助手暂时不可用，请稍后重试。"}

    def clear_memory(self, thread_id: str = "default") -> None:
        self._sessions.pop(thread_id, None)
        logger.info("Android助手对话记忆已清除: thread_id=%s", thread_id)

    @property
    def tools(self) -> list[str]:
        return ANDROID_TOOL_NAMES

    def probe(self) -> dict[str, Any]:
        if llm_settings.api_key in {"", "your_api_key_here"}:
            return {"ok": False, "model": llm_settings.model_name, "detail": "LLM API key is not configured"}
        if not self._llm:
            return {"ok": False, "detail": "LLM未初始化"}
        try:
            self._llm.invoke("ping")
            return {"ok": True, "model": llm_settings.model_name}
        except Exception as exc:
            return {"ok": False, "model": llm_settings.model_name, "detail": str(exc)}


AndroidAssisant = AndroidAssistant
android_assisant = AndroidAssistant
