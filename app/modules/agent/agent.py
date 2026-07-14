import asyncio
import json
import logging
import re
from typing import Any, AsyncIterator, Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from app.config import llm_settings
from app.modules.agent.base import BaseAgent
from app.modules.agent.tool import (
    answer_general_question,
    batch_dispatch_unassigned,
    batch_ignore_work_orders,
    dispatch_work_order,
    dispatch_work_order_range,
    dry_run_batch_dispatch,
    get_work_order_detail,
    query_staff,
    query_work_order_stats,
    query_work_orders,
    search_work_orders,
    suggest_handling,
)

logger = logging.getLogger(__name__)

MAX_REACT_TURNS = 10

TOOLS = [
    query_work_orders,
    get_work_order_detail,
    search_work_orders,
    query_staff,
    query_work_order_stats,
    suggest_handling,
    dispatch_work_order,
    dry_run_batch_dispatch,
    batch_dispatch_unassigned,
    dispatch_work_order_range,
    batch_ignore_work_orders,
    answer_general_question,
]

TOOL_NAMES = [t.__name__ for t in TOOLS]
TOOL_BY_NAME = {t.__name__: t for t in TOOLS}

TOOL_DESCRIPTIONS = {
    "query_work_orders": "查询工单列表,参数: user_id(可选,int), status(可选,str), level(可选,str), limit(可选,int), offset(可选,int)",
    "get_work_order_detail": "获取工单详情,参数: work_order_id(str)",
    "search_work_orders": "搜索工单,参数: keyword(str)",
    "query_staff": "查询可派发人员列表,无必需参数",
    "query_work_order_stats": "查询工单统计概况,无必需参数",
    "suggest_handling": "分析工单并生成处置建议,参数: work_order_id(str)",
    "dispatch_work_order": "派发单个工单,参数: work_order_id(str), user_id(int)",
    "dry_run_batch_dispatch": "批量分配预演(不修改数据库),参数: start_id(可选,int), end_id(可选,int), stage(可选,str), event_level(可选,str), required_category(可选,str)",
    "batch_dispatch_unassigned": "批量派发所有未派发工单,无必需参数",
    "dispatch_work_order_range": "区间派发工单,参数: start_id(int), end_id(int)",
    "batch_ignore_work_orders": "批量忽略无法处理的工单,参数: start_id(可选,int), end_id(可选,int), stage(可选,str), event_level(可选,str), required_category(可选,str), reason(可选,str)",
    "answer_general_question": "检索交通法规回答通用问题,参数: question(str)",
}

SYSTEM_PROMPT = """你是智能交通指挥Agent。当用户发出指令时,你必须**优先执行操作**,直接调用工具,不要先输出分析报告。

## 可用工具
{tool_list}

## 工具调用格式
当你需要调用工具时,严格按以下JSON格式输出(不要加任何其他文字):

<tool_call>
{{"name": "工具名", "arguments": {{"参数名": "参数值"}}}}
</tool_call>

工具执行后你会收到结果,然后可以继续调用其他工具或给出最终回答。

## 执行规则 (最高优先级)
1. 用户说"派发"(单条,如"派发工单335") → 调用 dispatch_work_order。若用户未指定人员,必须先调用 query_staff 获取同类别(work_order_count最小)的人员ID,再调用 dispatch_work_order,严禁编造 user_id;若用户指定了人员名,也先 query_staff 取其 user_id
2. 用户说"全部派发"/"批量派发"/"批量处理"/"全部尚未派发"/"批量分配"/"全部分配"/"自动指派"(未指定编号范围) → 调用 batch_dispatch_unassigned 一次完成,不要逐个派发
3. 用户说"查询/查看/有哪些" → 调用对应的查询Tool,用1-3句话回复关键信息
4. 用户说"怎么处理/怎么办" → 调用 suggest_handling,用3-5条操作步骤回复
5. 用户说"统计/概况" → 调用 query_work_order_stats,用3-4行回复
6. 用户说"法规/规定/标准" → 调用 answer_general_question;该工具仅返回法规检索依据,需由你据此组织最终中文回答
7. 批量区间(如"把300~330的工单派发/分配/指派"/"300到330"/"300-330全部派发") → 只需提取起止两个整数,调用 dispatch_work_order_range(300, 330) 一次完成;严禁自行展开成多次 dispatch_work_order,严禁只派发其中一单
8. 用户说"无法处理"/"没有可用人员"/"不再处理"/"批量忽略"/"全部忽略" → 调用 batch_ignore_work_orders。若用户给出编号范围,传 start_id/end_id;否则默认只忽略未派发(unassigned)工单。必须保留或概括用户给出的忽略原因

## 预演规则
用户说"预演"/"模拟"/"看看会怎么分配"/"不要改数据库"/"先检查批量分配"时,调用 dry_run_batch_dispatch;若用户给出编号范围或过滤条件,传 start_id/end_id、event_level、required_category,严禁调用会写库的批量派发工具。

## 歧义兜底
若用户指令无法匹配任何工具,用一句话说明你能做什么(查询工单/查询人员/派发/批量派发/区间派发/批量忽略/统计/处置建议/法规问答),不要编造结果或数据。

## 输出格式
- 操作类: "已派发工单xxx至[人员名],状态:待处理。"
- 查询类: "工单xxx,类型[xxx],等级[xxx],状态[xxx],负责人[xxx]。"
- 建议类: "该工单为[等级]的[类型],当前[状态]。建议:1.xxx 2.xxx 3.xxx"
- 统计类: "当前共N单,未解决M单(高优K单),人员平均负载X单。"
- 批量/区间/忽略类: 严格遵循工具返回的 presentation_hint,用1行汇总;不要逐条罗列明细,不要输出表格
- 严禁输出Markdown表格;严禁出现"结论""依据""未核验项"等结构化小标题;严禁输出内部JSON键名

## 字段翻译(内部JSON→用户输出)
work_order_stage→工单状态:unassigned→未派发 pending→待处理 processing→处理中 completed→已完成 ignored→已忽略
event_level→等级:low→一般 medium→中等 high→紧急
required_category→所需类别:traffic_police→交警 road_maintenance→道路养护 municipal_facilities→市政设施
work_order_count→负载, camera_name→点位, assignee→负责人, distance_km→距离
禁止输出英文JSON键名、数据库表名、内部表名;user_id 等纯数字ID仅供内部传参,不向用户展示

## 派发规则
- 用户说"派发"即视为授权,直接调用 dispatch_work_order
- 类别不匹配时,先尝试派发,工具会返回错误;将错误原样告知用户
- 批量派发时调用一次 batch_dispatch_unassigned"""


def _build_tool_list_text() -> str:
    lines: list[str] = []
    for name, desc in TOOL_DESCRIPTIONS.items():
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def _parse_tool_call(text: str) -> Optional[dict[str, Any]]:
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        logger.warning("Failed to parse tool call JSON: %s", m.group(1)[:200])
        return None


async def _execute_tool(name: str, arguments: dict[str, Any]) -> str:
    tool = TOOL_BY_NAME.get(name)
    if tool is None:
        return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
    try:
        result = await asyncio.to_thread(tool, **arguments)
    except Exception as exc:
        logger.exception("Tool %s execution failed", name)
        return json.dumps({"error": str(exc)}, ensure_ascii=False)

    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, default=str)


def _truncate_tool_result(result: str, max_chars: int = 3000) -> str:
    if len(result) <= max_chars:
        return result
    return result[:max_chars] + "\n…[结果过长已截断]"


class ReActConversation:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_result(self, content: str) -> None:
        self.messages.append({"role": "tool", "content": content})

    def build_prompt(self) -> str:
        tool_list = _build_tool_list_text()
        header = SYSTEM_PROMPT.format(tool_list=tool_list) + "\n\n## 对话历史\n"
        parts: list[str] = [header]
        for msg in self.messages[-20:]:
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

    def trim(self, keep: int = 20) -> None:
        if len(self.messages) > keep:
            self.messages = self.messages[-keep:]


class Agent(BaseAgent):
    """智能交通指挥Agent,基于ReAct模式,不依赖原生function calling。"""

    _instance: Optional["Agent"] = None

    def __new__(cls) -> "Agent":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        self._llm: Optional[ChatOpenAI] = None
        self._sessions: dict[str, ReActConversation] = {}

        self._init_llm()
        logger.info("SmartDispatchAgent(ReAct) 初始化完成")

    def _init_llm(self) -> None:
        self._llm = ChatOpenAI(
            base_url=llm_settings.url,
            api_key=llm_settings.api_key,  # type: ignore[arg-type]
            model=llm_settings.model_name,
            temperature=0.1,
            max_tokens=2048,
            timeout=60,
            max_retries=2,
        )

    def _get_conversation(self, thread_id: str) -> ReActConversation:
        if thread_id not in self._sessions:
            self._sessions[thread_id] = ReActConversation()
        return self._sessions[thread_id]

    async def chat(self, user_message: str, thread_id: str = "default") -> str:
        if not self._llm:
            return "Agent 未初始化,请检查 LLM 配置。"

        conv = self._get_conversation(thread_id)
        conv.add_user(user_message)

        try:
            for turn in range(MAX_REACT_TURNS):
                prompt = conv.build_prompt()
                response = await self._llm.ainvoke([HumanMessage(content=prompt)])
                text = str(response.content).strip() if response.content else ""

                if not text:
                    return "Agent 未返回有效结果。"

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

                    logger.info("ReAct turn %d: calling %s with %s", turn + 1, tool_name, str(clean_args)[:200])
                    conv.add_assistant(text)
                    result = await _execute_tool(tool_name, clean_args)
                    result = _truncate_tool_result(result)
                    conv.add_tool_result(result)
                else:
                    conv.add_assistant(text)
                    conv.trim(keep=30)
                    return text

            return "Agent 处理达到最大轮次限制,请简化问题后重试。"
        except Exception as exc:
            logger.exception("Agent对话异常")
            raise RuntimeError("Agent 对话执行失败,请检查模型服务与工具连接。") from exc

    def chat_sync(self, user_message: str, thread_id: str = "default") -> str:
        if not self._llm:
            return "Agent 未初始化,请检查 LLM 配置。"

        conv = self._get_conversation(thread_id)
        conv.add_user(user_message)

        try:
            for turn in range(MAX_REACT_TURNS):
                prompt = conv.build_prompt()
                response = self._llm.invoke([HumanMessage(content=prompt)])
                text = str(response.content).strip() if response.content else ""

                if not text:
                    return "Agent 未返回有效结果。"

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

                    logger.info("ReAct sync turn %d: calling %s", turn + 1, tool_name)
                    conv.add_assistant(text)
                    result = _execute_tool_sync(tool_name, clean_args)
                    result = _truncate_tool_result(result)
                    conv.add_tool_result(result)
                else:
                    conv.add_assistant(text)
                    conv.trim(keep=30)
                    return text

            return "Agent 处理达到最大轮次限制,请简化问题后重试。"
        except Exception as exc:
            logger.exception("Agent同步对话异常")
            raise RuntimeError("Agent 对话执行失败,请检查模型服务与工具连接。") from exc

    async def astream(self, user_message: str, thread_id: str = "default") -> AsyncIterator[dict[str, Any]]:
        if not self._llm:
            yield {"type": "error", "content": "Agent 未初始化,请检查 LLM 配置。"}
            return

        conv = self._get_conversation(thread_id)
        conv.add_user(user_message)

        try:
            for turn in range(MAX_REACT_TURNS):
                prompt = conv.build_prompt()
                response = await self._llm.ainvoke([HumanMessage(content=prompt)])
                text = str(response.content).strip() if response.content else ""

                if not text:
                    yield {"type": "error", "content": "Agent 未返回有效结果。"}
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
                    result = await _execute_tool(tool_name, clean_args)
                    preview = result[:200] + "…" if len(result) > 200 else result
                    yield {"type": "tool_end", "name": tool_name, "preview": preview}
                    result = _truncate_tool_result(result)
                    conv.add_tool_result(result)
                else:
                    conv.add_assistant(text)
                    conv.trim(keep=30)
                    for i in range(0, len(text), 20):
                        yield {"type": "token", "content": text[i:i + 20]}
                    return

            yield {"type": "error", "content": "Agent 处理达到最大轮次限制"}
        except Exception:
            logger.exception("Agent流式对话异常")
            yield {"type": "error", "content": "Agent 对话执行失败,请检查模型服务与工具连接。"}

    def clear_memory(self, thread_id: str = "default") -> None:
        self._sessions.pop(thread_id, None)
        logger.info("Agent 对话记忆已清除: thread_id=%s", thread_id)

    @property
    def tools(self) -> list[str]:
        return TOOL_NAMES

    def probe(self) -> dict[str, Any]:
        if llm_settings.api_key in {"", "your_api_key_here"}:
            return {
                "ok": False,
                "model": llm_settings.model_name,
                "detail": "LLM API key is not configured",
            }
        if not self._llm:
            return {"ok": False, "detail": "LLM 未初始化"}
        try:
            self._llm.invoke("ping")
            return {"ok": True, "model": llm_settings.model_name}
        except Exception as exc:
            return {"ok": False, "model": llm_settings.model_name, "detail": str(exc)}


def _execute_tool_sync(name: str, arguments: dict[str, Any]) -> str:
    tool = TOOL_BY_NAME.get(name)
    if tool is None:
        return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
    try:
        result = tool(**arguments)
    except Exception as exc:
        logger.exception("Tool %s sync execution failed", name)
        return json.dumps({"error": str(exc)}, ensure_ascii=False)

    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, default=str)
