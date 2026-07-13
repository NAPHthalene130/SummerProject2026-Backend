import logging
from typing import Any, AsyncIterator, Optional

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from app.config import llm_settings
from app.modules.agent.base import BaseAgent
from app.modules.agent.tool import (
    query_work_orders,
    get_work_order_detail,
    search_work_orders,
    query_staff,
    query_work_order_stats,
    suggest_handling,
    dispatch_work_order,
    batch_dispatch_unassigned,
    dispatch_work_order_range,
    answer_general_question,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是智能交通指挥Agent。当用户发出指令时,你必须**优先执行操作**,不要先输出分析报告。

## 执行规则 (最高优先级)
1. 用户说"派发"(单条,如"派发工单335") → 调用 dispatch_work_order。若用户未指定人员,必须先调用 query_staff 获取同类别(work_order_count最小)的人员ID,再调用 dispatch_work_order,严禁编造 user_id;若用户指定了人员名,也先 query_staff 取其 user_id
2. 用户说"全部派发"/"批量派发"/"批量处理"/"全部尚未派发"(未指定编号范围) → 调用 batch_dispatch_unassigned 一次完成,不要逐个派发
3. 用户说"查询/查看/有哪些" → 调用对应的查询Tool,用1-3句话回复关键信息
4. 用户说"怎么处理/怎么办" → 调用 suggest_handling,用3-5条操作步骤回复
5. 用户说"统计/概况" → 调用 query_work_order_stats,用3-4行回复
6. 用户说"法规/规定/标准" → 调用 answer_general_question;该工具仅返回法规检索依据,需由你据此组织最终中文回答
7. 批量区间(如"把300~330的工单派发"/"300到330"/"300-330全部派发") → 只需提取起止两个整数,调用 dispatch_work_order_range(300, 330) 一次完成;严禁自行展开成多次 dispatch_work_order,严禁只派发其中一单

## 歧义兜底
若用户指令无法匹配任何工具,用一句话说明你能做什么(查询工单/查询人员/派发/批量派发/区间派发/统计/处置建议/法规问答),不要编造结果或数据。

## 输出格式
- 操作类: "已派发工单xxx至[人员名],状态:待处理。"
- 查询类: "工单xxx,类型[xxx],等级[xxx],状态[xxx],负责人[xxx]。"
- 建议类: "该工单为[等级]的[类型],当前[状态]。建议:1.xxx 2.xxx 3.xxx"
- 统计类: "当前共N单,未解决M单(高优K单),人员平均负载X单。"
- 批量/区间类: 严格遵循工具返回的 presentation_hint,用1行汇总,例如"区间300~330内共派发28单,跳过0单,失败0单。";不要逐条罗列明细,不要输出表格
- 严禁输出Markdown表格;严禁出现"结论""依据""未核验项"等结构化小标题(无论是否带##);严禁输出内部JSON键名

## 字段翻译(内部JSON→用户输出)
work_order_stage→工单状态:unassigned→未派发 pending→待处理 processing→处理中 completed→已完成 ignored→已忽略
event_level→等级:low→一般 medium→中等 high→紧急
required_category→所需类别:traffic_police→交警 road_maintenance→道路养护 municipal_facilities→市政设施
work_order_count→负载, camera_name→点位, assignee→负责人, distance_km→距离
禁止输出英文JSON键名、数据库表名、内部表名;user_id 等纯数字ID仅供内部传参,不向用户展示

## 派发规则
- 用户说"派发"即视为授权,直接调用 dispatch_work_order
- 类别不匹配时,先尝试派发,工具会返回错误;将错误原样告知用户
- 批量派发时逐条执行,每条汇报结果"""

TOOLS = [
    query_work_orders,
    get_work_order_detail,
    search_work_orders,
    query_staff,
    query_work_order_stats,
    suggest_handling,
    dispatch_work_order,
    batch_dispatch_unassigned,
    dispatch_work_order_range,
    answer_general_question,
]

TOOL_NAMES = [t.__name__ for t in TOOLS]


class Agent(BaseAgent):
    """智能交通指挥Agent,基于LangChain 1.x构建。"""

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
        self._graph: Optional[CompiledStateGraph] = None
        self._memory: Optional[MemorySaver] = None

        self._init_llm()
        self._init_agent()
        logger.info("SmartDispatchAgent 初始化完成(单例)")

    def _init_llm(self) -> None:
        self._llm = ChatOpenAI(
            base_url=llm_settings.url,
            api_key=llm_settings.api_key,  # type: ignore[arg-type]
            model=llm_settings.model_name,
            temperature=0.1,
            max_tokens=2048,
            timeout=30,
            max_retries=2,
        )

    def _init_agent(self) -> None:
        self._memory = MemorySaver()
        self._graph = create_agent(
            model=self._llm,
            tools=TOOLS,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=self._memory,
        )

    def _build_config(self, thread_id: str = "default") -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    async def chat(self, user_message: str, thread_id: str = "default") -> str:
        if not self._graph:
            return "Agent 未初始化,请检查 LLM 配置。"

        try:
            result = await self._graph.ainvoke(
                {"messages": [HumanMessage(content=user_message)]},
                config=self._build_config(thread_id),
            )
            messages: list[Any] = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                return str(last_msg.content)
            return "Agent 未返回有效结果。"
        except Exception as exc:
            logger.exception("Agent对话异常")
            raise RuntimeError("Agent 对话执行失败,请检查模型服务与工具连接。") from exc

    def chat_sync(self, user_message: str, thread_id: str = "default") -> str:
        if not self._graph:
            return "Agent 未初始化,请检查 LLM 配置。"

        try:
            result = self._graph.invoke(
                {"messages": [HumanMessage(content=user_message)]},
                config=self._build_config(thread_id),
            )
            messages: list[Any] = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                return str(last_msg.content)
            return "Agent 未返回有效结果。"
        except Exception as exc:
            logger.exception("Agent同步对话异常")
            raise RuntimeError("Agent 对话执行失败,请检查模型服务与工具连接。") from exc

    async def astream(self, user_message: str, thread_id: str = "default") -> AsyncIterator[dict[str, Any]]:
        """异步流式输出,逐个产出事件字典。

        事件类型:
        - {"type": "tool_start", "name": 工具名}
        - {"type": "tool_end", "name": 工具名, "preview": 结果摘要}
        - {"type": "token", "content": 文本片段}
        - {"type": "error", "content": 错误信息}
        """
        if not self._graph:
            yield {"type": "error", "content": "Agent 未初始化,请检查 LLM 配置。"}
            return

        config = self._build_config(thread_id)
        try:
            async for event in self._graph.astream_events(
                {"messages": [HumanMessage(content=user_message)]},
                config=config,
                version="v2",
            ):
                kind = event.get("event", "")
                name = event.get("name", "")
                data = event.get("data") or {}

                if kind == "on_tool_start":
                    yield {"type": "tool_start", "name": name}
                elif kind == "on_tool_end":
                    yield {
                        "type": "tool_end",
                        "name": name,
                        "preview": self._extract_tool_preview(data.get("output")),
                    }
                elif kind == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    content = getattr(chunk, "content", "")
                    if isinstance(content, str) and content:
                        yield {"type": "token", "content": content}
        except Exception:
            logger.exception("Agent流式对话异常")
            yield {
                "type": "error",
                "content": "Agent 对话执行失败,请检查模型服务与工具连接。",
            }

    @staticmethod
    def _extract_tool_preview(output: Any) -> str:
        if output is None:
            return ""
        text = getattr(output, "content", output)
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if len(text) > 200:
            return text[:200] + "…"
        return text

    def clear_memory(self, thread_id: str = "default") -> None:
        if not self._memory or not thread_id:
            return
        try:
            self._memory.delete_thread(thread_id)
            logger.info("Agent 对话记忆已清除: thread_id=%s", thread_id)
        except Exception:
            logger.exception("清除对话记忆失败: thread_id=%s", thread_id)

    @property
    def tools(self) -> list[str]:
        return TOOL_NAMES

    def probe(self) -> dict[str, Any]:
        """轻量探活:对模型服务发起一次极小请求,验证连通性。"""
        if llm_settings.api_key in {"", "your_api_key_here"}:
            return {
                "ok": False,
                "model": llm_settings.model_name,
                "detail": "LLM API key is not configured",
            }
        if not self._llm:
            return {"ok": False, "detail": "LLM 未初始化"}
        try:
            self._llm.invoke("ping", config={"tags": ["health-probe"]})
            return {"ok": True, "model": llm_settings.model_name}
        except Exception as exc:
            return {"ok": False, "model": llm_settings.model_name, "detail": str(exc)}
