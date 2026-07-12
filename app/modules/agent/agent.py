import logging
from typing import Any, Optional

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from app.config import llm_settings
from app.modules.agent.tool import (
    query_work_orders,
    get_work_order_detail,
    search_work_orders,
    query_staff,
    suggest_handling,
    dispatch_work_order,
    answer_general_question,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是智能交通问答助手(SmartQAAgent),负责回答关于交通管理、道路运营、工单处置等方面的问题。

你的核心职责:
1. 问答: 回答关于工单状态、道路状况、摄像头信息、人员配置等运营问题
2. 查询: 根据工单编号检索工单详情和派发记录
3. 知识: 回答交通管理规范、处置流程、事故分类等知识性问题

回答原则:
- 回答简洁专业,避免冗长
- 涉及工单ID时使用标准格式 WO-YYYYMMDD-NNN
- 不确定的信息请明确说明
- 需要查询数据时主动调用相关工具"""

TOOLS = [
    query_work_orders,
    get_work_order_detail,
    search_work_orders,
    query_staff,
    suggest_handling,
    dispatch_work_order,
    answer_general_question,
]

TOOL_NAMES = [t.__name__ for t in TOOLS]


class Agent:
    """智能交通问答Agent,基于LangChain 1.x构建,负责问答交互。

    注册Tool:
    - 问答: query_work_orders / get_work_order_detail / search_work_orders / query_staff / answer_general_question
    - 建议: suggest_handling
    - 派发: dispatch_work_order

    TODO:
    - 接入RAG知识库(运维手册/SOP)增强问答能力
    - 实现对话历史持久化到数据库(当前使用内存MemorySaver)
    - 支持多租户/多会话隔离(thread_id)
    - 新增评估反馈机制(thumb-up/thumb-down)
    - 支持流式(streaming)响应,实时返回生成内容
    """

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
        logger.info("Agent 初始化完成(单例)")

    def _init_llm(self) -> None:
        self._llm = ChatOpenAI(
            base_url=llm_settings.url,
            api_key=llm_settings.api_key,  # type: ignore[arg-type]
            model=llm_settings.model_name,
            temperature=0.3,
            max_tokens=1024,
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
        """与Agent对话的异步入口。

        thread_id: 会话线程ID,用于隔离不同用户的对话历史。
        """
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
            return f"处理请求时出现异常: {exc}"

    def chat_sync(self, user_message: str, thread_id: str = "default") -> str:
        """同步对话入口(用于测试和非异步环境)。"""
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
            return f"处理请求时出现异常: {exc}"

    def clear_memory(self, thread_id: str = "default") -> None:
        """清除指定线程的对话历史。

        TODO: 支持选择性清除(仅清除某条对话记录)
        TODO: 持久化清除操作到数据库
        """
        logger.info(
            "Agent 对话记忆清除提示: 使用新的 thread_id 即可隔离对话"
        )
        _ = thread_id

    @property
    def tools(self) -> list[str]:
        """返回已注册的Tool名称列表。"""
        return TOOL_NAMES
