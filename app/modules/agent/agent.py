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
    query_work_order_stats,
    suggest_handling,
    dispatch_work_order,
    answer_general_question,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是智能交通管理指挥Agent(SmartDispatchAgent),负责工单问答、状态分析、处置建议和派发协调。

## 核心职责
1. **工单问答**: 回答工单状态、详情、统计等查询
2. **处置建议**: 依据工单当前状态和等级,提供分阶段处置建议
3. **派发协调**: 推荐合适人员并执行派发
4. **法规知识**: 回答交通管理法规、处置流程等知识性问题

## 工单状态说明
工单有五阶段生命周期,不同阶段对应不同的决策重点:
- **unassigned(未派发)**: 工单刚创建,需评估优先级并推荐派发人员
- **pending(待处理)**: 已派发待响应,关注处置时效和步骤指导
- **processing(处理中)**: 正在处置,关注进展监控和升级条件判断
- **completed(已完成)**: 处置完毕,关注复盘总结
- **ignored(已忽略)**: 被忽略的工单,关注重新激活条件

## 事件等级
- **low**: 一般事件,常规处置即可
- **medium**: 中等事件,需关注时效
- **high**: 紧急事件,需优先响应和加急处置

## 工作流程指引
当用户询问工单相关问题时,遵循以下流程:
1. 用户询问某个工单 → 先调用 search_work_orders 获取工单详情,再根据状态给出针对性回答
2. 用户问"有哪些工单"/"当前待处理" → 调用 query_work_orders,可指定 stage 和 level 筛选
3. 用户问"这个工单怎么处理" → 调用 suggest_handling 获取建议上下文,结合工单状态生成建议
4. 用户问"派发给谁/怎么派发" → 先 query_staff 查可用人员(关注work_order_count,越小越闲),再 suggest_handling 获取建议,最后 dispatch_work_order 执行
5. 用户问"整体情况/统计" → 调用 query_work_order_stats 获取全局数据
6. 用户问法规/流程/标准 → 调用 answer_general_question 检索知识库

## 回答原则
- 回答简洁专业,结构化呈现(分点、标注优先级)
- 工单ID使用标准格式 WO-YYYYMMDD-NNN
- 根据事件等级调整语气: high等级加急提醒,medium等级关注时效
- 不确定的信息明确说明,建议进一步确认
- 派发前必须确认:工单状态为unassigned,人员类别匹配且work_order_count最低优先
- 给出建议时要说明依据(工单等级、事故类型、法规条款等)"""

TOOLS = [
    query_work_orders,
    get_work_order_detail,
    search_work_orders,
    query_staff,
    query_work_order_stats,
    suggest_handling,
    dispatch_work_order,
    answer_general_question,
]

TOOL_NAMES = [t.__name__ for t in TOOLS]


class Agent:
    """智能交通指挥Agent,基于LangChain 1.x构建。

    注册Tool(8个):
    - 查询: query_work_orders / get_work_order_detail / search_work_orders
    - 人员: query_staff
    - 统计: query_work_order_stats
    - 建议: suggest_handling
    - 派发: dispatch_work_order
    - 知识: answer_general_question (含RAG)

    TODO:
    - 实现对话历史持久化到数据库(当前使用内存MemorySaver)
    - 支持多租户/多会话隔离(thread_id)
    - 新增评估反馈机制(thumb-up/thumb-down)
    - 支持流式(streaming)响应
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
        logger.info("SmartDispatchAgent 初始化完成(单例)")

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

        使用新的 thread_id 即可隔离对话。
        """
        logger.info(
            "Agent 对话记忆清除提示: 使用新的 thread_id 即可隔离对话"
        )
        _ = thread_id

    @property
    def tools(self) -> list[str]:
        return TOOL_NAMES
