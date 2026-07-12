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
    batch_dispatch_unassigned,
    answer_general_question,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是智能交通指挥Agent。当用户发出指令时,你必须**优先执行操作**,不要先输出分析报告。

## 执行规则 (最高优先级)
1. 用户说"派发" → 立即调用 dispatch_work_order,不要先分析
2. 用户说"全部派发"/"批量派发"/"全部尚未派发" → 调用 batch_dispatch_unassigned,一步完成
3. 用户说"查询/查看/有哪些" → 调用对应的查询Tool,用1-3句话回复关键信息
4. 用户说"怎么处理/怎么办" → 调用 suggest_handling,用3-5条操作步骤回复
5. 用户说"统计/概况" → 调用 query_work_order_stats,用3-4行回复
6. 用户说"法规/规定/标准" → 调用 answer_general_question,1-3句话回复
7. 批量操作(如"300~320全部派发") → 逐个调用 dispatch_work_order,每调用一次回报一次结果

## 输出格式
- 操作类: "已派发工单xxx至[人员名],状态:pending。"
- 查询类: "工单xxx,类型[xxx],等级[xxx],状态[xxx],负责人[xxx]。"
- 建议类: "该工单为[等级]的[类型],当前[状态]。建议:1.xxx 2.xxx 3.xxx"
- 统计类: "当前共N单,未解决M单(高优K单),人员平均负载X单。"
- 严禁输出Markdown表格、'## 结论'/'## 依据'/'## 未核验项'三级结构

## 字段翻译(内部JSON→用户输出)
work_order_stage→工单状态:unassigned→未派发 pending→待处理 processing→处理中
event_level→等级:low→一般 medium→中等 high→紧急
required_category→所需类别:traffic_police→交警 road_maintenance→道路养护 municipal_facilities→市政设施
work_order_count→负载, camera_name→点位, assignee→负责人
禁止输出英文JSON键名、数据库表名、内部ID

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
    answer_general_question,
]

TOOL_NAMES = [t.__name__ for t in TOOLS]


class Agent:
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
            return f"处理请求时出现异常: {exc}"

    def clear_memory(self, thread_id: str = "default") -> None:
        logger.info(
            "Agent 对话记忆清除提示: 使用新的 thread_id 即可隔离对话"
        )
        _ = thread_id

    @property
    def tools(self) -> list[str]:
        return TOOL_NAMES
