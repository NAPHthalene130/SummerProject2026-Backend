import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.modules.agent.agent import Agent
from app.modules.agent.workflow import AgentWorkflow

logger = logging.getLogger(__name__)

agent_router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096, description="用户输入的问题")
    thread_id: Optional[str] = Field(default=None, description="会话线程ID,不传则自动创建新会话")


class ChatResponse(BaseModel):
    reply: str
    thread_id: str


class ToolInfo(BaseModel):
    name: str
    description: str


@agent_router.post("/chat", response_model=ChatResponse, summary="Agent对话")
async def chat(request: ChatRequest) -> ChatResponse:
    """向Agent发送消息并获取回复。

    通过thread_id维护多轮对话上下文:
    - 首次对话不传thread_id,系统自动创建并返回
    - 后续对话传入之前返回的thread_id,保持上下文连续
    - 切换thread_id开启新会话
    """
    thread_id = request.thread_id or str(uuid.uuid4())

    try:
        agent = Agent()
        reply = await agent.chat(request.message, thread_id=thread_id)
        return ChatResponse(reply=reply, thread_id=thread_id)
    except Exception as exc:
        logger.exception("Agent chat failed")
        raise HTTPException(status_code=500, detail=f"Agent处理异常: {exc}")


@agent_router.get("/health", summary="Agent健康检查")
async def health() -> dict:
    """检查Agent是否可用。"""
    try:
        agent = Agent()
        return {
            "status": "healthy",
            "tools": agent.tools,
        }
    except Exception as exc:
        return {
            "status": "unhealthy",
            "error": str(exc),
        }


TOOL_DESCRIPTIONS: dict[str, str] = {
    "query_work_orders": "查询工单列表,支持按用户、状态、等级筛选",
    "get_work_order_detail": "获取指定工单的完整详情",
    "search_work_orders": "根据工单编号检索工单及派发信息",
    "query_staff": "查询所有可派发人员信息及空闲状态",
    "query_work_order_stats": "查询工单统计数据,按状态和等级汇总",
    "suggest_handling": "分析工单并基于当前状态生成分阶段处置建议",
    "dispatch_work_order": "将工单派发给指定人员",
    "answer_general_question": "回答交通管理法规和处置流程等通用问题",
}


@agent_router.get("/tools", response_model=list[ToolInfo], summary="可用Tool列表")
async def list_tools() -> list[ToolInfo]:
    """返回Agent当前注册的所有Tool及功能说明。"""
    agent = Agent()
    return [
        ToolInfo(name=name, description=TOOL_DESCRIPTIONS.get(name, ""))
        for name in agent.tools
    ]


@agent_router.get("/workflow/diagnose/{work_order_id}", summary="工单诊断工作流")
async def workflow_diagnose(work_order_id: str) -> dict:
    """工单全链路诊断: 详情→建议→人员推荐→行动计划。"""
    result = AgentWorkflow.diagnose(work_order_id)
    if not result.success:
        raise HTTPException(status_code=404, detail=result.error)
    return result.data


@agent_router.get("/workflow/brief", summary="态势简报工作流")
async def workflow_brief() -> dict:
    """态势简报: 统计概览→高优工单→人员状态。"""
    result = AgentWorkflow.brief()
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error)
    return result.data


@agent_router.get("/workflow/dispatch-recommend/{work_order_id}", summary="智能派发推荐工作流")
async def workflow_dispatch_recommend(work_order_id: str, top_k: int = 3) -> dict:
    """智能派发推荐: 工单匹配→人员筛选→Top-K推荐。"""
    result = AgentWorkflow.dispatch_recommend(work_order_id, top_k=top_k)
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    return result.data
