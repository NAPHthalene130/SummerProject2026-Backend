import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.modules.agent.agent import Agent
from app.modules.agent.android_assistant import AndroidAssistant
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
    except RuntimeError as exc:
        logger.exception("Agent chat failed")
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Agent chat failed")
        raise HTTPException(status_code=500, detail=f"Agent处理异常: {exc}")


@agent_router.post(
    "/android/chat",
    response_model=ChatResponse,
    summary="Android交通助手对话",
)
async def android_chat(request: ChatRequest) -> ChatResponse:
    """Android端只读交通助手，支持按工单ID查询和交通法规RAG检索。"""
    thread_id = request.thread_id or str(uuid.uuid4())

    try:
        reply = await AndroidAssistant().chat(request.message, thread_id=thread_id)
        return ChatResponse(reply=reply, thread_id=thread_id)
    except RuntimeError as exc:
        logger.exception("Android assistant chat failed")
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Android assistant chat failed")
        raise HTTPException(status_code=500, detail=f"Android助手处理异常: {exc}")


@agent_router.post("/chat/stream", summary="Agent流式对话(SSE)")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """流式返回Agent回复,通过SSE推送工具执行进度与文本token。

    SSE事件(data行,JSON):
    - {"type":"tool_start","name":"..."} 工具开始执行
    - {"type":"tool_end","name":"...","preview":"..."} 工具执行完毕(含结果摘要)
    - {"type":"token","content":"..."} 模型文本片段
    - {"type":"error","content":"..."} 错误
    - {"type":"done","thread_id":"...","reply":"..."} 结束(含完整回复与会话ID)
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    agent = Agent()

    async def event_generator():
        full_reply: list[str] = []
        try:
            async for evt in agent.astream(request.message, thread_id=thread_id):
                if evt.get("type") == "token":
                    full_reply.append(evt.get("content", ""))
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.exception("Agent stream failed")
            err = {"type": "error", "content": f"Agent处理异常: {exc}"}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
        finally:
            done = {
                "type": "done",
                "thread_id": thread_id,
                "reply": "".join(full_reply),
            }
            yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@agent_router.get("/health", summary="Agent健康检查")
async def health() -> dict:
    """检查Agent是否可用,包含对模型服务的轻量探活。"""
    try:
        agent = Agent()
        probe = agent.probe()
        return {
            "status": "healthy" if probe.get("ok") else "degraded",
            "model": probe.get("model"),
            "tools": agent.tools,
            "detail": probe.get("detail"),
        }
    except Exception as exc:
        return {"status": "unhealthy", "error": str(exc)}


@agent_router.delete("/session/{thread_id}", summary="清除指定会话记忆")
async def clear_session(thread_id: str) -> dict:
    """清除指定thread_id的对话上下文,开启全新会话。"""
    try:
        Agent().clear_memory(thread_id)
        return {"status": "ok", "thread_id": thread_id}
    except Exception as exc:
        logger.exception("clear session failed")
        raise HTTPException(status_code=500, detail=f"清除会话失败: {exc}")


@agent_router.delete("/android/session/{thread_id}", summary="清除Android助手会话记忆")
async def clear_android_session(thread_id: str) -> dict:
    try:
        AndroidAssistant().clear_memory(thread_id)
        return {"status": "ok", "thread_id": thread_id}
    except Exception as exc:
        logger.exception("clear Android assistant session failed")
        raise HTTPException(status_code=500, detail=f"清除Android助手会话失败: {exc}")


TOOL_DESCRIPTIONS: dict[str, str] = {
    "query_work_orders": "查询工单列表,支持按用户、状态、等级筛选",
    "get_work_order_detail": "获取指定工单的完整详情",
    "search_work_orders": "根据工单编号检索工单及派发信息",
    "query_staff": "查询所有可派发人员信息及空闲状态",
    "query_work_order_stats": "查询工单统计数据,按状态和等级汇总",
    "suggest_handling": "分析工单并基于当前状态生成分阶段处置建议",
    "dispatch_work_order": "将单个工单派发给指定人员",
    "batch_dispatch_unassigned": "批量派发:自动将所有未派发工单按类别匹配最优人员并派发(不指定编号范围)",
    "dispatch_work_order_range": "区间派发:派发指定编号区间(如300~330)内的所有未派发工单",
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
