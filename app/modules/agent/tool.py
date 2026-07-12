import json
import logging
from typing import Any, Optional

from app.database import mysql_connection
from app.repository.work_order_repository import (
    WorkOrderRepository,
    parse_work_order_id,
    rank_to_level,
)

logger = logging.getLogger(__name__)


def _parse_work_order_id(work_order_id: str) -> int:
    text = work_order_id.strip()
    if text.upper().startswith("WO-"):
        text = text.split("-")[-1]
    return int(text.lstrip("0") or "0")


def _serialize_datetime(value: Any) -> Optional[str]:
    if value is None:
        return None
    return value.strftime("%Y-%m-%d %H:%M:%S") if hasattr(value, "strftime") else str(value)


def _format_work_order_for_agent(wo: Any) -> dict[str, Any]:
    return {
        "work_order_id": wo.work_order_id,
        "event_id": wo.event_id,
        "camera_id": wo.camera_id,
        "camera_name": wo.camera_name,
        "monitor_address": wo.monitor_address,
        "accident_info": wo.accident_info,
        "event_time": wo.event_time,
        "event_level": wo.event_level,
        "status": wo.status,
        "assignee": wo.assignee,
        "description": wo.description,
        "ai_suggestion": wo.ai_suggestion,
        "scene_info": wo.scene_info,
        "process_message": wo.process_message,
        "completed_at": wo.completed_at,
        "required_category": wo.required_category,
    }


def _format_staff_for_agent(staff: Any) -> dict[str, Any]:
    return {
        "id": staff.id,
        "name": staff.name,
        "role": staff.role,
        "work_order_count": staff.work_order_count,
        "distance_km": staff.distance_km,
        "personnel_category": staff.personnel_category,
        "user_work_describe": staff.user_work_describe,
    }


def search_work_orders(work_order_id: str) -> str:
    """根据工单编号检索工单及其派发信息,返回JSON格式。
    work_order_id: 工单编号,如 WO-20260709-001 或纯数字 1"""
    try:
        numeric_id = _parse_work_order_id(work_order_id)
    except (ValueError, IndexError):
        return json.dumps({"error": f"无效的工单编号格式: {work_order_id}", "data": None}, ensure_ascii=False)

    try:
        with mysql_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        wo.work_order_id,
                        wo.event_id,
                        wo.camera_id,
                        wo.camera_name,
                        wo.segment_id,
                        wo.segment_name,
                        wo.monitor_address,
                        wo.work_order_type,
                        wo.work_order_describe,
                        wo.work_order_img_url,
                        wo.work_order_rank,
                        wo.work_order_time,
                        wo.work_order_stage,
                        wo.work_order_status,
                        wo.ai_suggestion,
                        wo.scene_info,
                        wo.completed_at,
                        ou.order_user_id,
                        ou.user_id AS assignee_user_id,
                        ou.order_user_time,
                        ou.order_user_status,
                        u.user_name AS assignee_name,
                        u.user_type AS assignee_type
                    FROM work_orders wo
                    LEFT JOIN order_user ou ON wo.work_order_id = ou.work_order_id
                    LEFT JOIN users u ON ou.user_id = u.user_id
                    WHERE wo.work_order_id = %s
                    """,
                    (numeric_id,),
                )
                row = cursor.fetchone()

        if not row:
            return json.dumps({"error": f"未找到工单: {work_order_id}", "data": None}, ensure_ascii=False)

        work_order = {
            "work_order_id": row["work_order_id"],
            "event_id": row.get("event_id"),
            "camera_id": row.get("camera_id"),
            "camera_name": row.get("camera_name"),
            "segment_id": row.get("segment_id"),
            "segment_name": row.get("segment_name"),
            "monitor_address": row.get("monitor_address"),
            "work_order_type": row.get("work_order_type"),
            "work_order_describe": row.get("work_order_describe"),
            "work_order_img_url": row.get("work_order_img_url"),
            "work_order_rank": row.get("work_order_rank"),
            "work_order_time": _serialize_datetime(row.get("work_order_time")),
            "work_order_stage": row.get("work_order_stage"),
            "work_order_status": row.get("work_order_status"),
            "ai_suggestion": row.get("ai_suggestion"),
            "scene_info": row.get("scene_info"),
            "completed_at": _serialize_datetime(row.get("completed_at")),
        }

        assignment: Optional[dict[str, Any]] = None
        if row.get("assignee_user_id") is not None:
            assignment = {
                "order_user_id": row.get("order_user_id"),
                "user_id": row.get("assignee_user_id"),
                "user_name": row.get("assignee_name"),
                "user_type": row.get("assignee_type"),
                "order_user_time": _serialize_datetime(row.get("order_user_time")),
                "order_user_status": row.get("order_user_status"),
            }

        return json.dumps({"work_order": work_order, "assignment": assignment}, ensure_ascii=False)

    except Exception as exc:
        logger.exception("search_work_orders 查询失败")
        return json.dumps({"error": f"数据库查询异常: {exc}", "data": None}, ensure_ascii=False)


def query_work_orders(
    user_id: Optional[int] = None,
    stage: Optional[str] = None,
    event_level: Optional[str] = None,
    limit: int = 20,
) -> str:
    """查询工单列表,支持按用户、状态、等级筛选。
    user_id: 可选,筛选该人员对应类别的工单
    stage: 可选,工单阶段(unassigned/pending/processing/completed/ignored),默认返回所有未完成
    event_level: 可选,事件等级(low/medium/high)
    limit: 返回数量上限,默认20条"""
    try:
        work_orders = WorkOrderRepository.list_work_orders(user_id=user_id)
    except Exception as exc:
        logger.exception("query_work_orders 查询失败")
        return json.dumps({"error": f"查询工单列表失败: {exc}", "data": None}, ensure_ascii=False)

    if stage:
        work_orders = [wo for wo in work_orders if wo.status == stage]
    if event_level:
        work_orders = [wo for wo in work_orders if wo.event_level == event_level]
    if user_id is None and stage is None:
        work_orders = [wo for wo in work_orders if wo.status not in ("completed", "ignored")]

    work_orders = work_orders[:limit]

    items = []
    for wo in work_orders:
        items.append(_format_work_order_for_agent(wo))

    stats = {
        "total": len(items),
        "by_stage": {},
        "by_level": {},
    }
    for item in items:
        s = item["status"]
        stats["by_stage"][s] = stats["by_stage"].get(s, 0) + 1
        lv = item["event_level"]
        stats["by_level"][lv] = stats["by_level"].get(lv, 0) + 1

    return json.dumps({"work_orders": items, "stats": stats}, ensure_ascii=False)


def get_work_order_detail(work_order_id: str) -> str:
    """根据工单编号获取工单完整详情。
    返回事故描述、AI建议、现场信息、处理记录、派发人员等全部字段。"""
    try:
        wo = WorkOrderRepository.get_work_order(work_order_id)
    except Exception as exc:
        logger.exception("get_work_order_detail 查询失败")
        return json.dumps({"error": f"数据库查询异常: {exc}", "data": None}, ensure_ascii=False)

    if wo is None:
        return json.dumps({"error": f"未找到工单: {work_order_id}", "data": None}, ensure_ascii=False)

    detail = _format_work_order_for_agent(wo)
    detail["scene_images"] = wo.scene_images
    detail["process_images"] = wo.process_images

    reply_message = None
    if wo.process_message:
        reply_message = wo.process_message

    detail["reply_message"] = reply_message
    return json.dumps(detail, ensure_ascii=False)


def query_staff() -> str:
    """查询所有可派发人员信息。
    返回人员列表,包括姓名、角色、当前未处理工单数(work_order_count)、人员类别、工作描述等。
    work_order_count越小表示人员负载越低,更适合派发新工单。"""
    try:
        staff_list = WorkOrderRepository.list_staff()
    except Exception as exc:
        logger.exception("query_staff 查询失败")
        return json.dumps({"error": f"查询人员失败: {exc}", "data": None}, ensure_ascii=False)

    sorted_staff = sorted(staff_list, key=lambda s: s.work_order_count)
    avg_workload = round(sum(s.work_order_count for s in sorted_staff) / len(sorted_staff), 1) if sorted_staff else 0

    items = [_format_staff_for_agent(s) for s in sorted_staff]
    return json.dumps(
        {
            "staff": items,
            "summary": {
                "total": len(items),
                "avg_work_order_count": avg_workload,
                "min_work_order_count": sorted_staff[0].work_order_count if sorted_staff else 0,
                "max_work_order_count": sorted_staff[-1].work_order_count if sorted_staff else 0,
            },
        },
        ensure_ascii=False,
    )


def query_work_order_stats() -> str:
    """查询工单统计数据,按状态和等级汇总。
    返回各状态工单数量及各等级分布,用于宏观决策参考。"""
    try:
        work_orders = WorkOrderRepository.list_work_orders()
    except Exception as exc:
        logger.exception("query_work_order_stats 查询失败")
        return json.dumps({"error": f"统计查询失败: {exc}", "data": None}, ensure_ascii=False)

    by_stage: dict[str, int] = {}
    by_level: dict[str, int] = {}
    high_unresolved: list[dict[str, Any]] = []

    for wo in work_orders:
        s = wo.status
        by_stage[s] = by_stage.get(s, 0) + 1
        lv = wo.event_level
        by_level[lv] = by_level.get(lv, 0) + 1
        if lv == "high" and s not in ("completed", "ignored"):
            high_unresolved.append(_format_work_order_for_agent(wo))

    return json.dumps(
        {
            "total": len(work_orders),
            "by_stage": by_stage,
            "by_level": by_level,
            "unresolved_high_priority": high_unresolved[:5],
        },
        ensure_ascii=False,
    )


def suggest_handling(work_order_id: str) -> str:
    """分析指定工单并生成处置建议。
    基于工单状态、事故等级、现场信息和处置经验给出专业的分阶段建议。
    当前状态不同,建议侧重不同:未派发→推荐人员和优先级,待处理→处置步骤,处理中→升级条件,已完成/忽略→复盘要点。"""
    try:
        wo = WorkOrderRepository.get_work_order(work_order_id)
    except Exception as exc:
        logger.exception("suggest_handling 获取工单失败")
        return json.dumps({"error": f"获取工单失败: {exc}", "data": None}, ensure_ascii=False)

    if wo is None:
        return json.dumps({"error": f"未找到工单: {work_order_id}", "data": None}, ensure_ascii=False)

    detail = _format_work_order_for_agent(wo)
    status = detail["status"]
    level = detail["event_level"]
    accident_type = detail["accident_info"]
    scene_info = detail["scene_info"]
    description = detail["description"]

    rag_hint = ""
    try:
        from app.modules.agent.rag.rag_manager import rag_manager as rm
        search_query = f"{accident_type} {description}"
        rag_result = rm.search_rag(search_query)
        if rag_result:
            articles = []
            for law_title, items in rag_result.items():
                for item in items[:3]:
                    articles.append(f"[{item.get('chapter', '')}] {item.get('describe', '')}")
            if articles:
                rag_hint = "\n相关法规参考:\n" + "\n".join(articles)
    except Exception as rag_exc:
        logger.warning("suggest_handling RAG检索失败: %s", rag_exc)

    suggestion_context = {
        "work_order_id": wo.work_order_id,
        "accident_type": accident_type,
        "event_level": level,
        "current_stage": status,
        "description": description,
        "scene_info": scene_info,
        "location": detail["camera_name"],
        "monitor_address": detail["monitor_address"],
        "assignee": detail["assignee"],
        "existing_ai_suggestion": detail["ai_suggestion"],
        "process_message": detail.get("process_message"),
        "rag_reference": rag_hint,
    }

    stage_guidance = {
        "unassigned": "工单尚未派发,建议从优先级评估、推荐人员类别、预估响应时效三个维度给出建议。",
        "pending": "工单已派发待处理,建议从处置步骤、现场注意事项、需要协调的资源三个维度给出建议。",
        "processing": "工单正在处理中,建议从当前进展评估、是否需要升级、结案标准三个维度给出建议。",
        "completed": "工单已完成,建议从处置复盘、经验总结、预防措施三个维度给出建议。",
        "ignored": "工单已被忽略,建议从重新激活条件、历史原因分析、后续监控三个维度给出建议。",
    }

    suggestion_context["stage_guidance"] = stage_guidance.get(status, "请综合工单信息给出通用处置建议。")

    return json.dumps(suggestion_context, ensure_ascii=False)


def dispatch_work_order(work_order_id: str, user_id: int) -> str:
    """将指定工单派发给指定人员。
    work_order_id: 工单编号,如 WO-20260709-001
    user_id: 人员数字ID,从query_staff查询获得。
    派发前会校验工单状态和人员类别匹配,派发后工单状态变更为pending。"""
    try:
        result = WorkOrderRepository.dispatch_work_order(work_order_id, user_id)
    except ValueError as exc:
        return json.dumps({"error": f"派发失败: {exc}", "data": None}, ensure_ascii=False)
    except Exception as exc:
        logger.exception("dispatch_work_order 派发异常")
        return json.dumps({"error": f"派发异常: {exc}", "data": None}, ensure_ascii=False)

    if result is None:
        return json.dumps({"error": "派发失败,工单或人员不存在", "data": None}, ensure_ascii=False)

    dispatch_result = _format_work_order_for_agent(result)
    dispatch_result["message"] = f"工单 {work_order_id} 已成功派发,当前状态: {result.status}"
    return json.dumps(dispatch_result, ensure_ascii=False)


def answer_general_question(question: str) -> str:
    """回答关于交通管理、道路运营、工单处置流程、法规条例等通用问题。
    支持:交通事故分类标准、工单等级划分规则、派发流程说明、处置规范等。
    问题示例: 什么是工单等级、如何判断事故严重程度、派发流程是怎样的。"""
    rag_articles: list[str] = []
    try:
        from app.modules.agent.rag.rag_manager import rag_manager as rm
        rag_result = rm.search_rag(question)
        for law_title, items in rag_result.items():
            for item in items:
                rag_articles.append(
                    f"【{law_title}】{item.get('chapter', '')}: {item.get('describe', '')}"
                )
    except Exception as rag_exc:
        logger.warning("answer_general_question RAG检索失败: %s", rag_exc)

    return json.dumps(
        {
            "question": question,
            "rag_references": rag_articles,
            "hint": "请基于以上法规条款和交通管理知识,用简洁专业的语言回答用户问题。如无匹配法规,请基于通用知识回答并说明信息来源。",
        },
        ensure_ascii=False,
    )


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
