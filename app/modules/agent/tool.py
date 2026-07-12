import json
import logging
from typing import Any, Optional

from app.database import mysql_connection

logger = logging.getLogger(__name__)


def _parse_work_order_id(work_order_id: str) -> int:
    """从 'WO-20260709-001' 或纯数字字符串中提取数字ID。"""
    text = work_order_id.strip()
    if text.upper().startswith("WO-"):
        text = text.split("-")[-1]
    return int(text.lstrip("0") or "0")


def _serialize_datetime(value: Any) -> Optional[str]:
    if value is None:
        return None
    return value.strftime("%Y-%m-%d %H:%M:%S") if hasattr(value, "strftime") else str(value)


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

        result: dict[str, Any] = {
            "work_order": work_order,
            "assignment": assignment,
        }
        return json.dumps(result, ensure_ascii=False)

    except Exception as exc:
        logger.exception("search_work_orders 查询失败")
        return json.dumps({"error": f"数据库查询异常: {exc}", "data": None}, ensure_ascii=False)


def query_work_orders(user_id: Optional[int] = None) -> str:
    """查询当前所有工单信息。可选传入user_id筛选该人员对应类别的工单。
    返回工单列表,包括工单编号、事故描述、状态、等级、派发人员等。"""
    # TODO: 调用WorkOrderRepository.list_work_orders()获取工单列表
    # TODO: 按status/event_level/assignee/required_category格式化输出
    raise NotImplementedError("query_work_orders not implemented")


def get_work_order_detail(work_order_id: str) -> str:
    """根据工单编号获取工单详细信息。
    返回完整的工单详情,包括事故描述、AI建议、现场信息、处理记录等。"""
    # TODO: 调用WorkOrderRepository.get_work_order()获取单个工单
    # TODO: 格式化返回event_id, accident_info, event_level, status, camera_name, monitor_address, assignee, required_category, description, ai_suggestion, scene_info, scene_images, process_message等字段
    raise NotImplementedError("get_work_order_detail not implemented")


def query_staff() -> str:
    """查询所有可派发人员信息。
    返回人员列表,包括姓名、角色、空闲状态、人员类别、关联工单数等。"""
    # TODO: 调用WorkOrderRepository.list_staff()获取人员列表
    # TODO: 按id/name/role/status/personnel_category/distance_km格式化输出
    raise NotImplementedError("query_staff not implemented")


def suggest_handling(work_order_id: str) -> str:
    """分析指定工单并生成处置建议。
    基于事故等级、现场信息、路段情况和历史处置经验给出专业建议。"""
    # TODO: 调用WorkOrderRepository.get_work_order()获取工单详情
    # TODO: 根据event_level(high/medium/low)生成分级处置建议
    # TODO: 接入历史工单数据库,基于相似事故类型和路段进行智能推荐
    # TODO: 接入实时天气和路况数据增强建议准确性
    # TODO: 根据实时AI检测结果动态调整优先级
    # TODO: 生成多套可选方案供人工选择
    raise NotImplementedError("suggest_handling not implemented")


def dispatch_work_order(work_order_id: str, user_id: int) -> str:
    """将指定工单派发给指定人员。
    work_order_id: 工单编号,如 WO-20260709-001
    user_id: 人员数字ID,从query_staff查询获得"""
    # TODO: 调用WorkOrderRepository.get_work_order()获取工单,校验状态(未解决且未派发)
    # TODO: 调用WorkOrderRepository.dispatch_work_order()执行派发
    # TODO: 处理ValueError(类别不匹配)和通用异常
    raise NotImplementedError("dispatch_work_order not implemented")


def answer_general_question(question: str) -> str:
    """回答关于交通管理、道路运营、工单处置流程等通用问题。
    支持的问法包括: 什么是工单等级、如何判断事故严重程度、派发流程说明等。"""
    # TODO: 接入RAG知识库,从运维手册和SOP文档中检索答案
    # TODO: 接入实时交通态势感知数据,提供更精准的回答
    # TODO: 对接LLM推理引擎,利用大模型回答未覆盖的开放性问题
    raise NotImplementedError("answer_general_question not implemented")
