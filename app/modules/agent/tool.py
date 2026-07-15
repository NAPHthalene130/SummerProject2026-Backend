import base64
import json
import logging
from pathlib import Path
from typing import Any, Optional

from app.config import llm_settings
from app.database import mysql_connection
from app.repository.work_order_repository import (
    WorkOrderRepository,
    parse_work_order_id,
    rank_to_level,
)

logger = logging.getLogger(__name__)
_rag_manager_instance: Any = None
_vlm_client: Optional[Any] = None

_IMAGE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "orderImg"


def _get_vlm_client() -> Any:
    global _vlm_client
    if _vlm_client is None:
        from openai import OpenAI

        _vlm_client = OpenAI(
            base_url=llm_settings.url,
            api_key=llm_settings.api_key,
            timeout=60.0,
        )
    return _vlm_client


def _image_url_to_path(url: str) -> Optional[Path]:
    """Convert a DB-stored image URL to an absolute filesystem path."""
    if not url:
        return None
    filename = url.rsplit("/", 1)[-1] if "/" in url else url
    filepath = _IMAGE_DIR / filename
    return filepath if filepath.is_file() else None


def _load_images_base64(image_urls: list[str]) -> list[dict[str, str]]:
    """Load images from disk and return as base64 data URIs for VLM input."""
    images: list[dict[str, str]] = []
    for url in image_urls:
        path = _image_url_to_path(url)
        if path is None:
            images.append({"error": f"图片文件不存在: {url}"})
            continue
        try:
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            ext = path.suffix.lower().lstrip(".")
            mime = f"image/{ext}" if ext in ("jpg", "jpeg", "png", "webp") else "image/jpeg"
            images.append({"data": f"data:{mime};base64,{data}", "filename": path.name})
        except Exception as exc:
            logger.warning("加载图片失败 %s: %s", url, exc)
            images.append({"error": f"图片读取失败: {url}"})
    return images


def _call_vlm_with_images(images: list[dict[str, str]], prompt: str) -> str:
    """Send images + text prompt to the VLM and return the analysis result."""
    client = _get_vlm_client()
    content: list[dict] = [{"type": "text", "text": prompt}]
    for img in images:
        if "data" in img:
            content.append({
                "type": "image_url",
                "image_url": {"url": img["data"]},
            })

    try:
        response = client.chat.completions.create(
            model=llm_settings.model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=400,
            temperature=0.3,
        )
        return response.choices[0].message.content or "图片分析无返回内容"
    except Exception as exc:
        logger.warning("VLM图片分析调用失败: %s", exc)
        return f"图片分析调用失败: {exc}"


def _search_regulations(query: str) -> dict[str, list[dict[str, str]]]:
    global _rag_manager_instance
    if _rag_manager_instance is None:
        from app.modules.agent.rag.rag_manager import RagManager
        _rag_manager_instance = RagManager()
    return _rag_manager_instance.search_rag(query)


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
                        wo.work_order_id, wo.event_id, wo.camera_id, wo.camera_name,
                        wo.segment_id, wo.segment_name, wo.monitor_address,
                        wo.work_order_type, wo.work_order_describe, wo.work_order_img_url,
                        wo.work_order_rank, wo.work_order_time,
                        wo.work_order_stage, wo.work_order_status,
                        wo.ai_suggestion, wo.scene_info, wo.completed_at,
                        ou.order_user_id, ou.user_id AS assignee_user_id,
                        ou.order_user_time, ou.order_user_status,
                        u.user_name AS assignee_name, u.user_type AS assignee_type
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
            "assignee_name": row.get("assignee_name"),
            "assignee_user_id": row.get("assignee_user_id"),
            "assignee_type": row.get("assignee_type"),
        }

        return json.dumps(work_order, ensure_ascii=False)

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
    stage: 可选,工单阶段(unassigned/pending/processing/completed/ignored)。未指定时默认返回所有未结案(排除completed/ignored)
    event_level: 可选,事件等级(low/medium/high)
    limit: 返回数量上限,默认20条"""
    try:
        work_orders = WorkOrderRepository.list_work_orders(user_id=user_id)
    except Exception as exc:
        logger.exception("query_work_orders 查询失败")
        return json.dumps({"error": f"查询工单列表失败: {exc}", "data": None}, ensure_ascii=False)

    if stage:
        work_orders = [wo for wo in work_orders if wo.status == stage]
    else:
        work_orders = [
            wo for wo in work_orders if wo.status not in ("completed", "ignored")
        ]
    if event_level:
        work_orders = [wo for wo in work_orders if wo.event_level == event_level]

    work_orders = work_orders[:limit]

    items = []
    for wo in work_orders:
        item = _format_work_order_for_agent(wo)
        item["scene_image_count"] = len(wo.scene_images)
        items.append(item)

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
    detail["scene_image_count"] = len(wo.scene_images)
    detail["process_image_count"] = len(wo.process_images or [])
    detail["has_process_message"] = bool(wo.process_message)
    detail["has_completed_at"] = bool(wo.completed_at)

    return json.dumps(detail, ensure_ascii=False)


def query_staff() -> str:
    """查询所有可派发人员信息。
    work_order_count越小表示人员负载越低,更适合派发新工单。"""
    try:
        staff_list = WorkOrderRepository.list_staff()
    except Exception as exc:
        logger.exception("query_staff 查询失败")
        return json.dumps({"error": f"查询人员失败: {exc}", "data": None}, ensure_ascii=False)

    sorted_staff = sorted(staff_list, key=lambda s: s.work_order_count)
    items = [_format_staff_for_agent(s) for s in sorted_staff]

    return json.dumps(
        {
            "staff": items,
            "total": len(items),
            "avg_workload": round(sum(s.work_order_count for s in sorted_staff) / len(sorted_staff), 1) if sorted_staff else 0,
        },
        ensure_ascii=False,
    )


def query_work_order_stats() -> str:
    """查询工单统计数据,按状态和等级汇总。"""
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
            high_unresolved.append({
                "id": wo.work_order_id,
                "type": wo.accident_info,
                "level": wo.event_level,
                "stage": s,
                "location": wo.camera_name,
            })

    return json.dumps(
        {
            "total": len(work_orders),
            "by_stage": by_stage,
            "by_level": by_level,
            "high_priority_unresolved": high_unresolved[:5],
        },
        ensure_ascii=False,
    )


def suggest_handling(work_order_id: str) -> str:
    """分析指定工单并生成处置建议上下文(JSON)。
    未派发→推荐人员与优先级,待处理→处置步骤,处理中→升级条件,已完成/忽略→复盘。
    自动加载工单关联的现场照片,调用视觉模型分析画面内容,将分析结果一并返回。
    注意:本工具含图片识别,耗时约5-15秒;返回的是上下文JSON,需由你据此组织最终中文建议(3-5条步骤)。"""
    try:
        wo = WorkOrderRepository.get_work_order(work_order_id)
    except Exception as exc:
        logger.exception("suggest_handling 获取工单失败")
        return json.dumps({"error": f"获取工单失败: {exc}", "data": None}, ensure_ascii=False)

    if wo is None:
        return json.dumps({"error": f"未找到工单: {work_order_id}", "data": None}, ensure_ascii=False)

    rag_articles: list[dict[str, str]] = []
    try:
        search_query = f"{wo.accident_info} {wo.description}"
        rag_result = _search_regulations(search_query)
        if rag_result:
            for law_title, items in rag_result.items():
                for item in items[:3]:
                    rag_articles.append({
                        "law": law_title,
                        "article": item.get("chapter", ""),
                        "excerpt": item.get("describe", "")[:120],
                    })
    except Exception as rag_exc:
        logger.warning("suggest_handling RAG检索失败: %s", rag_exc)

    image_analysis = ""
    scene_images = list(getattr(wo, "scene_images", None) or [])
    if scene_images:
        loaded = _load_images_base64(scene_images)
        valid_images = [img for img in loaded if "data" in img]
        if valid_images:
            img_prompt = (
                "你是交通监控图片分析助手。请用3-5句中文描述以下现场照片中的关键信息,"
                "重点观察: 1)车辆数量、位置和状态(是否有碰撞变形); "
                "2)道路占用情况(哪几条车道被阻断); "
                "3)现场是否有人员活动(围观、施救、摆放警示标志); "
                "4)天气和光照条件。"
                "只描述你能清晰看到的内容,不确定的请说明'画面中无法确认'。"
            )
            image_analysis = _call_vlm_with_images(valid_images, img_prompt)

    stage_guidance = {
        "unassigned": "工单尚未派发,从优先级、推荐人员类别、响应时效给出建议",
        "pending": "工单已派发待处理,从处置步骤、现场注意事项、协调资源给出建议",
        "processing": "工单处理中,从进展评估、升级条件、结案标准给出建议",
        "completed": "工单已完成,总结处置经验",
        "ignored": "工单已忽略,评估重新激活条件",
    }

    return json.dumps({
        "work_order_id": wo.work_order_id,
        "accident_type": wo.accident_info,
        "event_level": wo.event_level,
        "current_stage": wo.status,
        "description": wo.description,
        "scene_info": wo.scene_info,
        "location": f"{wo.camera_name} ({wo.monitor_address})",
        "assignee": wo.assignee,
        "process_message": wo.process_message,
        "scene_image_count": len(scene_images),
        "image_analysis": image_analysis,
        "stage_guidance": stage_guidance.get(wo.status, "综合工单信息给出通用建议"),
        "rag_references": rag_articles,
    }, ensure_ascii=False)


def dispatch_work_order(work_order_id: str, user_id: int) -> str:
    """将指定工单派发给指定人员。
    派发前先从数据库重新获取工单最新状态,确认仍为unassigned后再执行派发。
    work_order_id: 工单编号,如 WO-20260709-001
    user_id: 人员数字ID,从query_staff查询获得。"""
    latest = WorkOrderRepository.get_work_order(work_order_id)
    if latest is None:
        return json.dumps({"error": f"工单不存在: {work_order_id}", "data": None}, ensure_ascii=False)

    if latest.status != "unassigned":
        return json.dumps(
            {
                "error": f"工单当前状态为 '{latest.status}',"
                f"仅未派发(unassigned)状态可派发。可能已被他人派发,请刷新后重试。",
                "data": {"work_order_id": latest.work_order_id, "status": latest.status, "assignee": latest.assignee},
            },
            ensure_ascii=False,
        )

    try:
        result = WorkOrderRepository.dispatch_work_order(work_order_id, user_id)
    except ValueError as exc:
        return json.dumps({"error": f"派发失败: {exc}", "data": None}, ensure_ascii=False)
    except Exception as exc:
        logger.exception("dispatch_work_order 派发异常")
        return json.dumps({"error": f"派发异常: {exc}", "data": None}, ensure_ascii=False)

    if result is None:
        return json.dumps({"error": "派发失败,工单或人员不存在", "data": None}, ensure_ascii=False)

    return json.dumps({
        "work_order_id": result.work_order_id,
        "status": result.status,
        "assignee": result.assignee,
        "message": f"工单 {work_order_id} 已成功派发至 {result.assignee},当前状态: {result.status}",
    }, ensure_ascii=False)


def _select_best_staff(
    staff_list: list[Any],
    category: str,
    workload_override: dict[int, int],
) -> Optional[Any]:
    """在同类别人员中选取(当前负载+本批次已派发增量)最低者,实现批次内负载均衡。"""
    candidates = [s for s in staff_list if s.personnel_category == category]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda s: (
            s.work_order_count + workload_override.get(int(s.id), 0),
            s.distance_km,
        ),
    )


_BATCH_PRESENTATION_HINT = (
        "请用1-2句话回复,例如:'区间300~330内共派发28单,跳过0单,失败0单。'"
        "禁止输出Markdown表格或'结论/依据/未核验项'等结构。"
    )


def _split_batch_details(details: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    items = [item for item in details if item.get("result") in {"success", "planned", "ignored"}]
    skipped_items = [item for item in details if item.get("result") == "skipped"]
    failed_items = [item for item in details if item.get("result") == "failed"]
    return items, skipped_items, failed_items


def _fallback_candidates_from(skipped_items: list[dict[str, Any]], failed_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in skipped_items + failed_items:
        work_order_id = item.get("work_order_id")
        if not work_order_id:
            continue
        candidates.append({
            "work_order_id": work_order_id,
            "type": item.get("type"),
            "level": item.get("level"),
            "required_category": item.get("required_category"),
            "reason": item.get("reason", "无法处理"),
        })
    return candidates


def _structured_batch_payload(
    *,
    tool_name: str,
    mode: str,
    action: str,
    total: int,
    success: int = 0,
    skipped: int = 0,
    failed: int = 0,
    details: list[dict[str, Any]],
    presentation_hint: str,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    items, skipped_items, failed_items = _split_batch_details(details)
    fallback_candidates = _fallback_candidates_from(skipped_items, failed_items)
    payload: dict[str, Any] = {
        "ok": failed == 0,
        "tool": tool_name,
        "mode": mode,
        "action": action,
        "summary": {"total": total, "success": success, "skipped": skipped, "failed": failed},
        "affected_work_order_ids": [item["work_order_id"] for item in items if item.get("result") in {"success", "ignored"}],
        "items": items,
        "skipped_items": skipped_items,
        "failed_items": failed_items,
        "fallback_candidates": fallback_candidates,
        "next_actions": [
            {"tool": "batch_ignore_work_orders", "reason": "存在无法自动分配的工单", "candidate_count": len(fallback_candidates)}
        ] if fallback_candidates else [],
        "needs_manual_fallback": bool(fallback_candidates),
        "fallback_hint": (
            "若用户确认这些跳过/失败工单无需继续处理,可调用 batch_ignore_work_orders "
            "按 fallback_candidates 或筛选条件批量忽略。" if fallback_candidates else ""
        ),
        "presentation_hint": presentation_hint,
        "total": total,
        "success": success,
        "skipped": skipped,
        "failed": failed,
        "details": details,
    }
    if extra:
        payload.update(extra)
    return payload


def _plan_unassigned_orders(
    orders: list[Any],
    staff_list: list[Any],
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    workload_override: dict[int, int] = {}
    planned_count = 0
    skip_count = 0

    for wo in orders:
        category = wo.required_category or "traffic_police"
        best = _select_best_staff(staff_list, category, workload_override)
        if best is None:
            details.append({
                "work_order_id": wo.work_order_id,
                "type": wo.accident_info,
                "level": wo.event_level,
                "required_category": category,
                "result": "skipped",
                "reason": f"无{category}类别可用人员",
            })
            skip_count += 1
            continue

        details.append({
            "work_order_id": wo.work_order_id,
            "type": wo.accident_info,
            "level": wo.event_level,
            "required_category": category,
            "assigned_to": best.name,
            "staff_id": int(best.id),
            "result": "planned",
        })
        planned_count += 1
        workload_override[int(best.id)] = workload_override.get(int(best.id), 0) + 1

    return {
        "total": len(orders),
        "planned": planned_count,
        "skipped": skip_count,
        "details": details,
    }


def _dispatch_unassigned_orders(
    orders: list[Any],
    staff_list: list[Any],
    *,
    tool_name: str = "batch_dispatch_unassigned",
    action: str = "batch_dispatch",
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    plan = _plan_unassigned_orders(orders, staff_list)
    results: list[dict[str, Any]] = []
    success_count = 0
    fail_count = 0

    order_by_id = {str(wo.work_order_id): wo for wo in orders}
    for planned in plan["details"]:
        if planned.get("result") == "skipped":
            results.append(planned)
            continue
        if planned.get("result") != "planned":
            continue
        wo = order_by_id[str(planned["work_order_id"])]
        try:
            dispatch_result = WorkOrderRepository.dispatch_work_order(
                str(wo.work_order_id), int(planned["staff_id"])
            )
        except ValueError as exc:
            results.append({
                "work_order_id": wo.work_order_id,
                "type": wo.accident_info,
                "level": wo.event_level,
                "required_category": planned.get("required_category"),
                "result": "failed",
                "reason": str(exc),
            })
            fail_count += 1
            continue
        except Exception as exc:
            logger.exception("batch dispatch 派发异常 work_order=%s", wo.work_order_id)
            results.append({
                "work_order_id": wo.work_order_id,
                "type": wo.accident_info,
                "level": wo.event_level,
                "required_category": planned.get("required_category"),
                "result": "failed",
                "reason": str(exc),
            })
            fail_count += 1
            continue

        if dispatch_result is None:
            results.append({
                "work_order_id": wo.work_order_id,
                "type": wo.accident_info,
                "level": wo.event_level,
                "required_category": planned.get("required_category"),
                "result": "failed",
                "reason": "派发返回空结果",
            })
            fail_count += 1
        else:
            results.append({
                "work_order_id": dispatch_result.work_order_id,
                "type": dispatch_result.accident_info,
                "level": dispatch_result.event_level,
                "assigned_to": dispatch_result.assignee,
                "staff_id": int(planned["staff_id"]),
                "result": "success",
            })
            success_count += 1

    return _structured_batch_payload(
        tool_name=tool_name,
        mode="write",
        action=action,
        total=len(orders),
        success=success_count,
        skipped=plan["skipped"],
        failed=fail_count,
        details=results,
        presentation_hint=_BATCH_PRESENTATION_HINT,
        extra=extra,
    )


def dry_run_batch_dispatch(
    start_id: Optional[int] = None,
    end_id: Optional[int] = None,
    stage: str = "unassigned",
    event_level: Optional[str] = None,
    required_category: Optional[str] = None,
) -> str:
    """批量分配预演:只计算哪些工单会被分配给哪些人员,不修改数据库。
    start_id/end_id: 可选编号区间,必须同时提供。
    stage: 默认预演未派发(unassigned)工单。
    event_level: 可选等级过滤(low/medium/high)。
    required_category: 可选人员类别过滤。
    适用于正式批量派发前的预检查和风险评估。"""
    allowed_stages = {"unassigned", "pending", "processing"}
    if stage not in allowed_stages:
        return json.dumps({
            "ok": False, "tool": "dry_run_batch_dispatch", "mode": "dry_run",
            "error": f"不支持预演阶段 '{stage}'", "total": 0, "success": 0,
        }, ensure_ascii=False)

    lo: Optional[int] = None
    hi: Optional[int] = None
    if start_id is not None or end_id is not None:
        if start_id is None or end_id is None:
            return json.dumps({
                "ok": False, "tool": "dry_run_batch_dispatch", "mode": "dry_run",
                "error": "编号区间必须同时提供 start_id 和 end_id", "total": 0, "success": 0,
            }, ensure_ascii=False)
        lo, hi = sorted((int(start_id), int(end_id)))

    try:
        all_orders = WorkOrderRepository.list_work_orders()
        staff_list = WorkOrderRepository.list_staff()
    except Exception as exc:
        logger.exception("dry_run_batch_dispatch 查询失败")
        return json.dumps({"error": f"批量分配预演失败: {exc}", "data": None}, ensure_ascii=False)

    selected: list[Any] = []
    for wo in all_orders:
        try:
            numeric_id = parse_work_order_id(wo.work_order_id)
        except (TypeError, ValueError):
            continue
        if lo is not None and hi is not None and not (lo <= numeric_id <= hi):
            continue
        if wo.status != stage:
            continue
        if not _matches_optional_filter(wo.event_level, event_level):
            continue
        if not _matches_optional_filter(getattr(wo, "required_category", None), required_category):
            continue
        selected.append(wo)

    plan = _plan_unassigned_orders(selected, staff_list)
    payload = _structured_batch_payload(
        tool_name="dry_run_batch_dispatch",
        mode="dry_run",
        action="dry_run_batch_dispatch",
        total=plan["total"],
        success=plan["planned"],
        skipped=plan["skipped"],
        failed=0,
        details=plan["details"],
        presentation_hint="请用1句话说明预演结果,例如:'预演发现可分配250单,无法自动分配200单。'不要声称已执行数据库修改。",
        extra={
            "range": [lo, hi] if lo is not None and hi is not None else None,
            "filters": {"stage": stage, "event_level": event_level, "required_category": required_category},
            "would_modify_database": False,
        },
    )
    return json.dumps(payload, ensure_ascii=False)


def batch_dispatch_unassigned() -> str:
    """批量派发/批量分配:将所有未派发(unassigned)工单自动匹配同类别负载最低的人员并派发。
    无需参数,自动完成查询匹配派发全流程,返回每条工单的派发结果摘要。
    适用于\"批量处理/全部派发/批量派发/批量分配/全部分配/自动指派\"等不指定编号范围的批量指令——调用一次即可,不要逐个派发。"""
    try:
        all_orders = WorkOrderRepository.list_work_orders()
        staff_list = WorkOrderRepository.list_staff()
    except Exception as exc:
        logger.exception("batch_dispatch_unassigned 查询失败")
        return json.dumps({"error": f"批量派发查询失败: {exc}", "data": None}, ensure_ascii=False)

    unassigned = [wo for wo in all_orders if wo.status == "unassigned"]
    if not unassigned:
        return json.dumps({"message": "当前没有未派发工单", "total": 0, "success": 0}, ensure_ascii=False)
    if not staff_list:
        return json.dumps({"error": "无可派发人员", "total": len(unassigned)}, ensure_ascii=False)

    summary = _dispatch_unassigned_orders(unassigned, staff_list)
    return json.dumps(summary, ensure_ascii=False)


def dispatch_work_order_range(start_id: int, end_id: int) -> str:
    """批量派发/批量分配指定编号区间内的所有未派发工单(自动按类别匹配负载最低人员)。
    start_id: 起始工单编号(含),如 300
    end_id: 结束工单编号(含),如 330
    一次性派发区间内所有 unassigned 工单;非 unassigned 状态自动跳过。
    适用于\"把300~330的工单派发/分配/指派\"这类指定编号区间的批量指令——调用一次即可,不要逐个展开派发。"""
    lo, hi = sorted((int(start_id), int(end_id)))
    try:
        all_orders = WorkOrderRepository.list_work_orders()
        staff_list = WorkOrderRepository.list_staff()
    except Exception as exc:
        logger.exception("dispatch_work_order_range 查询失败")
        return json.dumps({"error": f"区间派发查询失败: {exc}", "data": None}, ensure_ascii=False)

    in_range: list[Any] = []
    for wo in all_orders:
        try:
            num = parse_work_order_id(wo.work_order_id)
        except (ValueError, TypeError):
            continue
        if lo <= num <= hi and wo.status == "unassigned":
            in_range.append(wo)

    if not in_range:
        return json.dumps({
            "message": f"区间 {lo}~{hi} 内没有未派发工单",
            "range": [lo, hi],
            "total": 0,
            "success": 0,
        }, ensure_ascii=False)
    if not staff_list:
        return json.dumps({
            "error": "无可派发人员",
            "range": [lo, hi],
            "total": len(in_range),
        }, ensure_ascii=False)

    summary = _dispatch_unassigned_orders(
        in_range,
        staff_list,
        tool_name="dispatch_work_order_range",
        action="dispatch_work_order_range",
        extra={"range": [lo, hi]},
    )
    return json.dumps(summary, ensure_ascii=False)


def _matches_optional_filter(value: Any, expected: Optional[str]) -> bool:
    return expected is None or value == expected


def batch_ignore_work_orders(
    start_id: Optional[int] = None,
    end_id: Optional[int] = None,
    stage: str = "unassigned",
    event_level: Optional[str] = None,
    required_category: Optional[str] = None,
    reason: str = "无法处理,批量忽略。",
) -> str:
    """批量忽略无法处理的工单。
    start_id: 可选,起始工单编号(含),如300;与end_id同时提供时仅处理该编号区间。
    end_id: 可选,结束工单编号(含),如330;与start_id同时提供时仅处理该编号区间。
    stage: 要忽略的工单阶段,默认只忽略未派发(unassigned);可传pending/processing处理极端无法继续处理的情况。
    event_level: 可选,仅忽略指定等级(low/medium/high)。
    required_category: 可选,仅忽略指定人员类别无法处理的工单。
    reason: 忽略原因,会写入处理记录。
    适用于\"无法处理/没有可用人员/不再处理/批量忽略/全部忽略\"等兜底指令。"""
    allowed_stages = {"unassigned", "pending", "processing"}
    if stage not in allowed_stages:
        return json.dumps({
            "error": f"不允许批量忽略阶段 '{stage}',仅支持 {sorted(allowed_stages)}",
            "total": 0, "ignored": 0,
        }, ensure_ascii=False)

    lo: Optional[int] = None
    hi: Optional[int] = None
    if start_id is not None or end_id is not None:
        if start_id is None or end_id is None:
            return json.dumps({
                "error": "编号区间必须同时提供 start_id 和 end_id",
                "total": 0, "ignored": 0,
            }, ensure_ascii=False)
        lo, hi = sorted((int(start_id), int(end_id)))

    try:
        all_orders = WorkOrderRepository.list_work_orders()
    except Exception as exc:
        logger.exception("batch_ignore_work_orders 查询失败")
        return json.dumps({"error": f"批量忽略查询失败: {exc}", "data": None}, ensure_ascii=False)

    candidates: list[Any] = []
    skipped: list[dict[str, Any]] = []
    for wo in all_orders:
        try:
            numeric_id = parse_work_order_id(wo.work_order_id)
        except (TypeError, ValueError):
            skipped.append({
                "work_order_id": getattr(wo, "work_order_id", ""),
                "result": "skipped",
                "reason": "工单编号无法解析",
            })
            continue

        in_range = lo is None or hi is None or lo <= numeric_id <= hi
        if not in_range:
            continue
        if not _matches_optional_filter(wo.status, stage):
            continue
        if not _matches_optional_filter(wo.event_level, event_level):
            continue
        if not _matches_optional_filter(getattr(wo, "required_category", None), required_category):
            continue
        candidates.append(wo)

    results: list[dict[str, Any]] = []
    ignored_count = 0
    failed_count = 0
    normalized_reason = reason.strip() or "无法处理,批量忽略。"

    for wo in candidates:
        previous_stage = wo.status
        try:
            updated = WorkOrderRepository.update_work_order_status(
                str(wo.work_order_id),
                "ignored",
                process_message=normalized_reason,
            )
        except Exception as exc:
            logger.exception("batch_ignore_work_orders 忽略异常 work_order=%s", wo.work_order_id)
            results.append({
                "work_order_id": wo.work_order_id,
                "result": "failed",
                "reason": str(exc),
            })
            failed_count += 1
            continue

        if updated is None:
            results.append({
                "work_order_id": wo.work_order_id,
                "result": "failed",
                "reason": "更新返回空结果",
            })
            failed_count += 1
            continue

        results.append({
            "work_order_id": updated.work_order_id,
            "type": updated.accident_info,
            "level": updated.event_level,
            "previous_stage": previous_stage,
            "result": "ignored",
            "reason": normalized_reason,
        })
        ignored_count += 1

    payload = _structured_batch_payload(
        tool_name="batch_ignore_work_orders",
        mode="write",
        action="batch_ignore",
        total=len(candidates),
        success=ignored_count,
        skipped=len(skipped),
        failed=failed_count,
        details=results + skipped,
        presentation_hint="请用1句话回复,例如:'已将无法处理的未派发工单批量忽略,共忽略N单,失败M单。'禁止输出Markdown表格。",
        extra={
            "range": [lo, hi] if lo is not None and hi is not None else None,
            "filters": {"stage": stage, "event_level": event_level, "required_category": required_category},
            "ignored": ignored_count,
        },
    )
    return json.dumps(payload, ensure_ascii=False)


def answer_general_question(question: str) -> str:
    """回答交通管理法规、工单处置流程等通用问题。
    本工具仅从法规知识库检索相关条款作为依据(返回rag_references),不直接生成最终答案;
    需由你据此组织最终中文回答,并在无匹配时说明"知识库未检索到相关条款"。"""
    rag_articles: list[dict[str, str]] = []
    try:
        rag_result = _search_regulations(question)
        for law_title, items in rag_result.items():
            for item in items:
                rag_articles.append({
                    "law": law_title,
                    "article": item.get("chapter", ""),
                    "excerpt": item.get("describe", "")[:150],
                })
    except Exception as rag_exc:
        logger.warning("answer_general_question RAG检索失败: %s", rag_exc)

    return json.dumps({
        "question": question,
        "rag_references": rag_articles,
        "note": "仅将rag_references中实际返回的条款作为依据,无匹配时说明知识库未检索到",
    }, ensure_ascii=False)


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
