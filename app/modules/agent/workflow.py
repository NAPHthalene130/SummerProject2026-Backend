from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from app.repository.work_order_repository import WorkOrderRepository

logger = logging.getLogger(__name__)


@dataclass
class WorkflowResult:
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class AgentWorkflow:
    """Agent工作流编排器,将多个Tool调用串联为完整业务场景。

    提供三个标准工作流:
    - diagnose: 工单全链路诊断(详情→建议→人员推荐)
    - brief: 态势简报(统计概览→高优工单→人员负载)
    - dispatch_recommend: 智能派发推荐(工单匹配→人员筛选→按负载排序)
    """

    @staticmethod
    def diagnose(work_order_id: str) -> WorkflowResult:
        """工单全链路诊断。

        流程: 获取工单详情 → 生成处置建议 → 匹配可用人员 → 输出完整诊断报告
        """
        try:
            wo = WorkOrderRepository.get_work_order(work_order_id)
            if wo is None:
                return WorkflowResult(success=False, error=f"工单不存在: {work_order_id}")

            staff_list = WorkOrderRepository.list_staff()
            matching_staff = [
                s for s in staff_list if s.personnel_category == wo.required_category
            ]
            matching_staff.sort(key=lambda s: s.work_order_count)

            stage_map = {
                "unassigned": "未派发",
                "pending": "待处理",
                "processing": "处理中",
                "completed": "已完成",
                "ignored": "已忽略",
            }

            level_map = {"low": "一般", "medium": "中等", "high": "紧急"}

            report = {
                "work_order_id": wo.work_order_id,
                "event_id": wo.event_id,
                "accident_type": wo.accident_info,
                "event_level": wo.event_level,
                "event_level_label": level_map.get(wo.event_level, "未知"),
                "current_stage": wo.status,
                "current_stage_label": stage_map.get(wo.status, "未知"),
                "location": f"{wo.camera_name} ({wo.monitor_address})",
                "description": wo.description,
                "scene_info": wo.scene_info,
                "ai_suggestion": wo.ai_suggestion,
                "assignee": wo.assignee,
                "process_message": wo.process_message,
                "event_time": wo.event_time,
                "completed_at": wo.completed_at,
                "recommended_actions": AgentWorkflow._build_action_plan(wo.status, wo.event_level),
                "available_staff": {
                    "matching_staff": [
                        {
                            "id": s.id, "name": s.name, "role": s.role,
                            "work_order_count": s.work_order_count,
                            "distance_km": s.distance_km,
                            "user_work_describe": s.user_work_describe,
                        }
                        for s in matching_staff[:5]
                    ],
                    "total_matching": len(matching_staff),
                },
            }

            return WorkflowResult(success=True, data=report)
        except Exception as exc:
            logger.exception("diagnose workflow failed")
            return WorkflowResult(success=False, error=str(exc))

    @staticmethod
    def brief() -> WorkflowResult:
        """态势简报。

        流程: 工单统计 → 高优先级未解决 → 人员负载汇总
        """
        try:
            work_orders = WorkOrderRepository.list_work_orders()
            staff_list = WorkOrderRepository.list_staff()

            by_stage: dict[str, int] = {}
            by_level: dict[str, int] = {}
            unresolved_high: list[dict[str, Any]] = []
            unresolved_medium: list[dict[str, Any]] = []

            for wo in work_orders:
                s = wo.status
                by_stage[s] = by_stage.get(s, 0) + 1
                lv = wo.event_level
                by_level[lv] = by_level.get(lv, 0) + 1
                if s not in ("completed", "ignored"):
                    item = {
                        "id": wo.work_order_id,
                        "type": wo.accident_info,
                        "level": wo.event_level,
                        "stage": s,
                        "location": wo.camera_name,
                        "assignee": wo.assignee,
                        "time": wo.event_time,
                    }
                    if lv == "high":
                        unresolved_high.append(item)
                    elif lv == "medium":
                        unresolved_medium.append(item)

            staff_workloads = [s.work_order_count for s in staff_list]
            total_workload = sum(staff_workloads)
            max_workload = max(staff_workloads) if staff_workloads else 0

            unresolved_total = sum(
                count for stage, count in by_stage.items() if stage not in ("completed", "ignored")
            )

            return WorkflowResult(
                success=True,
                data={
                    "summary": {
                        "total_work_orders": len(work_orders),
                        "unresolved_total": unresolved_total,
                        "resolved_total": by_stage.get("completed", 0) + by_stage.get("ignored", 0),
                        "staff_total": len(staff_list),
                        "staff_total_workload": total_workload,
                        "staff_max_workload": max_workload,
                        "staff_avg_workload": round(total_workload / len(staff_list), 1) if staff_list else 0,
                    },
                    "by_stage": by_stage,
                    "by_level": by_level,
                    "high_priority_unresolved": unresolved_high,
                    "medium_priority_unresolved": unresolved_medium[:10],
                },
            )
        except Exception as exc:
            logger.exception("brief workflow failed")
            return WorkflowResult(success=False, error=str(exc))

    @staticmethod
    def dispatch_recommend(work_order_id: str, top_k: int = 3) -> WorkflowResult:
        """智能派发推荐。

        流程: 获取工单 → 匹配同类别人员 → 按work_order_count升序(负载低优先) → 返回Top-K推荐
        """
        try:
            wo = WorkOrderRepository.get_work_order(work_order_id)
            if wo is None:
                return WorkflowResult(success=False, error=f"工单不存在: {work_order_id}")

            if wo.status != "unassigned":
                return WorkflowResult(
                    success=False,
                    error=f"工单当前状态为 '{wo.status}',仅未派发(unassigned)工单可派发",
                )

            staff_list = WorkOrderRepository.list_staff()
            candidates = [
                s
                for s in staff_list
                if s.personnel_category == wo.required_category
            ]
            candidates.sort(key=lambda s: (s.work_order_count, s.distance_km))
            top_k_candidates = candidates[:top_k]

            return WorkflowResult(
                success=True,
                data={
                    "work_order_id": wo.work_order_id,
                    "accident_type": wo.accident_info,
                    "event_level": wo.event_level,
                    "required_category": wo.required_category,
                    "location": wo.camera_name,
                    "total_candidates": len(candidates),
                    "recommendations": [
                        {
                            "rank": idx + 1,
                            "user_id": int(s.id),
                            "name": s.name,
                            "role": s.role,
                            "work_order_count": s.work_order_count,
                            "distance_km": s.distance_km,
                            "user_work_describe": s.user_work_describe,
                        }
                        for idx, s in enumerate(top_k_candidates)
                    ],
                },
            )
        except Exception as exc:
            logger.exception("dispatch_recommend workflow failed")
            return WorkflowResult(success=False, error=str(exc))

    @staticmethod
    def _build_action_plan(stage: str, level: str) -> list[str]:
        plans: dict[str, dict[str, list[str]]] = {
            "unassigned": {
                "high": [
                    "【紧急】建议立即派发至当前负载最低(work_order_count最小)的同类别人员",
                    "通知现场值班负责人,启动加急响应",
                    "确认是否需要交警、消防或医疗协同",
                ],
                "medium": [
                    "建议15分钟内完成派发",
                    "选择同类别中work_order_count最低、距离最近的人员",
                    "确认是否需要调用周边摄像头辅助监控",
                ],
                "low": [
                    "建议30分钟内完成派发",
                    "可选择work_order_count较低的同类别人员处理",
                    "持续监控摄像头,关注事态变化",
                ],
            },
            "pending": {
                "high": [
                    "已超时未响应,建议升级通知至负责人",
                    "确认派发人员是否已到达现场",
                    "检查通信是否正常,必要时更换人员",
                ],
                "medium": [
                    "关注派发人员响应状态",
                    "如超过预期响应时间,发送提醒",
                ],
                "low": [
                    "跟踪人员响应进度",
                    "根据现场情况评估是否需要升级",
                ],
            },
            "processing": {
                "high": [
                    "密切监控处置进展,15分钟一次状态更新",
                    "确认现场是否需要额外支援",
                    "准备结案报告模板,处置完成后立即归档",
                ],
                "medium": [
                    "定期跟进处置进度",
                    "评估是否达到结案标准",
                ],
                "low": [
                    "跟踪处置进展,确保按时完成",
                ],
            },
            "completed": {
                "high": ["归档处置报告", "总结处置经验,更新应急预案", "检查是否有衍生风险"],
                "medium": ["归档工单", "记录处置要点供后续参考"],
                "low": ["归档工单"],
            },
            "ignored": {
                "high": ["评估是否需要重新激活", "分析忽略原因,避免同类遗漏"],
                "medium": ["评估是否需要重新激活"],
                "low": ["确认忽略决策合理性"],
            },
        }

        default_actions = ["参考标准处置流程", "根据现场情况灵活调整"]
        stage_plans = plans.get(stage, {})
        level_actions = stage_plans.get(level, stage_plans.get("medium", default_actions))
        return level_actions
