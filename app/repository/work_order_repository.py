from datetime import datetime
from typing import Any, Optional

from app.database import mysql_connection
from app.models.user import StaffResponse
from app.models.work_order import WorkOrderItemResponse, WorkOrderStatus


UNRESOLVED_STATUSES = {"unassigned", "pending", "processing"}
SOLVED_STATUSES = {"completed", "false_alarm"}


def parse_work_order_id(work_order_id: str) -> int:
    if work_order_id.isdigit():
        return int(work_order_id)

    return int(work_order_id.rsplit("-", 1)[-1])


def format_datetime(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def format_work_order_code(work_order_id: int, work_order_time: Any) -> str:
    if isinstance(work_order_time, datetime):
        date_part = work_order_time.strftime("%Y%m%d")
    else:
        date_part = "20260709"
    return f"WO-{date_part}-{work_order_id:03d}"


def rank_to_level(rank: int) -> str:
    if rank >= 3:
        return "high"
    if rank == 2:
        return "medium"
    return "low"


def split_images(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


class WorkOrderRepository:
    @staticmethod
    def list_work_orders() -> list[WorkOrderItemResponse]:
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                      wo.*,
                      u.user_name AS assignee,
                      reply.work_order_reply_msg,
                      reply.work_order_reply_img_url,
                      reply.work_order_reply_time
                    FROM work_orders wo
                    LEFT JOIN order_user ou ON ou.work_order_id = wo.work_order_id
                    LEFT JOIN users u ON u.user_id = ou.user_id
                    LEFT JOIN work_order_replies reply
                      ON reply.work_order_reply_id = (
                        SELECT r.work_order_reply_id
                        FROM work_order_replies r
                        WHERE r.work_order_id = wo.work_order_id
                        ORDER BY r.work_order_reply_time DESC, r.work_order_reply_id DESC
                        LIMIT 1
                      )
                    ORDER BY wo.work_order_time DESC, wo.work_order_id DESC
                    """
                )
                return [WorkOrderRepository._row_to_response(row) for row in cursor.fetchall()]

    @staticmethod
    def get_work_order(work_order_id: str) -> Optional[WorkOrderItemResponse]:
        numeric_id = parse_work_order_id(work_order_id)
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                      wo.*,
                      u.user_name AS assignee,
                      reply.work_order_reply_msg,
                      reply.work_order_reply_img_url,
                      reply.work_order_reply_time
                    FROM work_orders wo
                    LEFT JOIN order_user ou ON ou.work_order_id = wo.work_order_id
                    LEFT JOIN users u ON u.user_id = ou.user_id
                    LEFT JOIN work_order_replies reply
                      ON reply.work_order_reply_id = (
                        SELECT r.work_order_reply_id
                        FROM work_order_replies r
                        WHERE r.work_order_id = wo.work_order_id
                        ORDER BY r.work_order_reply_time DESC, r.work_order_reply_id DESC
                        LIMIT 1
                      )
                    WHERE wo.work_order_id = %s
                    """,
                    (numeric_id,),
                )
                row = cursor.fetchone()
                return WorkOrderRepository._row_to_response(row) if row else None

    @staticmethod
    def list_staff() -> list[StaffResponse]:
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                      u.user_id,
                      u.user_name,
                      u.user_type,
                      COUNT(
                        CASE
                          WHEN wo.work_order_status IN ('pending', 'processing') THEN 1
                        END
                      ) AS active_order_count
                    FROM users u
                    LEFT JOIN order_user ou ON ou.user_id = u.user_id
                    LEFT JOIN work_orders wo ON wo.work_order_id = ou.work_order_id
                    GROUP BY u.user_id, u.user_name, u.user_type
                    ORDER BY u.user_id
                    """
                )
                staff: list[StaffResponse] = []
                for row in cursor.fetchall():
                    active_count = int(row["active_order_count"] or 0)
                    user_id = int(row["user_id"])
                    staff.append(
                        StaffResponse(
                            id=str(user_id),
                            name=row["user_name"],
                            role=row["user_type"],
                            status="busy" if active_count else "idle",
                            distance_km=round(0.45 + (user_id % 5) * 0.35, 1),
                        )
                    )
                return staff

    @staticmethod
    def dispatch_work_order(work_order_id: str, user_id: int) -> Optional[WorkOrderItemResponse]:
        numeric_id = parse_work_order_id(work_order_id)
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO order_user (work_order_id, user_id, order_user_status)
                    VALUES (%s, %s, 'assigned')
                    ON DUPLICATE KEY UPDATE
                      user_id = VALUES(user_id),
                      order_user_status = VALUES(order_user_status),
                      order_user_time = CURRENT_TIMESTAMP
                    """,
                    (numeric_id, user_id),
                )
                cursor.execute(
                    """
                    UPDATE work_orders
                    SET work_order_status = 'pending',
                        work_order_is_solve = 0,
                        completed_at = NULL
                    WHERE work_order_id = %s
                    """,
                    (numeric_id,),
                )

        return WorkOrderRepository.get_work_order(str(numeric_id))

    @staticmethod
    def update_work_order_status(
        work_order_id: str,
        status: WorkOrderStatus,
        process_message: Optional[str] = None,
        process_image_url: Optional[str] = None,
    ) -> Optional[WorkOrderItemResponse]:
        numeric_id = parse_work_order_id(work_order_id)
        is_solved = status in SOLVED_STATUSES

        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE work_orders
                    SET work_order_status = %s,
                        work_order_is_solve = %s,
                        completed_at = CASE WHEN %s THEN CURRENT_TIMESTAMP ELSE completed_at END
                    WHERE work_order_id = %s
                    """,
                    (status, int(is_solved), int(is_solved), numeric_id),
                )
                cursor.execute(
                    """
                    UPDATE order_user
                    SET order_user_status = %s
                    WHERE work_order_id = %s
                    """,
                    (status, numeric_id),
                )

                if is_solved and process_message:
                    cursor.execute(
                        """
                        INSERT INTO work_order_replies (
                          work_order_id,
                          work_order_reply_img_url,
                          work_order_reply_msg,
                          work_order_reply_status
                        )
                        VALUES (%s, %s, %s, 1)
                        """,
                        (numeric_id, process_image_url or "", process_message),
                    )

        return WorkOrderRepository.get_work_order(str(numeric_id))

    @staticmethod
    def _row_to_response(row: dict[str, Any]) -> WorkOrderItemResponse:
        status = row.get("work_order_status") or (
            "completed" if row.get("work_order_is_solve") else "unassigned"
        )
        process_images = split_images(row.get("work_order_reply_img_url"))
        completed_at = row.get("completed_at") or (
            row.get("work_order_reply_time") if status in SOLVED_STATUSES else None
        )

        return WorkOrderItemResponse(
            work_order_id=format_work_order_code(row["work_order_id"], row.get("work_order_time")),
            event_id=row.get("event_id") or f"evt_db_{row['work_order_id']:03d}",
            camera_id=row.get("camera_id") or "",
            camera_name=row.get("camera_name") or "未关联摄像头",
            segment_id=row.get("segment_id") or "",
            segment_name=row.get("segment_name") or "",
            monitor_address=row.get("monitor_address") or "未填写监控地址",
            accident_info=row.get("work_order_type") or "未命名工单",
            event_time=format_datetime(row.get("work_order_time")) or "",
            event_level=rank_to_level(int(row.get("work_order_rank") or 0)),
            status=status,
            assignee=row.get("assignee"),
            description=row.get("work_order_describe") or "",
            ai_suggestion=row.get("ai_suggestion") or "建议联系现场人员确认情况，并按事件等级进行派发。",
            scene_images=split_images(row.get("work_order_img_url")),
            scene_info=row.get("scene_info") or "暂无现场补充信息。",
            process_message=row.get("work_order_reply_msg"),
            process_images=process_images or None,
            completed_at=format_datetime(completed_at),
        )
