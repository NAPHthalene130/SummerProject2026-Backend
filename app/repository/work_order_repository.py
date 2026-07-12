from datetime import datetime
from typing import Any, Optional

from app.database import mysql_connection
from app.models.user import StaffResponse
from app.models.work_order import WorkOrderItemResponse, WorkOrderStage, WorkOrderStatus


UNRESOLVED_STATUSES = {"unassigned", "pending", "processing"}
TERMINAL_STATUSES = {"completed", "ignored"}
WORK_ORDER_STATUS_BY_STAGE = {
    "unassigned": WorkOrderStatus.UNRESOLVED,
    "pending": WorkOrderStatus.UNRESOLVED,
    "processing": WorkOrderStatus.UNRESOLVED,
    "completed": WorkOrderStatus.RESOLVED,
    "ignored": WorkOrderStatus.IGNORED,
}


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
    def create_work_order(
        camera_id: str,
        camera_name: str,
        incident_type: str,
        description: str,
        rank: int,
        image_url: str = "",
        scene_info: str = "",
        required_category: str = "traffic_police",
    ) -> Optional[WorkOrderItemResponse]:
        event_id = f"evt_{camera_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        now = datetime.now()
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO work_orders (
                      event_id, camera_id, camera_name,
                      work_order_type, work_order_describe, work_order_img_url,
                      work_order_rank, work_order_stage, work_order_status,
                      work_order_time, scene_info, required_category
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'unassigned', 0, %s, %s, %s)
                    """,
                    (
                        event_id, camera_id, camera_name,
                        incident_type, description, image_url,
                        rank, now, scene_info, required_category,
                    ),
                )
                work_order_id = cursor.lastrowid
        if work_order_id:
            return WorkOrderRepository.get_work_order(str(work_order_id))
        return None

    @staticmethod
    def list_work_orders(user_id: Optional[int] = None) -> list[WorkOrderItemResponse]:
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                personnel_category = None
                if user_id is not None:
                    cursor.execute("SELECT personnel_category FROM users WHERE user_id=%s", (user_id,))
                    user = cursor.fetchone()
                    if user is None:
                        return []
                    personnel_category = user.get("personnel_category") or "traffic_police"
                sql = """
                    SELECT
                      wo.*,
                      ou.user_id AS assignee_user_id,
                      u.user_name AS assignee,
                      reply.work_order_reply_msg,
                      reply.work_order_reply_img_url,
                      reply.work_order_reply_time,
                      reply.work_order_reply_status,
                      reply.requested_status,
                      reply.review_message,
                      reply.reviewed_at
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
                """
                args = ()
                if personnel_category is not None:
                    sql += " WHERE wo.required_category = %s"
                    args = (personnel_category,)
                sql += " ORDER BY wo.work_order_time DESC, wo.work_order_id DESC"
                cursor.execute(sql, args)
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
                      ou.user_id AS assignee_user_id,
                      u.user_name AS assignee,
                      reply.work_order_reply_msg,
                      reply.work_order_reply_img_url,
                      reply.work_order_reply_time,
                      reply.work_order_reply_status,
                      reply.requested_status,
                      reply.review_message,
                      reply.reviewed_at
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
                      u.personnel_category,
                      u.user_work_describe,
                      COUNT(
                        CASE
                          WHEN wo.work_order_status = 0
                           AND wo.work_order_stage IN ('pending', 'processing') THEN 1
                        END
                      ) AS work_order_count
                    FROM users u
                    LEFT JOIN order_user ou ON ou.user_id = u.user_id
                    LEFT JOIN work_orders wo ON wo.work_order_id = ou.work_order_id
                    GROUP BY u.user_id, u.user_name, u.user_type, u.personnel_category, u.user_work_describe
                    ORDER BY u.user_id
                    """
                )
                staff: list[StaffResponse] = []
                for row in cursor.fetchall():
                    work_count = int(row["work_order_count"] or 0)
                    user_id = int(row["user_id"])
                    staff.append(
                        StaffResponse(
                            id=str(user_id),
                            name=row["user_name"],
                            role=row["user_type"],
                            work_order_count=work_count,
                            distance_km=round(0.45 + (user_id % 5) * 0.35, 1),
                            personnel_category=row.get("personnel_category") or "traffic_police",
                            user_work_describe=row.get("user_work_describe"),
                        )
                    )
                return staff

    @staticmethod
    def dispatch_work_order(work_order_id: str, user_id: int) -> Optional[WorkOrderItemResponse]:
        numeric_id = parse_work_order_id(work_order_id)
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT required_category FROM work_orders WHERE work_order_id=%s", (numeric_id,))
                order_row = cursor.fetchone()
                cursor.execute("SELECT personnel_category FROM users WHERE user_id=%s", (user_id,))
                user_row = cursor.fetchone()
                if not order_row or not user_row:
                    return None
                if (order_row.get("required_category") or "traffic_police") != (user_row.get("personnel_category") or "traffic_police"):
                    raise ValueError("人员类别与工单要求不匹配")
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
                    SET work_order_stage = 'pending',
                        work_order_status = 0,
                        completed_at = NULL
                    WHERE work_order_id = %s
                    """,
                    (numeric_id,),
                )

        return WorkOrderRepository.get_work_order(str(numeric_id))

    @staticmethod
    def update_work_order_status(
        work_order_id: str,
        status: WorkOrderStage,
        process_message: Optional[str] = None,
        process_image_url: Optional[str] = None,
    ) -> Optional[WorkOrderItemResponse]:
        numeric_id = parse_work_order_id(work_order_id)
        status_code = WORK_ORDER_STATUS_BY_STAGE[status]
        is_terminal = status in TERMINAL_STATUSES

        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE work_orders
                    SET work_order_stage = %s,
                        work_order_status = %s,
                        completed_at = CASE WHEN %s THEN CURRENT_TIMESTAMP ELSE NULL END
                    WHERE work_order_id = %s
                    """,
                    (status, int(status_code), int(is_terminal), numeric_id),
                )
                cursor.execute(
                    """
                    UPDATE order_user
                    SET order_user_status = %s
                    WHERE work_order_id = %s
                    """,
                    (status, numeric_id),
                )

                if is_terminal and process_message:
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
    def submit_mobile_feedback(
        work_order_id: str,
        user_id: int,
        requested_status: str,
        process_message: str,
        process_image_url: Optional[str] = None,
    ) -> Optional[WorkOrderItemResponse]:
        numeric_id = parse_work_order_id(work_order_id)
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT wo.required_category,u.personnel_category FROM work_orders wo "
                    "JOIN users u ON u.user_id=%s WHERE wo.work_order_id=%s",
                    (user_id, numeric_id),
                )
                row = cursor.fetchone()
                if row is None:
                    return None
                if (row.get("required_category") or "traffic_police") != (row.get("personnel_category") or "traffic_police"):
                    raise ValueError("只能提交本职责组工单的处置结果")
                cursor.execute(
                    "SELECT work_order_reply_id FROM work_order_replies "
                    "WHERE work_order_id=%s AND work_order_reply_status=0 ORDER BY work_order_reply_id DESC LIMIT 1",
                    (numeric_id,),
                )
                if cursor.fetchone():
                    raise ValueError("已有处置结果等待电脑端审核")
                cursor.execute(
                    """
                    INSERT INTO work_order_replies (
                      work_order_id, work_order_reply_img_url, work_order_reply_msg,
                      work_order_reply_status, requested_status, submitted_by_user_id
                    ) VALUES (%s,%s,%s,0,%s,%s)
                    """,
                    (numeric_id, process_image_url or "", process_message, requested_status, user_id),
                )
                cursor.execute(
                    "UPDATE work_orders SET work_order_stage='processing',work_order_status=0,completed_at=NULL WHERE work_order_id=%s",
                    (numeric_id,),
                )
        return WorkOrderRepository.get_work_order(str(numeric_id))

    @staticmethod
    def review_mobile_feedback(
        work_order_id: str,
        decision: str,
        review_message: Optional[str] = None,
    ) -> Optional[WorkOrderItemResponse]:
        numeric_id = parse_work_order_id(work_order_id)
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT work_order_reply_id,requested_status FROM work_order_replies "
                    "WHERE work_order_id=%s AND work_order_reply_status=0 ORDER BY work_order_reply_id DESC LIMIT 1",
                    (numeric_id,),
                )
                reply = cursor.fetchone()
                if reply is None:
                    return None
                approved = decision == "approve"
                cursor.execute(
                    "UPDATE work_order_replies SET work_order_reply_status=%s,review_message=%s,reviewed_at=CURRENT_TIMESTAMP "
                    "WHERE work_order_reply_id=%s",
                    (1 if approved else 2, review_message or ("审核通过" if approved else "审核未通过"), reply["work_order_reply_id"]),
                )
                if approved:
                    requested_status = reply.get("requested_status") or "completed"
                    status_code = WORK_ORDER_STATUS_BY_STAGE[requested_status]
                    cursor.execute(
                        "UPDATE work_orders SET work_order_stage=%s,work_order_status=%s,completed_at=CURRENT_TIMESTAMP WHERE work_order_id=%s",
                        (requested_status, int(status_code), numeric_id),
                    )
                    cursor.execute("UPDATE order_user SET order_user_status=%s WHERE work_order_id=%s", (requested_status, numeric_id))
                else:
                    cursor.execute(
                        "UPDATE work_orders SET work_order_stage='processing',work_order_status=0,completed_at=NULL WHERE work_order_id=%s",
                        (numeric_id,),
                    )
        return WorkOrderRepository.get_work_order(str(numeric_id))

    @staticmethod
    def _row_to_response(row: dict[str, Any]) -> WorkOrderItemResponse:
        status_code = WorkOrderStatus(int(row.get("work_order_status") or 0))
        status = str(row.get("work_order_stage") or "unassigned")
        if status_code == WorkOrderStatus.IGNORED:
            status = "ignored"
        elif status_code == WorkOrderStatus.RESOLVED:
            status = "completed"
        elif status not in UNRESOLVED_STATUSES:
            status = "unassigned"
        process_images = split_images(row.get("work_order_reply_img_url"))
        completed_at = row.get("completed_at") or (
            row.get("work_order_reply_time") if status in TERMINAL_STATUSES else None
        )
        reply_status = row.get("work_order_reply_status")
        feedback_review_status = "none"
        if reply_status == 0:
            feedback_review_status = "pending"
        elif reply_status == 1 and row.get("requested_status"):
            feedback_review_status = "approved"
        elif reply_status == 2:
            feedback_review_status = "rejected"

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
            work_order_status=status_code,
            assignee=row.get("assignee"),
            assignee_user_id=row.get("assignee_user_id"),
            description=row.get("work_order_describe") or "",
            ai_suggestion=row.get("ai_suggestion") or "建议联系现场人员确认情况，并按事件等级进行派发。",
            scene_images=split_images(row.get("work_order_img_url")),
            scene_info=row.get("scene_info") or "暂无现场补充信息。",
            process_message=row.get("work_order_reply_msg"),
            process_images=process_images or None,
            completed_at=format_datetime(completed_at),
            required_category=row.get("required_category") or "traffic_police",
            feedback_review_status=feedback_review_status,
            feedback_requested_status=row.get("requested_status"),
            feedback_review_message=row.get("review_message"),
        )
