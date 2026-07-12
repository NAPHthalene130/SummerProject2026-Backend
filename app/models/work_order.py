from datetime import datetime
from enum import IntEnum
from typing import Literal, Optional

from pydantic import BaseModel


class WorkOrder(BaseModel):
    work_order_id: int
    work_order_type: str
    work_order_describe: str
    work_order_img_url: str
    work_order_rank: int
    work_order_time: datetime
    work_order_stage: str
    work_order_status: int


class WorkOrderStatus(IntEnum):
    UNRESOLVED = 0
    RESOLVED = 1
    IGNORED = 2


WorkOrderStage = Literal["unassigned", "pending", "processing", "completed", "ignored"]
WorkOrderLevel = Literal["low", "medium", "high"]


class WorkOrderItemResponse(BaseModel):
    work_order_id: str
    event_id: str
    camera_id: str
    camera_name: str
    segment_id: str
    segment_name: str
    monitor_address: str
    accident_info: str
    event_time: str
    event_level: WorkOrderLevel
    status: WorkOrderStage
    work_order_status: WorkOrderStatus
    assignee: Optional[str] = None
    assignee_user_id: Optional[int] = None
    description: str
    ai_suggestion: str
    scene_images: list[str]
    scene_info: str
    process_message: Optional[str] = None
    process_images: Optional[list[str]] = None
    completed_at: Optional[str] = None
    required_category: str = "traffic_police"


class WorkOrderDispatchRequest(BaseModel):
    user_id: int


class WorkOrderStatusUpdateRequest(BaseModel):
    status: WorkOrderStage
    process_message: Optional[str] = None
    process_image_url: Optional[str] = None
