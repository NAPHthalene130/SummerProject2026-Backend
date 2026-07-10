from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel


class WorkOrder(BaseModel):
    work_order_id: int
    work_order_type: str
    work_order_describe: str
    work_order_img_url: str
    work_order_rank: int
    work_order_time: datetime
    work_order_is_solve: bool


WorkOrderStatus = Literal["unassigned", "pending", "processing", "completed", "false_alarm"]
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
    status: WorkOrderStatus
    assignee: Optional[str] = None
    description: str
    ai_suggestion: str
    scene_images: list[str]
    scene_info: str
    process_message: Optional[str] = None
    process_images: Optional[list[str]] = None
    completed_at: Optional[str] = None


class WorkOrderDispatchRequest(BaseModel):
    user_id: int


class WorkOrderStatusUpdateRequest(BaseModel):
    status: WorkOrderStatus
    process_message: Optional[str] = None
    process_image_url: Optional[str] = None
