from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.models.user import StaffResponse
from app.models.work_order import (
    WorkOrderDispatchRequest,
    WorkOrderItemResponse,
    WorkOrderStatusUpdateRequest,
    MobileFeedbackRequest,
    FeedbackReviewRequest,
)
from app.repository.work_order_repository import WorkOrderRepository


class CameraUnprocessedEvents(BaseModel):
    camera_id: str
    camera_name: str
    events: list[WorkOrderItemResponse]


class UnprocessedResponse(BaseModel):
    cameras: list[CameraUnprocessedEvents]


work_orders_router = APIRouter()
staff_router = APIRouter()


@work_orders_router.get("/unprocessed", response_model=UnprocessedResponse)
async def list_unprocessed():
    all_orders = WorkOrderRepository.list_work_orders()
    unprocessed = [
        wo for wo in all_orders
        if wo.status not in ("completed", "ignored")
    ]
    by_camera: dict[str, list[WorkOrderItemResponse]] = {}
    camera_names: dict[str, str] = {}
    for wo in unprocessed:
        cid = wo.camera_id or "unknown"
        if cid not in by_camera:
            by_camera[cid] = []
            camera_names[cid] = wo.camera_name or "未知摄像头"
        by_camera[cid].append(wo)

    cameras = sorted(
        [
            CameraUnprocessedEvents(
                camera_id=cid,
                camera_name=camera_names[cid],
                events=sorted(evts, key=lambda e: e.event_time or "", reverse=True),
            )
            for cid, evts in by_camera.items()
        ],
        key=lambda c: c.camera_id,
    )
    return UnprocessedResponse(cameras=cameras)


@work_orders_router.get("/detail/{work_order_id}", response_model=WorkOrderItemResponse)
async def get_work_order_detail(work_order_id: str):
    wo = WorkOrderRepository.get_work_order(work_order_id)
    if wo is None:
        raise HTTPException(status_code=404, detail=f"Work order '{work_order_id}' not found")
    return wo


@work_orders_router.get("/", response_model=list[WorkOrderItemResponse])
async def list_work_orders(user_id: int | None = None):
    return WorkOrderRepository.list_work_orders(user_id)


@work_orders_router.put("/{work_order_id}/dispatch", response_model=WorkOrderItemResponse)
async def dispatch_work_order(work_order_id: str, request: WorkOrderDispatchRequest):
    try:
        order = WorkOrderRepository.dispatch_work_order(work_order_id, request.user_id)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    if order is None:
        raise HTTPException(status_code=404, detail=f"Work order '{work_order_id}' not found")
    return order


@work_orders_router.patch("/{work_order_id}/status", response_model=WorkOrderItemResponse)
async def update_work_order_status(work_order_id: str, request: WorkOrderStatusUpdateRequest):
    order = WorkOrderRepository.update_work_order_status(
        work_order_id=work_order_id,
        status=request.status,
        process_message=request.process_message,
        process_image_url=request.process_image_url,
    )
    if order is None:
        raise HTTPException(status_code=404, detail=f"Work order '{work_order_id}' not found")
    return order


@work_orders_router.post("/{work_order_id}/mobile-feedback", response_model=WorkOrderItemResponse)
async def submit_mobile_feedback(work_order_id: str, request: MobileFeedbackRequest):
    try:
        order = WorkOrderRepository.submit_mobile_feedback(
            work_order_id, request.user_id, request.status, request.process_message, request.process_image_url,
        )
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    if order is None:
        raise HTTPException(status_code=404, detail="工单或用户不存在")
    return order


@work_orders_router.post("/{work_order_id}/feedback-review", response_model=WorkOrderItemResponse)
async def review_mobile_feedback(work_order_id: str, request: FeedbackReviewRequest):
    order = WorkOrderRepository.review_mobile_feedback(work_order_id, request.decision, request.review_message)
    if order is None:
        raise HTTPException(status_code=404, detail="没有待审核的处置结果")
    return order


@staff_router.get("/", response_model=list[StaffResponse])
async def list_staff():
    return WorkOrderRepository.list_staff()


