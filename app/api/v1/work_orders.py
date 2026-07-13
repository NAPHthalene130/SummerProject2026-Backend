from fastapi import APIRouter, HTTPException

from app.models.user import StaffResponse
from app.models.work_order import (
    WorkOrderDispatchRequest,
    WorkOrderItemResponse,
    WorkOrderStatusUpdateRequest,
    MobileFeedbackRequest,
    FeedbackReviewRequest,
)
from app.repository.work_order_repository import WorkOrderRepository


work_orders_router = APIRouter()
staff_router = APIRouter()


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


