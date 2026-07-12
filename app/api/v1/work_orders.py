from fastapi import APIRouter, HTTPException

from app.models.user import StaffResponse
from app.models.work_order import (
    WorkOrderDispatchRequest,
    WorkOrderItemResponse,
    WorkOrderStatusUpdateRequest,
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


@staff_router.get("/", response_model=list[StaffResponse])
async def list_staff():
    return WorkOrderRepository.list_staff()


@staff_router.delete("/{user_id}")
async def delete_staff(user_id: int):
    if not WorkOrderRepository.delete_staff(user_id):
        raise HTTPException(status_code=404, detail="人员不存在")
    return {"deleted": True, "user_id": user_id}
