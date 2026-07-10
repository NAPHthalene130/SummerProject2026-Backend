from fastapi import APIRouter

from app.api.v1.cameras import cameras_router
from app.api.v1.live import live_router
from app.api.v1.work_orders import staff_router, work_orders_router

api_router = APIRouter()

api_router.include_router(cameras_router, prefix="/cameras", tags=["cameras"])
api_router.include_router(live_router, prefix="/live", tags=["live"])
api_router.include_router(work_orders_router, prefix="/work-orders", tags=["work-orders"])
api_router.include_router(staff_router, prefix="/staff", tags=["staff"])


@api_router.get("/ping")
async def ping():
    return {"message": "pong"}
