from fastapi import APIRouter

from app.api.v1.agent import agent_router
from app.api.v1.cameras import cameras_router
from app.api.v1.live import live_router
from app.api.v1.risks import router as risks_router
from app.api.v1.users import users_router
from app.api.v1.work_orders import staff_router, work_orders_router
from app.api.v1.mobile import router as mobile_router
from app.api.v1.uploads import uploads_router
from app.api.v1.roads import router as roads_router

api_router = APIRouter()

api_router.include_router(agent_router, prefix="/agent", tags=["agent"])
api_router.include_router(cameras_router, prefix="/cameras", tags=["cameras"])
api_router.include_router(live_router, prefix="/live", tags=["live"])
api_router.include_router(risks_router, prefix="/risks", tags=["risks"])
api_router.include_router(work_orders_router, prefix="/work-orders", tags=["work-orders"])
api_router.include_router(staff_router, prefix="/staff", tags=["staff"])
api_router.include_router(users_router, prefix="/users", tags=["users"])
api_router.include_router(mobile_router, tags=["mobile"])
api_router.include_router(uploads_router, prefix="/uploads", tags=["uploads"])
api_router.include_router(roads_router, prefix="/roads", tags=["roads"])


@api_router.get("/ping")
async def ping():
    return {"message": "pong"}
