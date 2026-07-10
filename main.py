import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

import av
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.config import settings
from app.utils.camera_manager import CameraManager
from app.modules.agent.traffic_analyst import TrafficAnalyst
from app.modules.stream import StreamManager

os.environ["AV_LOG_FORCE_COLOR"] = "0"
av.logging.set_level(av.logging.FATAL)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("aioice.ice").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(application: FastAPI):
    cameras = CameraManager().get_all()
    logger.info("Loaded %d cameras from config", len(cameras))
    for cam in cameras:
        logger.info("  [%s] %s -> %s", cam.id, cam.name, cam.url)

    for cam in cameras:
        StreamManager().subscribe(cam.id)
        logger.info("  Subscribed camera %s for analysis", cam.id)

    await TrafficAnalyst().start()
    yield
    await TrafficAnalyst().stop()
    StreamManager().stop_all()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_orderimg_dir = Path(__file__).resolve().parent / "app" / "data" / "orderImg"
_orderimg_dir.mkdir(parents=True, exist_ok=True)
app.mount("/orderImg", StaticFiles(directory=str(_orderimg_dir)), name="orderImg")

app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
