import logging
import os
from contextlib import asynccontextmanager

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

import av
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.config import settings
from app.utils.camera_manager import CameraManager

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
    yield
    from app.modules.stream import StreamManager

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
