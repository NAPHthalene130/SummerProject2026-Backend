import logging
import logging.handlers
import os
from contextlib import asynccontextmanager
from pathlib import Path

# RTSP 低延迟选项：TCP 传输（避免 UDP 丢包卡顿）+ 关闭分析/减小缓冲，降低端到端延迟
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0|analyzeduration;0|probesize;512K"
)
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
# 可选硬件解码：设置 SP2026_HW_DECODER=1 并安装支持 NVDEC 的 OpenCV/ffmpeg 时启用
os.environ.setdefault("SP2026_HW_DECODER", "0")

import av
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.config import llm_settings, settings
from app.utils.camera_manager import CameraManager
from app.modules.agent.traffic_analyst import TrafficAnalyst
from app.modules.stream import StreamManager

os.environ["AV_LOG_FORCE_COLOR"] = "0"
av.logging.set_level(av.logging.FATAL)

_log_dir = Path(__file__).resolve().parent / "logs"
_log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.handlers.RotatingFileHandler(
            _log_dir / "app.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)
logging.getLogger("aioice.ice").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(application: FastAPI):
    try:
        from app.api.v1.mobile import ensure_schema
        from app.database import mysql_connection
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                ensure_schema(cursor)
    except Exception as error:
        logger.warning("Mobile work-order schema migration skipped: %s", error)
    cameras = CameraManager().get_all()
    logger.info("Loaded %d cameras from config", len(cameras))
    for cam in cameras:
        logger.info("  [%s] %s -> %s", cam.id, cam.name, cam.url)
    if settings.ENABLE_STREAMING:
        for cam in cameras:
            StreamManager().subscribe(cam.id)
            logger.info("  Subscribed camera %s for analysis", cam.id)
    else:
        logger.info("Camera streaming disabled; running in API-only mode")

    model_path = os.path.join(os.path.dirname(__file__), "app", "modules", "lstm", "traffic_risk_lstm_weights.pth")
    scaler_path = os.path.join(os.path.dirname(__file__), "app", "modules", "lstm", "scaler_params.json")

    if os.path.exists(model_path):
        from app.modules.lstm.predictor import risk_predictor
        try:
            device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            road_types = {cam.id: 0 for cam in cameras}
            risk_predictor.initialize(
                model_path,
                scaler_path=scaler_path if os.path.exists(scaler_path) else None,
                device=device,
                camera_road_types=road_types,
            )
            logger.info("RiskPredictor loaded: %s on %s", model_path, device)
        except Exception as e:
            logger.warning("RiskPredictor init failed: %s", e)
    else:
        logger.info("RiskPredictor weights not found at %s, skipping init", model_path)

    traffic_analyst_enabled = (
        settings.ENABLE_TRAFFIC_ANALYST
        and llm_settings.api_key not in {"", "your_api_key_here"}
    )
    if traffic_analyst_enabled:
        await TrafficAnalyst().start()
    else:
        logger.info("TrafficAnalyst disabled by configuration or missing API key")

    yield

    if traffic_analyst_enabled:
        await TrafficAnalyst().stop()
    if settings.ENABLE_STREAMING:
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
    import sys
    if sys.platform == "win32":
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
