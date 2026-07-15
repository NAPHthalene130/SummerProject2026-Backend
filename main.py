import logging
import logging.handlers
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# 抑制 ffmpeg/libav 的 H.264 解码告警刷屏（控制台红色 [h264 @ ...] co located POCs
# unavailable / mmco: unref short failure / Missing reference picture ...）。这些由 RTSP
# 解码器经 av_log 直接写 C 层 stderr(fd 2)，PowerShell 以红色显示。OPENCV_FFMPEG_LOGLEVEL
# 在部分 opencv 构建中对 RTSP 解码路径不生效，故在 fd 层将 stderr 重定向到 devnull；
# 同时把 Python 的 sys.stderr 重新绑定到原始 fd，保证 uvicorn 日志与异常栈仍在控制台显示。
_real_stderr_fd = os.dup(2)
_devnull_fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull_fd, 2)
os.close(_devnull_fd)
sys.stderr = os.fdopen(_real_stderr_fd, "w", encoding="utf-8", buffering=1)
sys.__stderr__ = sys.stderr

# RTSP 选项：仅用 TCP 传输（避免 UDP 丢包），并限制每路 FFmpeg 解码器为
# 单线程。30 路各自并行时再由操作系统调度到 8 个 CPU worker，避免每路
# 都自动创建一整组解码线程造成数百线程争抢。
# 注意：flags;low_delay / analyzeduration;0 会禁用 B 帧重排序，导致 mp4Tortsp 的含 B 帧码流
# 持续报 "reference picture missing during reorder" / "co located POCs unavailable" 并刷屏，故不启用。
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|threads;1"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

import av
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.api.v1.live import close_all_peer_connections
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

    # Lane segmentation worker (optional): 5s serial inference for lane count.
    # Only starts when enabled and model_path points to a valid file.
    lane_seg_worker = None
    if settings.LANE_SEG_ENABLED and settings.LANE_SEG_MODEL_PATH:
        try:
            from app.modules.yolo.lane_segmentation import LaneSegmentationWorker
            lane_seg_worker = LaneSegmentationWorker(
                model_path=settings.LANE_SEG_MODEL_PATH,
                interval=settings.LANE_SEG_INTERVAL,
                image_size=settings.LANE_SEG_IMAGE_SIZE,
            )
            lane_seg_worker.start()
        except Exception as e:
            logger.warning("LaneSegmentationWorker init failed: %s", e)
            lane_seg_worker = None
    else:
        logger.info("LaneSegmentationWorker disabled (lane_segmentation.enabled=false or model_path empty)")

    yield

    if lane_seg_worker is not None:
        lane_seg_worker.stop()
    if traffic_analyst_enabled:
        await TrafficAnalyst().stop()
    await close_all_peer_connections()
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
        reload=settings.RELOAD,
    )
