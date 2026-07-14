"""冷启动模块：模型加载与预热"""

import logging
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from app.config import settings

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "best.pt"


def load_model(
    model_path: str | None = None,
    min_confidence: float = 0.1,
    image_size: int | None = None,
):
    """加载 YOLO 模型并执行预热推理。

    Returns:
        (model, device, precision_kwargs)
    """
    model_path = model_path or settings.YOLO_MODEL_PATH or str(DEFAULT_MODEL_PATH)
    image_size = image_size or settings.YOLO_IMAGE_SIZE
    model = YOLO(model_path)

    try:
        import torch

        if torch.cuda.is_available():
            device = 0
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True
            logger.info("YOLO on GPU: %s", torch.cuda.get_device_name(0))
        else:
            device = "cpu"
            logger.warning("YOLO on CPU (CUDA not available)")
    except Exception:
        device = "cpu"
        logger.warning("YOLO on CPU (torch import failed)")

    model.to(device)

    use_half = device != "cpu" and settings.YOLO_HALF
    try:
        from ultralytics.cfg import get_cfg

        supports_quantize = hasattr(get_cfg(), "quantize")
    except Exception:
        supports_quantize = False

    precision_kwargs = (
        {"quantize": 16}
        if use_half and supports_quantize
        else {"half": True}
        if use_half
        else {}
    )

    for _ in range(2):
        dummy = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        model(dummy, imgsz=image_size, verbose=False, conf=min_confidence, **precision_kwargs)

    logger.info("YOLO warm-up complete (device=%s, half=%s)", device, bool(precision_kwargs))
    return model, device, precision_kwargs
