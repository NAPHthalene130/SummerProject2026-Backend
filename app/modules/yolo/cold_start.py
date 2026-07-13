"""冷启动模块：模型加载与预热"""

import logging

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def load_model(model_path: str = "app/modules/yolo/best.pt", min_confidence: float = 0.1):
    """加载YOLO模型并执行预热推理"""
    model = YOLO(model_path)
    try:
        import torch
        if torch.cuda.is_available():
            device = 0
            logger.info("YOLO on GPU: %s", torch.cuda.get_device_name(0))
        else:
            device = "cpu"
            logger.warning("YOLO on CPU")
    except Exception:
        device = "cpu"
    model.to(device)
    for _ in range(2):
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model(dummy, imgsz=640, verbose=False, conf=min_confidence)
    logger.info("YOLO warm-up complete (device=%s)", device)
    return model, device
