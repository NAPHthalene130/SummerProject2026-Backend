import random
import time

import cv2
import numpy as np

TRAFFIC_CLASSES = ["car", "truck", "bus", "pedestrian", "bicycle", "motorcycle"]
COLORS = {
    "car": (56, 56, 255),
    "truck": (255, 144, 30),
    "bus": (255, 224, 32),
    "pedestrian": (80, 200, 120),
    "bicycle": (255, 105, 180),
    "motorcycle": (128, 0, 128),
}


class YOLODetector:
<<<<<<< Updated upstream
    def __init__(self, model_path: str = ""):
=======
    """
    ═══════════════════════════════════════════════════════
    视频流接收接口:
      外部将视频帧传入 detect(frame, cam_id) 即可。v
      每路摄像头独立追踪，cam_id 用于区分不同摄像头。
    ═══════════════════════════════════════════════════════
    处理结果传输接口:
      - detect() 返回 annotated frame (绘制了框/轨迹/速度的帧)
      - get_detections(cam_id) → dict: 最近一帧的检测结果 JSON
      - get_all_detections() → dict: 所有摄像头的检测结果
    ═══════════════════════════════════════════════════════
    """

    def __init__(self, model_path: str = "best.pt"):
>>>>>>> Stashed changes
        self.model_path = model_path
        self._fps = 0.0
        self._frame_count = 0
        self._last_time = time.perf_counter()

<<<<<<< Updated upstream
    def detect(self, frame: np.ndarray) -> np.ndarray:
=======
        # YOLO 模型
        self.model = YOLO(model_path)
        try:
            import torch
            self.device = 0 if torch.cuda.is_available() else "cpu"
        except Exception:
            self.device = "cpu"
        self.model.to(self.device)
        self.class_names = self.model.names
        self.model_lock = threading.Lock()

        # 跟踪器 & 轨迹 (按摄像头独立)
        self.trackers = {}
        self.trails = {}
        self.speed_displays = {}
        self.frame_counts = {}

        # 存储最新检测结果供外部获取
        self._latest_detections = {}

    # ── 公共接口 ────────────────────────────────────────

    def detect(self, frame: np.ndarray, cam_id: str = "default") -> np.ndarray:
        """
        视频流接收接口:
          传入一帧 BGR 图像，返回绘制了检测框/轨迹/速度的帧。
          cam_id 区分不同摄像头，追踪器按 cam_id 独立维护。
        """
        # 确保该摄像头有独立的追踪器
        if cam_id not in self.trackers:
            self.trackers[cam_id] = sv.ByteTrack(
                lost_track_buffer=LOST_BUFFER
            )
            self.trails[cam_id] = {}
            self.speed_displays[cam_id] = {}
            self.frame_counts[cam_id] = 0

        # 缩放至推理尺寸
>>>>>>> Stashed changes
        h, w = frame.shape[:2]

        num_boxes = random.randint(0, 5)
        for _ in range(num_boxes):
            box_w = random.randint(int(w * 0.05), int(w * 0.25))
            box_h = random.randint(int(h * 0.05), int(h * 0.25))
            x1 = random.randint(0, w - box_w)
            y1 = random.randint(0, h - box_h)
            x2 = x1 + box_w
            y2 = y1 + box_h

            cls_name = random.choice(TRAFFIC_CLASSES)
            confidence = round(random.uniform(0.5, 0.99), 2)
            color = COLORS[cls_name]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {confidence:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        self._frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._last_time
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_time = now

        cv2.putText(
            frame,
            f"FPS: {self._fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        return frame
