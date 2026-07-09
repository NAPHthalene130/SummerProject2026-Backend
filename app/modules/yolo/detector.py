"""
YOLO 检测 + 跟踪模块
集成到主系统后，外部通过 YOLODetector.detect(frame) 调用。
"""

import json
import math
import threading
import time
import warnings

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

warnings.filterwarnings("ignore", message=".*ByteTrack.*deprecated.*")

TRAFFIC_CLASSES = ["car", "truck", "bus", "pedestrian", "bicycle", "motorcycle"]
COLORS = {
    "car": (56, 56, 255),
    "truck": (255, 144, 30),
    "bus": (255, 224, 32),
    "pedestrian": (80, 200, 120),
    "bicycle": (255, 105, 180),
    "motorcycle": (128, 0, 128),
}
COLOR_LIST = list(COLORS.values())
COLOR_NAMES = list(COLORS.keys())

# 检测参数
INFERENCE_SIZE = 320
DETECTION_FPS = 30
TRAIL_LEN = DETECTION_FPS  # 轨迹 = 1 秒
LOST_BUFFER = 1


class YOLODetector:
    """
    ═══════════════════════════════════════════════════════
    视频流接收接口:
      外部将视频帧传入 detect(frame, cam_id) 即可。
      每路摄像头独立追踪，cam_id 用于区分不同摄像头。
    ═══════════════════════════════════════════════════════
    处理结果传输接口:
      - detect() 返回 annotated frame (绘制了框/轨迹/速度的帧)
      - get_detections(cam_id) → dict: 最近一帧的检测结果 JSON
      - get_all_detections() → dict: 所有摄像头的检测结果
    ═══════════════════════════════════════════════════════
    """

    def __init__(self, model_path: str = "best.pt"):
        self.model_path = model_path
        self._fps = 0.0
        self._frame_count = 0
        self._last_time = time.perf_counter()

        # YOLO 模型
        self.model = YOLO(model_path)
        self.model.to("cuda")
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
        h, w = frame.shape[:2]
        if w > INFERENCE_SIZE:
            scale_r = INFERENCE_SIZE / w
            small = cv2.resize(frame, (INFERENCE_SIZE, int(h * scale_r)))
        else:
            scale_r = 1
            small = frame.copy()

        # YOLO 推理
        with self.model_lock:
            results = self.model(small, verbose=False)[0]

        det = sv.Detections.from_ultralytics(results)

        # ByteTrack 跟踪
        if len(det) > 0 and det.confidence is not None:
            det = self.trackers[cam_id].update_with_detections(det)
        else:
            det = sv.Detections.empty()

        # 标注和结果收集
        boxes_data = []
        display_scale = w / INFERENCE_SIZE if w > INFERENCE_SIZE else 1

        for i in range(len(det)):
            x1, y1, x2, y2 = det.xyxy[i].tolist()
            cx = (x1 + x2) / 2 * display_scale
            cy = (y1 + y2) / 2 * display_scale
            tid = int(det.tracker_id[i])

            # 轨迹
            trail = self.trails[cam_id].setdefault(tid, [])
            trail.append((round(cx, 1), round(cy, 1)))
            if len(trail) > TRAIL_LEN:
                trail.pop(0)

            # 稳定化速度 = 轨迹总路径 / 时间
            path_len = 0
            for j in range(1, len(trail)):
                dx = trail[j][0] - trail[j - 1][0]
                dy = trail[j][1] - trail[j - 1][1]
                path_len += math.sqrt(dx * dx + dy * dy)
            time_span = len(trail) / DETECTION_FPS
            speed = round(path_len / time_span, 1) if time_span > 0 else 0

            # 每秒更新一次显示速度
            self.frame_counts[cam_id] += 1
            if self.frame_counts[cam_id] >= DETECTION_FPS:
                self.frame_counts[cam_id] = 0
            if self.frame_counts[cam_id] == 0:
                self.speed_displays[cam_id][tid] = speed
            display_speed = self.speed_displays[cam_id].get(tid, speed)

            # 缩放坐标到原始帧空间用于绘制
            x1_d = x1 * display_scale
            y1_d = y1 * display_scale
            x2_d = x2 * display_scale
            y2_d = y2 * display_scale

            cls_id = int(det.class_id[i])
            cls_name = self.class_names.get(cls_id, "unknown")

            boxes_data.append({
                "x1": round(x1_d, 1),
                "y1": round(y1_d, 1),
                "x2": round(x2_d, 1),
                "y2": round(y2_d, 1),
                "label": cls_name,
                "confidence": round(float(det.confidence[i]), 2),
                "tracker_id": tid,
                "speed": display_speed,
            })

            # 在帧上绘制
            color = COLORS.get(cls_name, (0, 255, 0))
            cv2.rectangle(frame, (int(x1_d), int(y1_d)), (int(x2_d), int(y2_d)), color, 2)

            # 轨迹线
            if len(trail) >= 2:
                pts = np.array([(int(p[0]), int(p[1])) for p in trail], np.int32)
                cv2.polylines(frame, [pts], False, color, 2)

            # 标签
            label_text = f"{cls_name} {display_speed}px/s"
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (int(x1_d), int(y1_d) - th - baseline - 4),
                          (int(x1_d) + tw, int(y1_d)), color, -1)
            cv2.putText(frame, label_text, (int(x1_d), int(y1_d) - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 保存检测结果供外部获取
        self._latest_detections[cam_id] = boxes_data

        # FPS 统计
        self._frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._last_time
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_time = now
        cv2.putText(frame, f"FPS: {self._fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame

    # ── 处理结果传输接口 ────────────────────────────────

    def get_detections(self, cam_id: str = "default") -> dict:
        """
        处理结果传输接口:
          获取指定摄像头最新一帧的检测结果 JSON。
          返回: {"type": "detection", "cam": cam_id, "boxes": [...]}
        """
        boxes = self._latest_detections.get(cam_id, [])
        return {
            "type": "detection",
            "cam": cam_id,
            "boxes": boxes,
        }

    def get_all_detections(self) -> dict:
        """
        处理结果传输接口:
          获取所有摄像头的最新检测结果。
        """
        return {
            cam_id: self.get_detections(cam_id)
            for cam_id in self._latest_detections
        }
