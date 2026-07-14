"""Inference module: frame -> YOLO inference -> ByteTrack tracking -> speed/traffic calculation"""

import logging
import time
from collections import Counter

import cv2
import numpy as np
import supervision as sv

from app.modules.camera_data import BoundingBoxItem, CameraDataStore

# torchvision NMS 提供向量化 C++ 实现，性能优于手写 Python 循环。
# 在缺少 torchvision 的环境中优雅回退到 Python 版本，保证可用性。
try:
    import torch as _torch
    from torchvision.ops import nms as _tv_nms
    _TORCHVISION_NMS_AVAILABLE = True
except ImportError:  # pragma: no cover - 仅在精简环境中触发
    _torch = None
    _tv_nms = None
    _TORCHVISION_NMS_AVAILABLE = False

logger = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.3       # 提高阈值减少检测框数量（原 0.1），大幅降低 ByteTrack 处理量
NMS_OVERLAP = 0.5
LOST_BUFFER = 5            # 降低 ByteTrack 丢失轨迹管理开销（原 15）
TRAIL_MAX_AGE = 30
PIXEL_TO_METER = 0.05

COLOR_PALETTE = sv.ColorPalette.DEFAULT

# cv2 直接绘制用 BGR 颜色，避免 supervision Color 转换开销
_COLORS_BGR = [
    (46, 168, 224), (164, 73, 163), (158, 216, 102), (40, 184, 240),
    (92, 184, 92), (60, 76, 231), (180, 105, 255), (0, 215, 255),
]


class FrameProcessor:
    def __init__(self, cam_id: str, class_names: dict):
        self.cam_id = cam_id
        self.class_names = class_names
        self.trails: dict[int, list] = {}
        self.trail_age: dict[int, int] = {}
        self.frame_count = 0
        self.frame_h = 480
        self.frame_w = 640
        self.zx1 = self.zy1 = self.zx2 = self.zy2 = 0
        self._inside_zone: dict[int, bool] = {}
        self._entry_events: list = []
        self._exit_events: list = []
        self._prev_positions: dict[int, dict] = {}
        self._trajectories: dict[int, list] = {}
        self._speed_last_update: dict[int, float] = {}
        self._speed_stable: dict[int, float] = {}
        self._track_colors: dict[int, sv.Color] = {}
        self._lane_history: list[int] = []
        # 纯 IoU 跟踪（完全去掉 ByteTrack，消除卡尔曼滤波 GIL 开销）
        self._last_track_boxes: dict[int, list[float]] = {}  # track_id → bbox
        self._next_track_id: int = 1

    def init_zone(self, h: int, w: int, margin: float = 0.15):
        self.frame_h, self.frame_w = h, w
        self.zx1 = int(w * margin)
        self.zy1 = int(h * margin)
        self.zx2 = w - self.zx1
        self.zy2 = h - self.zy1

    def _get_color(self, track_id: int) -> sv.Color:
        if track_id not in self._track_colors:
            self._track_colors[track_id] = COLOR_PALETTE.by_idx(track_id % len(COLOR_PALETTE))
        return self._track_colors[track_id]

    def _perspective_scale(self, cy: float) -> float:
        """Perspective scaling based on y-coordinate:放大 (small cy), shrink (large cy)"""
        ratio = cy / max(self.frame_h, 1)
        return 0.6 + ratio * 0.8

    def process(self, model, frame: np.ndarray, device, results) -> tuple[list[dict], np.ndarray, dict]:
        self.frame_count += 1
        now = time.time()
        _t = [time.perf_counter()]

        # 手动构造 Detections（替代 sv.Detections.from_ultralytics，减少 ~2.5ms GIL）
        if hasattr(results, 'boxes') and results.boxes is not None and len(results.boxes) > 0:
            detections = sv.Detections(
                xyxy=results.boxes.xyxy.cpu().numpy(),
                confidence=results.boxes.conf.cpu().numpy(),
                class_id=results.boxes.cls.cpu().numpy().astype(int),
            )
        else:
            detections = sv.Detections.empty()

        if len(detections) > 0 and detections.confidence is not None:
            detections = detections[detections.confidence >= MIN_CONFIDENCE]
        if len(detections) > 1 and detections.xyxy is not None:
            detections = self._nms(detections)
        _t.append(time.perf_counter())

        # 纯 IoU 跟踪（完全去掉 ByteTrack，消除卡尔曼滤波 GIL 开销 ~5ms）
        detections = self._simple_iou_match(detections)
        _t.append(time.perf_counter())

        # 直接用 detections 作为 filtered（去掉 actual_tids 过滤，减少 ~1ms Python 循环）
        filtered = detections

        detection_list: list[dict] = []
        speed_map: dict[int, float] = {}

        if len(filtered) > 0:
            for i in range(len(filtered)):
                tid = int(filtered.tracker_id[i])
                cid = int(filtered.class_id[i]) if filtered.class_id is not None else -1
                cls_name = self.class_names.get(cid, "?")
                conf = float(filtered.confidence[i]) if filtered.confidence is not None else 0.0
                xyxy = filtered.xyxy[i].tolist() if filtered.xyxy is not None else [0, 0, 0, 0]
                detection_list.append({"track_id": tid, "class_name": cls_name, "confidence": conf, "bbox": xyxy})

                if tid == -1:
                    speed_map[tid] = 0.0
                    continue

                cx = (xyxy[0] + xyxy[2]) / 2
                cy = (xyxy[1] + xyxy[3]) / 2

                self.trails.setdefault(tid, []).append((cx, cy))
                if len(self.trails[tid]) > 15:
                    self.trails[tid].pop(0)
                self.trail_age[tid] = self.frame_count

                vel_kmh = self._calc_speed(tid, cx, cy, now)
                speed_map[tid] = vel_kmh

                was_in = self._inside_zone.get(tid, False)
                is_in = self.zx1 < cx < self.zx2 and self.zy1 < cy < self.zy2
                if not was_in and is_in:
                    self._entry_events.append((now, tid))
                elif was_in and not is_in:
                    self._exit_events.append((now, tid))
                self._inside_zone[tid] = is_in
        _t.append(time.perf_counter())

        self._update_store(detection_list, speed_map)
        stats = self._compute_stats(detection_list, speed_map)
        _t.append(time.perf_counter())

        # cv2 直接绘制（释放 GIL）
        if len(filtered) > 0:
            for i in range(len(filtered)):
                tid = int(filtered.tracker_id[i])
                spd = speed_map.get(tid, 0.0)
                cid = int(filtered.class_id[i]) if filtered.class_id is not None else -1
                nm = self.class_names.get(cid, "?")
                xyxy = filtered.xyxy[i].tolist() if filtered.xyxy is not None else [0, 0, 0, 0]
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                color = _COLORS_BGR[tid % len(_COLORS_BGR)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{nm} {spd:.0f}km/h"
                cv2.putText(frame, label, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        self._prune_events(now)
        cv2.rectangle(frame, (self.zx1, self.zy1), (self.zx2, self.zy2), (0, 255, 255), 2)
        cv2.putText(frame, "COUNT ZONE", (self.zx1 + 5, self.zy1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        flow = self.get_traffic_flow()
        for j, txt in enumerate([f"Entry:{flow['entry_count']}", f"Exit:{flow['exit_count']}", f"Flow:{flow['flow_per_min']}/min"]):
            cv2.putText(frame, txt, (self.zx1 + 5, self.zy2 - 10 - j * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        _t.append(time.perf_counter())

        total_ms = (_t[-1] - _t[0]) * 1000
        if total_ms > 20:
            logger.warning(
                "process cam=%s: parse=%.1f track=%.1f speed=%.1f store=%.1f draw=%.1f total=%.1fms",
                self.cam_id,
                (_t[1] - _t[0]) * 1000, (_t[2] - _t[1]) * 1000, (_t[3] - _t[2]) * 1000,
                (_t[4] - _t[3]) * 1000, (_t[5] - _t[4]) * 1000, total_ms,
            )

        self._cleanup_trails()
        return detection_list, frame, stats

    def _nms(self, detections: sv.Detections) -> sv.Detections:
        # 优先使用 torchvision.ops.nms：C++ 向量化实现，对密集检测帧提速 5-10x。
        # torchvision 不可用时回退到原 Python 版本，保证功能可用。
        if (
            _TORCHVISION_NMS_AVAILABLE
            and detections.xyxy is not None
            and len(detections) > 0
        ):
            boxes = _torch.from_numpy(np.asarray(detections.xyxy, dtype=np.float32))
            if detections.confidence is not None:
                scores = _torch.from_numpy(np.asarray(detections.confidence, dtype=np.float32))
            else:
                scores = _torch.ones(len(detections), dtype=_torch.float32)
            keep = _tv_nms(boxes, scores, NMS_OVERLAP)
            return detections[keep.cpu().numpy()]
        return self._nms_python(detections)

    def _nms_python(self, detections: sv.Detections) -> sv.Detections:
        boxes = np.array(detections.xyxy)
        scores = np.array(detections.confidence) if detections.confidence is not None else np.ones(len(detections))
        order = scores.argsort()[::-1]
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            xi1, yi1, xi2, yi2 = boxes[i]
            rest = boxes[order[1:]]
            ix1 = np.maximum(xi1, rest[:, 0])
            iy1 = np.maximum(yi1, rest[:, 1])
            ix2 = np.minimum(xi2, rest[:, 2])
            iy2 = np.minimum(yi2, rest[:, 3])
            inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
            area_i = (xi2 - xi1) * (yi2 - yi1)
            area_j = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
            iou = inter / (area_i + area_j - inter + 1e-6)
            remaining = np.where(iou <= NMS_OVERLAP)[0]
            order = order[remaining + 1]
        return detections[keep]

    def _simple_iou_match(self, detections: sv.Detections) -> sv.Detections:
        """纯 IoU 跟踪：将当前帧检测框与上次轨迹做 IoU 匹配。

        向量化匹配（numpy argmax + 降序贪心），避免 Python 双重 for 循环。
        清理旧轨迹（超过 50 个时删除最旧），防止 M 持续增长。
        """
        if len(detections) == 0:
            return detections

        # 清理旧轨迹（超过 50 个时只保留最近 50 个）
        if len(self._last_track_boxes) > 50:
            sorted_ids = sorted(self._last_track_boxes.keys())
            for tid in sorted_ids[:-50]:
                del self._last_track_boxes[tid]

        if len(self._last_track_boxes) == 0:
            N = len(detections)
            tracker_ids = np.arange(self._next_track_id, self._next_track_id + N, dtype=int)
            for i in range(N):
                self._last_track_boxes[int(tracker_ids[i])] = detections.xyxy[i].tolist()
            self._next_track_id += N
            detections.tracker_id = tracker_ids
            return detections

        det_boxes = np.asarray(detections.xyxy, dtype=np.float32)
        track_ids = np.array(list(self._last_track_boxes.keys()), dtype=int)
        track_boxes = np.array([self._last_track_boxes[tid] for tid in track_ids], dtype=np.float32)

        # 向量化 IoU 矩阵 (N, M) — 全 numpy，释放 GIL
        x1 = np.maximum(det_boxes[:, 0:1], track_boxes[:, 0:1].T)
        y1 = np.maximum(det_boxes[:, 1:2], track_boxes[:, 1:2].T)
        x2 = np.minimum(det_boxes[:, 2:3], track_boxes[:, 2:3].T)
        y2 = np.minimum(det_boxes[:, 3:4], track_boxes[:, 3:4].T)
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_det = (det_boxes[:, 2] - det_boxes[:, 0]) * (det_boxes[:, 3] - det_boxes[:, 1])
        area_track = (track_boxes[:, 2] - track_boxes[:, 0]) * (track_boxes[:, 3] - track_boxes[:, 1])
        iou = inter / (area_det[:, None] + area_track[None, :] - inter + 1e-6)

        # 向量化匹配：argmax + 降序贪心（仅 N 次 Python 循环，非 N×M）
        N = len(det_boxes)
        best_j = np.argmax(iou, axis=1)  # (N,) 每个检测框的最佳轨迹索引
        best_iou = iou[np.arange(N), best_j]  # (N,) 最佳 IoU 值
        order = np.argsort(-best_iou)  # 按 IoU 降序排列

        tracker_ids = np.full(N, -1, dtype=int)
        used = set()
        for idx in order:
            j = int(best_j[idx])
            if best_iou[idx] > 0.3 and j not in used:
                tracker_ids[idx] = track_ids[j]
                used.add(j)
                self._last_track_boxes[int(track_ids[j])] = det_boxes[idx].tolist()

        # 未匹配的分配新 ID
        for i in range(N):
            if tracker_ids[i] == -1:
                tracker_ids[i] = self._next_track_id
                self._next_track_id += 1
                self._last_track_boxes[int(tracker_ids[i])] = det_boxes[i].tolist()

        detections.tracker_id = tracker_ids
        return detections

    def _calc_speed(self, track_id: int, cx: float, cy: float, now: float) -> float:
        """简化速度计算：用最近两帧位移代替 polyfit，减少 GIL 持有时间。"""
        traj = self._trajectories.setdefault(track_id, [])
        traj.append((cx, cy, now))
        cutoff = now - 2.0
        self._trajectories[track_id] = [(x, y, t) for x, y, t in traj if t > cutoff]
        pts = self._trajectories[track_id]

        vel_kmh = self._speed_stable.get(track_id, 0.0)
        lu = self._speed_last_update.get(track_id, 0.0)
        if now - lu < 0.6:
            return vel_kmh

        if len(pts) < 2:
            self._prev_positions[track_id] = {"cx": cx, "cy": cy, "_vel": vel_kmh, "_vel_mps": vel_kmh / 3.6}
            return vel_kmh

        # 简单位移/时间（替代 polyfit，减少 ~1ms GIL）
        first = pts[0]
        last = pts[-1]
        dx = last[0] - first[0]
        dy = last[1] - first[1]
        dt = last[2] - first[2]
        if dt < 0.01:
            dt = 0.01
        dist_px = (dx * dx + dy * dy) ** 0.5
        speed_px = dist_px / dt
        avg_cy = (first[1] + last[1]) / 2
        scale = self._perspective_scale(avg_cy)
        new_kmh = speed_px * PIXEL_TO_METER * scale * 3.6

        if new_kmh < 0.5:
            new_kmh = 0.0
        if vel_kmh > 0 and abs(new_kmh - vel_kmh) < 8:
            new_kmh = vel_kmh * 0.7 + new_kmh * 0.3

        self._speed_stable[track_id] = new_kmh
        self._speed_last_update[track_id] = now
        self._prev_positions[track_id] = {"cx": cx, "cy": cy, "_vel": new_kmh, "_vel_mps": new_kmh / 3.6}
        return new_kmh

    def _update_store(self, detection_list: list[dict], speed_map: dict[int, float]):
        cx_list = [((d["bbox"][0] + d["bbox"][2]) / 2) for d in detection_list]
        # 优先读取 LaneSegmentationWorker 缓存的车道数（精确分割模型结果）；
        # 为 None 时回退到滑动窗口平滑启发式（方案 A），消除每帧抖动。
        cached_lane_count = CameraDataStore().get_lane_count(self.cam_id)
        lane_count = cached_lane_count if cached_lane_count is not None else self._estimate_lanes_stable(cx_list)
        speeds = [speed_map.get(d["track_id"], 0.0) for d in detection_list]
        avg_s = float(np.mean(speeds)) if speeds else 0.0
        max_s = float(np.max(speeds)) if speeds else 0.0
        cc = sum(1 for d in detection_list if d["class_name"].lower() == "car")
        tc = sum(1 for d in detection_list if d["class_name"].lower() in ("truck", "trailer"))
        bc = sum(1 for d in detection_list if d["class_name"].lower() == "bus")
        mc = sum(1 for d in detection_list if d["class_name"].lower() in ("motorcycle", "bike", "bicycle"))
        CameraDataStore().update(
            camera_id=self.cam_id,
            total_vehicle_count=len(detection_list),
            boxes=[BoundingBoxItem(track_id=int(d["track_id"]), class_name=str(d["class_name"]),
                                   confidence=float(d["confidence"]), bbox=list(d["bbox"])) for d in detection_list],
            lane_count=lane_count, avg_speed=avg_s, max_speed=max_s,
            car_count=cc, truck_count=tc, bus_count=bc, moto_count=mc,
        )

    def _estimate_lanes(self, cx_list: list[float]) -> int:
        if len(cx_list) < 3:
            return max(1, len(cx_list))
        cx = np.array(cx_list)
        cx.sort()
        gaps = np.diff(cx)
        mg = float(np.mean(gaps))
        if mg > 3:
            clusters = 1
            for g in gaps:
                if g > mg * 0.5:
                    clusters += 1
            return max(1, min(clusters, 8))
        return max(1, len(cx_list))

    def _estimate_lanes_stable(self, cx_list: list[float]) -> int:
        """滑动窗口平滑（30帧≈2s）取众数，消除每帧车道数抖动。"""
        raw = self._estimate_lanes(cx_list)
        self._lane_history.append(raw)
        if len(self._lane_history) > 30:
            self._lane_history.pop(0)
        return Counter(self._lane_history).most_common(1)[0][0]

    def _compute_stats(self, dlist: list[dict], smap: dict[int, float]) -> dict:
        speeds = [smap.get(d["track_id"], 0.0) for d in dlist]
        return {"avg_speed": float(np.mean(speeds)) if speeds else 0.0,
                "max_speed": float(np.max(speeds)) if speeds else 0.0}

    def _prune_events(self, now: float):
        cutoff = now - 60
        self._entry_events = [(t, tid) for t, tid in self._entry_events if t > cutoff]
        self._exit_events = [(t, tid) for t, tid in self._exit_events if t > cutoff]

    def get_traffic_flow(self) -> dict:
        now = time.time()
        self._prune_events(now)
        ec = len(self._entry_events)
        xc = len(self._exit_events)
        total = ec + xc
        return {"entry_count": ec, "exit_count": xc, "flow_per_min": round(total, 1)}

    def _cleanup_trails(self):
        stale = [tid for tid, age in self.trail_age.items() if self.frame_count - age > TRAIL_MAX_AGE]
        for tid in stale:
            self.trails.pop(tid, None)
            self.trail_age.pop(tid, None)
