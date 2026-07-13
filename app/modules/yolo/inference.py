"""推理模块：接收帧 → YOLO推理 → ByteTrack追踪 → 速度/车流计算"""

import collections
import logging
import time
from typing import Optional

import numpy as np
import supervision as sv

from app.modules.camera_data import BoundingBoxItem, CameraDataStore
from app.modules.lstm.predictor import risk_predictor

logger = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.1
NMS_OVERLAP = 0.5
LOST_BUFFER = 30
TRAIL_MAX_AGE = 30
PIXEL_TO_METER = 0.05

TRACK_COLORS = [
    (56, 56, 255), (255, 144, 30), (255, 224, 32),
    (80, 200, 120), (255, 105, 180), (128, 0, 128),
    (0, 200, 255), (200, 0, 200), (0, 255, 128), (255, 0, 128),
]


class FrameProcessor:
    """单路摄像头帧处理器"""

    def __init__(self, cam_id: str, class_names: dict):
        self.cam_id = cam_id
        self.class_names = class_names
        self.tracker = sv.ByteTrack(lost_track_buffer=LOST_BUFFER)
        self.trails: dict[int, list] = {}
        self.trail_age: dict[int, int] = {}
        self.frame_count = 0
        self.zx1 = self.zy1 = self.zx2 = self.zy2 = 0
        self._inside_zone: dict[int, bool] = {}
        self._entry_events = collections.deque()
        self._exit_events = collections.deque()
        self._prev_positions: dict[int, dict] = {}
        self._prev_velocities: dict[int, float] = {}
        self._trajectories: dict[int, list] = {}
        self._speed_last_update: dict[int, float] = {}
        self._speed_stable: dict[int, float] = {}

    def init_zone(self, h: int, w: int, margin: float = 0.15):
        mx, my = int(w * margin), int(h * margin)
        self.zx1, self.zy1, self.zx2, self.zy2 = mx, my, w - mx, h - my

    def process(self, model, frame: np.ndarray, device, results) -> tuple[list[dict], np.ndarray, float]:
        """处理一帧，返回 (detection_list, annotated_frame, now)"""
        self.frame_count += 1
        now = time.time()
        now_ms = now * 1000

        detections = sv.Detections.from_ultralytics(results)

        if len(detections) > 0 and detections.confidence is not None:
            detections = detections[detections.confidence >= MIN_CONFIDENCE]

        if len(detections) > 1 and detections.xyxy is not None:
            detections = self._nms(detections)

        actual_tids = set()
        raw_boxes = [list(b) for b in (detections.xyxy.tolist() if detections.xyxy is not None else [])]

        if len(detections) > 0:
            detections = self.tracker.update_with_detections(detections)
        else:
            empty = sv.Detections.empty()
            empty.tracker_id = np.array([], dtype=int)
            detections = empty

        if len(detections) > 0 and detections.tracker_id is not None:
            for i in range(len(detections)):
                tid = int(detections.tracker_id[i])
                xyxy = detections.xyxy[i].tolist() if detections.xyxy is not None else None
                if xyxy is None:
                    continue
                for rb in raw_boxes:
                    if all(abs(xyxy[k] - rb[k]) < 10 for k in range(4)):
                        actual_tids.add(tid)
                        break

        detection_list: list[dict] = []
        zx1, zy1, zx2, zy2 = self.zx1, self.zy1, self.zx2, self.zy2
        inside = self._inside_zone
        speed_map: dict[int, float] = {}

        for i in range(len(detections)):
            if detections.tracker_id is None or i >= len(detections.tracker_id):
                continue
            track_id = int(detections.tracker_id[i])
            if track_id not in actual_tids:
                continue
            class_id = int(detections.class_id[i]) if detections.class_id is not None else -1
            cls_name = self.class_names.get(class_id, "?")
            conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
            xyxy = detections.xyxy[i].tolist() if detections.xyxy is not None else [0, 0, 0, 0]
            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2

            self.trails.setdefault(track_id, []).append((cx, cy))
            if len(self.trails[track_id]) > 30:
                self.trails[track_id].pop(0)
            self.trail_age[track_id] = self.frame_count

            was_in = inside.get(track_id, False)
            is_in = zx1 < cx < zx2 and zy1 < cy < zy2
            if not was_in and is_in:
                self._entry_events.append((now, track_id))
            elif was_in and not is_in:
                self._exit_events.append((now, track_id))
            inside[track_id] = is_in

            # 速度计算
            vel, vel_kmh = self._calc_speed(track_id, cx, cy, now)

            detection_list.append({"track_id": track_id, "class_name": cls_name, "confidence": conf, "bbox": xyxy})
            speed_map[track_id] = vel_kmh

        # 更新CameraDataStore
        self._update_store(detection_list, speed_map)

        # 计算派生变量
        stats = self._compute_stats(detection_list, speed_map)

        # 绘制标注
        frame = self._annotate(frame, detections, actual_tids, speed_map)

        # 绘制进出区
        self._draw_zone(frame, now)

        # 清理过期轨迹
        self._cleanup_trails()

        return detection_list, frame, now, stats

    def _nms(self, detections: sv.Detections) -> sv.Detections:
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

    def _calc_speed(self, track_id: int, cx: float, cy: float, now: float):
        dt_sec = 0.033
        traj = self._trajectories.setdefault(track_id, [])
        traj.append((cx, cy, now))
        cutoff = now - 1.5
        self._trajectories[track_id] = [(x, y, t) for x, y, t in traj if t > cutoff]

        vel = 0.0
        lu = self._speed_last_update.get(track_id, 0.0)
        stable = self._speed_stable.get(track_id, 0.0)

        if now - lu >= 0.5:
            pts = self._trajectories[track_id]
            if len(pts) >= 5:
                xs = np.array([p[0] for p in pts])
                ys = np.array([p[1] for p in pts])
                ts = np.array([p[2] for p in pts])
                dt = ts[-1] - ts[0]
                if dt > 0.3:
                    total = sum(np.sqrt((xs[j] - xs[j-1])**2 + (ys[j] - ys[j-1])**2) for j in range(1, len(pts)))
                    speed_px = total / dt
                    vel = speed_px * PIXEL_TO_METER
                    vkmh = vel * 3.6
                    if stable > 0 and abs(vkmh - stable) < 5:
                        vkmh = stable
                    self._speed_stable[track_id] = vkmh
                    self._speed_last_update[track_id] = now
                else:
                    prev = self._prev_positions.get(track_id)
                    if prev:
                        dp = np.sqrt((cx - prev["cx"])**2 + (cy - prev["cy"])**2)
                        vel = (dp * PIXEL_TO_METER) / max(dt_sec, 0.01)
            else:
                prev = self._prev_positions.get(track_id)
                if prev:
                    dp = np.sqrt((cx - prev["cx"])**2 + (cy - prev["cy"])**2)
                    vel = (dp * PIXEL_TO_METER) / max(dt_sec, 0.01)

        vkmh = self._speed_stable.get(track_id, vel * 3.6)
        self._prev_velocities[track_id] = vel
        self._prev_positions[track_id] = {"cx": cx, "cy": cy}
        return vel, vkmh

    def _update_store(self, detection_list: list[dict], speed_map: dict[int, float]):
        cx_list = [((d["bbox"][0] + d["bbox"][2]) / 2) for d in detection_list]
        lane_count = self._estimate_lanes(cx_list)
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
            boxes=[BoundingBoxItem(track_id=int(d["track_id"]), class_name=str(d["class_name"]), confidence=float(d["confidence"]), bbox=list(d["bbox"])) for d in detection_list],
            lane_count=lane_count, avg_speed=avg_s, max_speed=max_s,
            car_count=cc, truck_count=tc, bus_count=bc, moto_count=mc,
        )

    def _estimate_lanes(self, cx_list: list[float]) -> int:
        if len(cx_list) < 3:
            return max(1, len(cx_list))
        cx_list.sort()
        gaps = [cx_list[i+1] - cx_list[i] for i in range(len(cx_list)-1)]
        mg = float(np.mean(gaps)) if gaps else 0
        if mg > 5:
            clusters = 1
            for g in gaps:
                if g > mg * 0.6:
                    clusters += 1
            return max(1, min(clusters, 8))
        return max(1, len(cx_list))

    def _compute_stats(self, dlist: list[dict], smap: dict[int, float]) -> dict:
        speeds = [smap.get(d["track_id"], 0.0) for d in dlist]
        return {"avg_speed": float(np.mean(speeds)) if speeds else 0.0, "max_speed": float(np.max(speeds)) if speeds else 0.0}

    def _annotate(self, frame: np.ndarray, detections: sv.Detections, actual_tids: set[int], smap: dict[int, float]) -> np.ndarray:
        import cv2
        drawn_set = set()
        for i in range(len(detections)):
            if detections.tracker_id is None or i >= len(detections.tracker_id):
                continue
            tid = int(detections.tracker_id[i])
            if tid not in actual_tids:
                continue
            if tid in drawn_set:
                continue
            drawn_set.add(tid)
            xyxy = detections.xyxy[i].tolist() if detections.xyxy is not None else [0,0,0,0]
            x1, y1, x2, y2 = map(int, xyxy)
            cid = int(detections.class_id[i]) if detections.class_id is not None else -1
            nm = self.class_names.get(cid, "?")
            spd = smap.get(tid, 0.0)
            color = TRACK_COLORS[tid % len(TRACK_COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            lbl = f"{nm} {spd:.0f}km/h"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(frame, lbl, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        return frame

    def _draw_zone(self, frame: np.ndarray, now: float):
        import cv2
        self._prune_events(self._entry_events, now)
        self._prune_events(self._exit_events, now)
        cv2.rectangle(frame, (self.zx1, self.zy1), (self.zx2, self.zy2), (0, 255, 255), 2)
        cv2.putText(frame, "COUNT ZONE", (self.zx1 + 5, self.zy1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        flow = self._get_flow()
        for j, txt in enumerate([f"Entry:{flow['entry_count']}", f"Exit:{flow['exit_count']}", f"Flow:{flow['flow_per_min']}/min"]):
            cv2.putText(frame, txt, (self.zx1 + 5, self.zy2 - 10 - j * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    def _prune_events(self, events: collections.deque, now: float):
        while events and now - events[0][0] > 60:
            events.popleft()

    def _get_flow(self) -> dict:
        now = time.time()
        self._prune_events(self._entry_events, now)
        self._prune_events(self._exit_events, now)
        total = len(self._entry_events) + len(self._exit_events)
        return {"entry_count": len(self._entry_events), "exit_count": len(self._exit_events), "flow_per_min": round(total, 1)}

    def _cleanup_trails(self):
        stale = [tid for tid, age in self.trail_age.items() if self.frame_count - age > TRAIL_MAX_AGE]
        for tid in stale:
            self.trails.pop(tid, None)
            self.trail_age.pop(tid, None)
            self._inside_zone.pop(tid, None)
