"""推理模块：接收帧 → YOLO推理 → ByteTrack追踪 → 速度/车流计算"""

import collections
import logging
import time

import cv2
import numpy as np
import supervision as sv

from app.modules.camera_data import BoundingBoxItem, CameraDataStore

logger = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.1
NMS_OVERLAP = 0.5
LOST_BUFFER = 30
TRAIL_MAX_AGE = 30
PIXEL_TO_METER = 0.05

COLOR_PALETTE = sv.ColorPalette.DEFAULT


def _sv_color_from_tuple(t: tuple) -> sv.Color:
    return sv.Color(r=t[0], g=t[1], b=t[2])


class FrameProcessor:
    """单路摄像头帧处理器（基于supervision）"""

    def __init__(self, cam_id: str, class_names: dict):
        self.cam_id = cam_id
        self.class_names = class_names
        self.tracker = sv.ByteTrack(lost_track_buffer=LOST_BUFFER)
        self.trails: dict[int, list] = {}
        self.trail_age: dict[int, int] = {}
        self.frame_count = 0
        self.line_zone: sv.LineZone | None = None
        self.zone_annotator = sv.LineZoneAnnotator(text_thickness=1, text_color=sv.Color.WHITE)
        self._prev_positions: dict[int, dict] = {}
        self._prev_velocities: dict[int, float] = {}
        self._trajectories: dict[int, list] = {}
        self._speed_last_update: dict[int, float] = {}
        self._speed_stable: dict[int, float] = {}
        self._track_colors: dict[int, sv.Color] = {}

    def init_zone(self, h: int, w: int, margin: float = 0.15):
        self.line_zone = sv.LineZone(
            start=sv.Point(0, int(h * 0.6)),
            end=sv.Point(w, int(h * 0.6)),
        )

    def _get_color(self, track_id: int) -> sv.Color:
        if track_id not in self._track_colors:
            self._track_colors[track_id] = COLOR_PALETTE.by_idx(track_id % len(COLOR_PALETTE))
        return self._track_colors[track_id]

    def _get_color_bgr(self, track_id: int) -> tuple:
        c = self._get_color(track_id)
        return (c.blue, c.green, c.red)

    def process(self, model, frame: np.ndarray, device, results) -> tuple[list[dict], np.ndarray, dict]:
        self.frame_count += 1
        now = time.time()

        detections = sv.Detections.from_ultralytics(results)

        if len(detections) > 0 and detections.confidence is not None:
            detections = detections[detections.confidence >= MIN_CONFIDENCE]

        if len(detections) > 1 and detections.xyxy is not None:
            detections = self._nms(detections)

        raw_boxes = [list(b) for b in (detections.xyxy.tolist() if detections.xyxy is not None else [])]

        if len(detections) > 0:
            detections = self.tracker.update_with_detections(detections)
        else:
            detections = sv.Detections.empty()

        actual_tids: set[int] = set()
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

        filtered = detections[[i for i in range(len(detections)) if int(detections.tracker_id[i]) in actual_tids]] if len(detections) > 0 else detections

        detection_list: list[dict] = []
        speed_map: dict[int, float] = {}

        if len(filtered) > 0:
            for i in range(len(filtered)):
                tid = int(filtered.tracker_id[i])
                cid = int(filtered.class_id[i]) if filtered.class_id is not None else -1
                cls_name = self.class_names.get(cid, "?")
                conf = float(filtered.confidence[i]) if filtered.confidence is not None else 0.0
                xyxy = filtered.xyxy[i].tolist() if filtered.xyxy is not None else [0, 0, 0, 0]
                cx = (xyxy[0] + xyxy[2]) / 2
                cy = (xyxy[1] + xyxy[3]) / 2
                self.trails.setdefault(tid, []).append((cx, cy))
                if len(self.trails[tid]) > 30:
                    self.trails[tid].pop(0)
                self.trail_age[tid] = self.frame_count
                vel, vel_kmh = self._calc_speed(tid, cx, cy, now)
                detection_list.append({"track_id": tid, "class_name": cls_name, "confidence": conf, "bbox": xyxy})
                speed_map[tid] = vel_kmh

        self._update_store(detection_list, speed_map)
        stats = self._compute_stats(detection_list, speed_map)

        # supervision标注
        if len(filtered) > 0:
            labels = []
            for i in range(len(filtered)):
                tid = int(filtered.tracker_id[i])
                spd = speed_map.get(tid, 0.0)
                cid = int(filtered.class_id[i]) if filtered.class_id is not None else -1
                nm = self.class_names.get(cid, "?")
                labels.append(f"{nm} {spd:.0f}km/h")
            box_annotator = sv.BoxAnnotator(color=COLOR_PALETTE, thickness=2)
            label_annotator = sv.LabelAnnotator(color=COLOR_PALETTE, text_color=sv.Color.WHITE,
                                                text_scale=0.35, text_thickness=1)
            frame = box_annotator.annotate(scene=frame, detections=filtered)
            frame = label_annotator.annotate(scene=frame, detections=filtered, labels=labels)

        # LineZone进出计数
        if self.line_zone is not None and len(filtered) > 0:
            self.line_zone.trigger(filtered)
            self.zone_annotator.annotate(frame=frame, line_counter=self.line_zone)

        self._cleanup_trails()

        return detection_list, frame, stats

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
            boxes=[BoundingBoxItem(track_id=int(d["track_id"]), class_name=str(d["class_name"]),
                                   confidence=float(d["confidence"]), bbox=list(d["bbox"])) for d in detection_list],
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
        return {"avg_speed": float(np.mean(speeds)) if speeds else 0.0,
                "max_speed": float(np.max(speeds)) if speeds else 0.0}

    def get_traffic_flow(self) -> dict:
        if self.line_zone is None:
            return {"entry_count": 0, "exit_count": 0, "flow_per_min": 0.0}
        return {"entry_count": self.line_zone.in_count, "exit_count": self.line_zone.out_count,
                "flow_per_min": round(self.line_zone.in_count + self.line_zone.out_count, 1)}

    def _cleanup_trails(self):
        stale = [tid for tid, age in self.trail_age.items() if self.frame_count - age > TRAIL_MAX_AGE]
        for tid in stale:
            self.trails.pop(tid, None)
            self.trail_age.pop(tid, None)
