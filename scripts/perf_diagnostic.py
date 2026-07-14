#!/usr/bin/env python3
"""YOLO 推理链路性能诊断脚本。

运行方式：
    cd SummerProject2026-Backend
    python scripts/perf_diagnostic.py

测量各环节耗时，定位 30 路视频流卡顿的真正瓶颈：
1. YOLO 推理时间（不同 batch size）
2. NMS 时间（torchvision vs python）
3. supervision 标注时间（BoxAnnotator + LabelAnnotator + TraceAnnotator）
4. FrameProcessor.process() 端到端后处理时间
5. av.VideoFrame 创建时间（WebRTC 编码前置）
6. 理论帧率与吞吐量计算
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

import numpy as np

# 确保能导入 app 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def bench(fn, iterations=20, warmup=3):
    """运行 benchmark，返回平均耗时(ms)和标准差。"""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    return avg, std


def main():
    import cv2

    # ================================================================
    # 1. YOLO 推理时间
    # ================================================================
    sep("1. YOLO 推理时间（不同 batch size）")
    try:
        import torch
        from ultralytics import YOLO

        model_path = "app/modules/yolo/models/best.pt"
        if not os.path.exists(model_path):
            model_path = "best.pt"
        model = YOLO(model_path)
        device = 0 if torch.cuda.is_available() else "cpu"
        half = torch.cuda.is_available()
        model.to(device)
        print(f"  模型: {model_path}")
        print(f"  设备: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
        print(f"  半精度: {half}")

        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        for bs in [1, 5, 10, 16, 30]:
            frames = [test_frame.copy() for _ in range(bs)]
            avg, std = bench(lambda: model(frames, device=device, imgsz=640, verbose=False, half=half))
            per_frame = avg / bs
            throughput = 1000 / avg * bs
            print(f"  batch={bs:2d}: {avg:6.1f}ms ± {std:.1f}ms  |  per_frame={per_frame:5.1f}ms  |  throughput={throughput:6.0f} fps")
    except Exception as e:
        print(f"  跳过: {e}")

    # ================================================================
    # 2. NMS 时间对比
    # ================================================================
    sep("2. NMS 时间对比（torchvision vs python）")
    try:
        import supervision as sv
        from app.modules.yolo.inference import FrameProcessor

        # 构造模拟检测框（50 个，模拟密集场景）
        np.random.seed(42)
        n = 50
        xyxy = np.column_stack([
            np.random.randint(0, 600, n),
            np.random.randint(0, 440, n),
            np.random.randint(0, 600, n) + 40,
            np.random.randint(0, 440, n) + 40,
        ]).astype(np.float32)
        confidence = np.random.rand(n).astype(np.float32) * 0.5 + 0.5
        class_id = np.zeros(n, dtype=int)

        det = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        fp = FrameProcessor("test-cam", {0: "car"})

        avg_py, std_py = bench(lambda: fp._nms_python(det))
        print(f"  Python NMS:    {avg_py:6.2f}ms ± {std_py:.2f}ms")

        try:
            import torch
            from torchvision.ops import nms as tv_nms
            boxes = torch.from_numpy(xyxy)
            scores = torch.from_numpy(confidence)

            def tv_nms_fn():
                keep = tv_nms(boxes, scores, 0.5)
                return det[keep.cpu().numpy()]

            avg_tv, std_tv = bench(tv_nms_fn)
            print(f"  Torchvision NMS: {avg_tv:6.2f}ms ± {std_tv:.2f}ms")
            print(f"  加速比: {avg_py / avg_tv:.1f}x")
        except ImportError:
            print("  Torchvision 不可用")
    except Exception as e:
        print(f"  跳过: {e}")

    # ================================================================
    # 3. supervision 标注时间
    # ================================================================
    sep("3. supervision 标注时间（BoxAnnotator + LabelAnnotator + TraceAnnotator）")
    try:
        import supervision as sv

        COLOR_PALETTE = sv.ColorPalette.DEFAULT
        box_annotator = sv.BoxAnnotator(color=COLOR_PALETTE, thickness=2)
        label_annotator = sv.LabelAnnotator(
            color=COLOR_PALETTE, text_color=sv.Color.WHITE,
            text_scale=0.35, text_thickness=1,
        )
        trace_annotator = sv.TraceAnnotator(color=COLOR_PALETTE, position=sv.Position.CENTER, trace_length=30)

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        n = 20
        xyxy = np.column_stack([
            np.random.randint(0, 600, n),
            np.random.randint(0, 440, n),
            np.random.randint(0, 600, n) + 40,
            np.random.randint(0, 440, n) + 40,
        ]).astype(np.float32)
        confidence = np.random.rand(n).astype(np.float32) * 0.5 + 0.5
        class_id = np.zeros(n, dtype=int)
        tracker_id = np.arange(1, n + 1, dtype=int)
        det = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id, tracker_id=tracker_id)
        labels = [f"car {i}km/h" for i in range(n)]

        def annotate_fn():
            f = frame.copy()
            f = box_annotator.annotate(scene=f, detections=det)
            f = label_annotator.annotate(scene=f, detections=det, labels=labels)
            f = trace_annotator.annotate(scene=f, detections=det)

        avg, std = bench(annotate_fn)
        print(f"  20 个目标标注: {avg:6.2f}ms ± {std:.2f}ms")
        print(f"  30 路 × 15fps = 450 次/s → 总 CPU 耗时: {avg * 450:.0f}ms/s")
        print(f"  12 个 post worker 分摊: {avg * 450 / 12:.0f}ms/s/worker")
    except Exception as e:
        print(f"  跳过: {e}")

    # ================================================================
    # 4. FrameProcessor.process() 端到端时间
    # ================================================================
    sep("4. FrameProcessor.process() 端到端后处理时间")
    try:
        from app.modules.yolo.inference import FrameProcessor
        from ultralytics import YOLO

        model_path = "app/modules/yolo/models/best.pt"
        if not os.path.exists(model_path):
            model_path = "best.pt"
        model = YOLO(model_path)
        device = 0 if torch.cuda.is_available() else "cpu"
        half = torch.cuda.is_available()
        model.to(device)

        fp = FrameProcessor("test-cam", model.names)
        fp.init_zone(480, 640)
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # 先做一次真实推理拿到 results
        results = model(test_frame, device=device, imgsz=640, verbose=False, half=half)

        def process_fn():
            f = test_frame.copy()
            fp.process(model, f, device, results[0])

        avg, std = bench(process_fn, iterations=15, warmup=2)
        print(f"  单帧 process() 端到端: {avg:6.1f}ms ± {std:.1f}ms")
        print(f"  30 路分摊到 12 worker: {avg * 30 / 12:.1f}ms/批")
        print(f"  理论最大帧率(30路): {1000 / (avg * 30 / 12):.1f} fps/路")
    except Exception as e:
        print(f"  跳过: {e}")

    # ================================================================
    # 5. av.VideoFrame 创建时间（WebRTC 编码前置）
    # ================================================================
    sep("5. av.VideoFrame 创建时间（WebRTC 编码前置）")
    try:
        import av

        frame_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        def av_fn():
            vf = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")

        avg, std = bench(av_fn)
        print(f"  VideoFrame.from_ndarray: {avg:6.2f}ms ± {std:.2f}ms")
        print(f"  30 路 WebRTC × 15fps = 450 次/s → 总耗时: {avg * 450:.0f}ms/s")
    except Exception as e:
        print(f"  跳过: {e}")

    # ================================================================
    # 6. cv2.putText + cv2.rectangle 时间
    # ================================================================
    sep("6. cv2 绘制时间（COUNT ZONE 矩形 + 3 行文字）")
    try:
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        def cv2_fn():
            f = frame.copy()
            cv2.rectangle(f, (96, 72), (544, 408), (0, 255, 255), 2)
            cv2.putText(f, "COUNT ZONE", (101, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            for j, txt in enumerate(["Entry:5", "Exit:3", "Flow:8/min"]):
                cv2.putText(f, txt, (101, 398 - j * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        avg, std = bench(cv2_fn)
        print(f"  cv2 绘制: {avg:6.2f}ms ± {std:.2f}ms")
    except Exception as e:
        print(f"  跳过: {e}")

    # ================================================================
    # 7. 理论帧率与吞吐量分析
    # ================================================================
    sep("7. 理论帧率与吞吐量分析")
    cams = 30
    target_fps = 15
    total_fps = cams * target_fps
    print(f"  摄像头数: {cams}")
    print(f"  目标帧率: {target_fps} fps/路")
    print(f"  总帧率需求: {total_fps} fps")
    print(f"  每帧时间预算: {1000/total_fps:.1f} ms")
    print()
    print(f"  注意：以上为各环节独立测量。实际瓶颈取决于最慢环节。")
    print(f"  如果 YOLO 推理 + 后处理 + WebRTC 编码 的总和 > {1000/target_fps:.1f}ms，")
    print(f"  则无法维持 {target_fps}fps × {cams} 路。")


if __name__ == "__main__":
    main()
