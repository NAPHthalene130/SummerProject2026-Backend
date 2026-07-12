"""
RTSP → HTTP MJPEG 中继
在校园网机器上运行，将本地 RTSP 流转为 HTTP MJPEG，供云端后端访问。
"""

import json
import logging
import os
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Optional

import cv2
import numpy as np
import yaml

os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["AV_LOG_FORCE_COLOR"] = "0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("rtsp_relay")

BACKEND_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "SummerProject2026-Backend", "config.yaml"
)
LOCAL_CONFIG = os.path.join(os.path.dirname(__file__), "cameras.yaml")
HOST = "0.0.0.0"
PORT = 8888
FPS = 10
JPEG_QUALITY = 40


class CameraCapture:
    def __init__(self, cam_id: str, name: str, url: str):
        self.cam_id = cam_id
        self.name = name
        self.url = url
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[bytes] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Started capture: [%s] %s -> %s", self.cam_id, self.name, self.url)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()
        logger.info("Stopped capture: [%s]", self.cam_id)

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._frame

    def _loop(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            logger.error("Cannot open [%s] %s", self.cam_id, self.url)
            return
        self._cap = cap
        interval = 1.0 / FPS
        while self._running:
            loop_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame drop [%s], reconnecting...", self.cam_id)
                cap.release()
                time.sleep(2)
                cap.open(self.url)
                continue
            ret_jpg, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ret_jpg:
                with self._lock:
                    self._frame = buf.tobytes()
            elapsed = time.perf_counter() - loop_start
            sleep = interval - elapsed
            if sleep > 0:
                time.sleep(sleep)
        cap.release()


class RelayHandler(BaseHTTPRequestHandler):
    cameras: dict[str, CameraCapture] = {}

    def do_GET(self):
        path = self.path.strip("/")
        if path == "":
            self._list_cameras()
        elif path in self.cameras:
            self._serve_mjpeg(path)
        elif path == "health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "cameras": list(self.cameras.keys())}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def _list_cameras(self):
        html = "<html><body><h1>RTSP Relay</h1><ul>"
        for cid, cam in self.cameras.items():
            html += f'<li><a href="/{cid}">{cam.name} [{cid}]</a></li>'
        html += "</ul></body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def _serve_mjpeg(self, cam_id: str):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Connection", "close")
        self.end_headers()
        cam = self.cameras[cam_id]
        boundary = b"--frame\r\n"
        blank_jpg = cv2.imencode(".jpg", np.zeros((240, 320, 3), dtype=np.uint8))[1].tobytes()
        max_wait = 200
        waited = 0
        while True:
            frame = cam.get_frame()
            if frame is None:
                if waited < max_wait:
                    waited += 1
                    time.sleep(0.1)
                    continue
                frame = blank_jpg
            waited = 0
            try:
                self.wfile.write(boundary)
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
            except Exception:
                break
            time.sleep(0.05)

    def log_message(self, format, *args):
        logger.debug(format, *args)


def load_cameras():
    for path in [LOCAL_CONFIG, BACKEND_CONFIG]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            cameras = data.get("cameras", [])
            if cameras:
                logger.info("Loaded %d cameras from %s", len(cameras), path)
                return cameras
    logger.warning("No camera config found. Create %s or ensure backend config.yaml exists.", LOCAL_CONFIG)
    sample = [
        {"id": "cam-01", "name": "Camera 1", "url": "rtsp://..."},
    ]
    logger.info("Using sample config (1 camera)")
    return sample


def main():
    cameras_cfg = load_cameras()
    captures = {}
    for cfg in cameras_cfg:
        cam_id = cfg.get("id", "unknown")
        cam = CameraCapture(
            cam_id=cam_id,
            name=cfg.get("name", cam_id),
            url=cfg.get("url", ""),
        )
        cam.start()
        captures[cam_id] = cam

    RelayHandler.cameras = captures
    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        allow_reuse_address = True
        daemon_threads = True
    server = ThreadedHTTPServer((HOST, PORT), RelayHandler)
    logger.info("RTSP Relay running on http://%s:%d", HOST, PORT)
    logger.info("Cameras:")
    for cid, cam in captures.items():
        logger.info("  http://%s:%d/%s  ->  %s", HOST, PORT, cid, cam.url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()
        for cam in captures.values():
            cam.stop()


if __name__ == "__main__":
    main()
