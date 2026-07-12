import asyncio
import time
from fractions import Fraction
from typing import TYPE_CHECKING, Optional

import av
import numpy as np

from app.modules.stream.camera_stream import CameraStream, TARGET_FPS, TEST_FRAME_H, TEST_FRAME_W

try:
    from aiortc import VideoStreamTrack
    _HAS_AIORTC = True
except Exception:
    _HAS_AIORTC = False

PTS_STEP = 90000 // TARGET_FPS
FRAME_INTERVAL = 1.0 / TARGET_FPS


class ProcessedVideoTrack:
    kind = "video"

    def __init__(self, camera_stream: CameraStream):
        if not _HAS_AIORTC:
            raise RuntimeError("aiortc not available (cryptography issue)")
        self._stream = camera_stream
        self._pts = 0
        self._next_frame_time: float | None = None
        self._last_frame_id: int = -1
        self._first_frame = True
        self._track = VideoStreamTrack()

    async def recv(self) -> av.VideoFrame:
        now = time.perf_counter()
        if self._next_frame_time is not None:
            delay = self._next_frame_time - now
            if delay > 0:
                await asyncio.sleep(delay)

        self._next_frame_time = time.perf_counter() + FRAME_INTERVAL

        frame_rgb, frame_id = self._stream.get_latest_frame()
        if frame_rgb is None:
            frame_rgb = np.zeros((TEST_FRAME_H, TEST_FRAME_W, 3), dtype=np.uint8)

        video_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = self._pts
        video_frame.time_base = Fraction(1, 90000)

        if self._first_frame:
            video_frame.pict_type = av.video.frame.PictureType.I
            self._first_frame = False

        self._pts += PTS_STEP

        return video_frame
