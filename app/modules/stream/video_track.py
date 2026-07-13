import asyncio
import time
from fractions import Fraction

import av
import cv2
import numpy as np

from aiortc import VideoStreamTrack

from app.config import settings
from app.modules.stream.camera_stream import CameraStream, TARGET_FPS, TEST_FRAME_H, TEST_FRAME_W

PTS_STEP = 90000 // TARGET_FPS
FRAME_INTERVAL = 1.0 / TARGET_FPS
WEBRTC_MAX_WIDTH = settings.WEBRTC_MAX_WIDTH


class ProcessedVideoTrack(VideoStreamTrack):
    """WebRTC track backed by the latest raw RTSP frame.

    Detection metadata is rendered in the browser.  Keeping the transport on
    the raw frame means video cadence is no longer coupled to the shared YOLO
    throughput.  Frames are converted to yuv420p here so aiortc's encoder does
    not repeat RGB/YUV conversion when it has to resend the same source frame.
    """

    kind = "video"

    def __init__(self, camera_stream: CameraStream):
        super().__init__()
        self._stream = camera_stream
        self._pts = 0
        self._next_frame_time: float | None = None
        self._last_frame_id: int = -1
        self._last_source_frame: np.ndarray | None = None
        self._cached_video_frame: av.VideoFrame | None = None
        self._offline_frames: dict[str, av.VideoFrame] = {}

    async def recv(self) -> av.VideoFrame:
        now = time.perf_counter()
        if self._next_frame_time is None:
            self._next_frame_time = now
        else:
            delay = self._next_frame_time - now
            if delay > 0:
                await asyncio.sleep(delay)

        send_time = time.perf_counter()
        if send_time - self._next_frame_time > FRAME_INTERVAL * 2:
            # Drop accumulated timing debt after an encoder or event-loop stall.
            self._next_frame_time = send_time
        self._next_frame_time += FRAME_INTERVAL

        # CameraStream replaces raw arrays atomically and never mutates an
        # already published one, so WebRTC does not need another full-frame copy.
        frame_bgr, frame_id = self._stream.get_raw_frame(copy=False)
        if frame_bgr is None:
            state = self._stream.connection_state
            video_frame = self._offline_frames.get(state)
            if video_frame is None:
                video_frame = self._make_offline_frame(state)
                self._offline_frames[state] = video_frame
            self._last_source_frame = None
            self._last_frame_id = -1
            self._cached_video_frame = None
        else:
            is_new = (
                frame_id != self._last_frame_id
                or frame_bgr is not self._last_source_frame
                or self._cached_video_frame is None
            )
            if is_new:
                self._cached_video_frame = self._prepare_video_frame(frame_bgr)
                self._last_frame_id = frame_id
                self._last_source_frame = frame_bgr
            video_frame = self._cached_video_frame

        video_frame.pts = self._pts
        video_frame.time_base = Fraction(1, 90000)
        self._pts += PTS_STEP
        return video_frame

    @staticmethod
    def _prepare_video_frame(frame_bgr: np.ndarray) -> av.VideoFrame:
        height, width = frame_bgr.shape[:2]
        target_width = min(width, WEBRTC_MAX_WIDTH)
        target_height = max(2, round(height * target_width / width))
        # yuv420p requires even chroma dimensions.
        target_width = max(2, target_width - target_width % 2)
        target_height = max(2, target_height - target_height % 2)
        if (target_width, target_height) != (width, height):
            frame_bgr = cv2.resize(
                frame_bgr,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )
        return av.VideoFrame.from_ndarray(
            frame_bgr,
            format="bgr24",
        ).reformat(format="yuv420p")

    @staticmethod
    def _make_offline_frame(state: str) -> av.VideoFrame:
        frame_rgb = np.zeros((TEST_FRAME_H, TEST_FRAME_W, 3), dtype=np.uint8)
        frame_rgb[:] = (18, 20, 24)
        label = f"STREAM {state.upper()}"
        cv2.putText(
            frame_rgb,
            label,
            (48, TEST_FRAME_H // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (210, 210, 210),
            2,
            cv2.LINE_AA,
        )
        return av.VideoFrame.from_ndarray(
            frame_rgb,
            format="rgb24",
        ).reformat(format="yuv420p")
