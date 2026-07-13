import asyncio
import logging

import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.modules.stream import ProcessedVideoTrack, StreamManager

logger = logging.getLogger(__name__)

live_router = APIRouter()


class OfferRequest(BaseModel):
    sdp: str
    type: str


class AnswerResponse(BaseModel):
    sdp: str
    type: str


@live_router.get("/{camera_id}/mjpeg")
async def camera_mjpeg(camera_id: str):
    manager = StreamManager()
    stream = manager.subscribe(camera_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    async def gen():
        blank = cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes()
        while True:
            frame_rgb, frame_id = stream.get_latest_frame()
            if frame_rgb is None:
                frame = blank
            else:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                _, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                frame = buf.tobytes()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            await asyncio.sleep(0.05)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@live_router.post("/{camera_id}/offer", response_model=AnswerResponse)
async def webrtc_offer(camera_id: str, offer: OfferRequest):
    manager = StreamManager()
    stream = manager.subscribe(camera_id)
    if stream is None:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")

    pc = RTCPeerConnection()
    video_track = ProcessedVideoTrack(stream)
    pc.addTrack(video_track)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await pc.close()
            manager.unsubscribe(camera_id)

    offer_desc = RTCSessionDescription(sdp=offer.sdp, type=offer.type)
    await pc.setRemoteDescription(offer_desc)

    answer_desc = await pc.createAnswer()
    await pc.setLocalDescription(answer_desc)

    return AnswerResponse(sdp=pc.localDescription.sdp, type=pc.localDescription.type)
