import logging

from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import APIRouter, HTTPException
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
