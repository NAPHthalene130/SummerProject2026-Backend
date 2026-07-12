import logging
import os

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


def _parse_ice_servers(raw: str | None) -> list[dict]:
    if not raw:
        return []
    servers = []
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(",")
        urls = [p for p in parts if p.startswith("stun:") or p.startswith("turn:")]
        username = ""
        credential = ""
        for p in parts:
            if p.startswith("username="):
                username = p.split("=", 1)[1]
            elif p.startswith("credential="):
                credential = p.split("=", 1)[1]
        if urls:
            s = {"urls": urls}
            if username and credential:
                s["username"] = username
                s["credential"] = credential
            servers.append(s)
    return servers


ICE_SERVERS = _parse_ice_servers(os.getenv("SP2026_ICE_SERVERS"))


@live_router.post("/{camera_id}/offer", response_model=AnswerResponse)
async def webrtc_offer(camera_id: str, offer: OfferRequest):
    manager = StreamManager()
    stream = manager.subscribe(camera_id)
    if stream is None:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")

    pc = RTCPeerConnection() if not ICE_SERVERS else RTCPeerConnection(iceServers=ICE_SERVERS)
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
