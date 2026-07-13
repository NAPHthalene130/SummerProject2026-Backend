import logging
import os
from urllib.request import urlopen

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.modules.stream import ProcessedVideoTrack, StreamManager

try:
    from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer, RTCSessionDescription
    _HAS_AIORTC = True
except Exception:
    _HAS_AIORTC = False

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


RELAY_BASE = os.getenv("SP2026_RELAY_URL", "http://127.0.0.1:8889")


@live_router.get("/{camera_id}/mjpeg")
async def camera_mjpeg(camera_id: str):
    relay_url = f"{RELAY_BASE}/{camera_id}"
    
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                req = client.build_request("GET", relay_url)
                resp = await client.send(req, stream=True)
                
                async def proxy_stream():
                    try:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
                    except Exception:
                        pass
                
                return StreamingResponse(
                    proxy_stream(),
                    media_type="multipart/x-mixed-replace; boundary=frame",
                )
        except Exception:
            if attempt < 2:
                import asyncio
                await asyncio.sleep(1)
                continue
            raise HTTPException(status_code=502, detail="Relay connection failed")


@live_router.post("/{camera_id}/offer", response_model=AnswerResponse)
async def webrtc_offer(camera_id: str, offer: OfferRequest):
    if not _HAS_AIORTC:
        raise HTTPException(status_code=501, detail="WebRTC not available (aiortc/cryptography)")
    manager = StreamManager()
    stream = manager.subscribe(camera_id)
    if stream is None:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")

    if ICE_SERVERS:
        ice_servers = [RTCIceServer(urls=s["urls"], username=s.get("username"), credential=s.get("credential")) for s in ICE_SERVERS]
        pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))
    else:
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
