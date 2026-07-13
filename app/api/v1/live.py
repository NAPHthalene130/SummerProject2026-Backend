import asyncio
import logging
from collections.abc import Awaitable, Callable

from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.modules.stream import ProcessedVideoTrack, StreamManager

logger = logging.getLogger(__name__)

live_router = APIRouter()
DISCONNECT_GRACE_SECONDS = 8.0
CONNECT_TIMEOUT_SECONDS = 30.0
NEGOTIATION_TIMEOUT_SECONDS = 20.0

PeerRelease = Callable[[], Awaitable[None]]
_peer_releases: dict[RTCPeerConnection, PeerRelease] = {}


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

    pc: RTCPeerConnection | None = None
    released = False
    disconnect_task: asyncio.Task | None = None
    connect_timeout_task: asyncio.Task | None = None

    async def release() -> None:
        nonlocal released, disconnect_task, connect_timeout_task
        if released:
            return
        released = True
        current_task = asyncio.current_task()
        if disconnect_task is not None and disconnect_task is not current_task:
            disconnect_task.cancel()
        if connect_timeout_task is not None and connect_timeout_task is not current_task:
            connect_timeout_task.cancel()
        if pc is not None:
            _peer_releases.pop(pc, None)
            if pc.connectionState != "closed":
                try:
                    await pc.close()
                except Exception:
                    logger.exception("WebRTC close failed for camera %s", camera_id)
        # The last subscriber may stop a blocking RTSP read.  Keep that work off
        # FastAPI's event loop so one broken camera cannot stall all WebRTC peers.
        try:
            await asyncio.to_thread(manager.unsubscribe, camera_id)
        except Exception:
            logger.exception("WebRTC unsubscribe failed for camera %s", camera_id)

    async def close_if_still_disconnected() -> None:
        try:
            await asyncio.sleep(DISCONNECT_GRACE_SECONDS)
            if pc is not None and pc.connectionState == "disconnected":
                logger.info(
                    "WebRTC disconnect grace expired for camera %s",
                    camera_id,
                )
                await release()
        except asyncio.CancelledError:
            pass

    async def close_if_never_connected() -> None:
        try:
            await asyncio.sleep(CONNECT_TIMEOUT_SECONDS)
            if pc is not None and pc.connectionState in {"new", "connecting"}:
                logger.info("WebRTC connect timeout for camera %s", camera_id)
                await release()
        except asyncio.CancelledError:
            pass

    try:
        pc = RTCPeerConnection()
        _peer_releases[pc] = release
        video_track = ProcessedVideoTrack(stream)
        pc.addTrack(video_track)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            nonlocal disconnect_task, connect_timeout_task
            state = pc.connectionState
            logger.info("WebRTC camera=%s state=%s", camera_id, state)
            if state == "connected":
                if connect_timeout_task is not None:
                    connect_timeout_task.cancel()
                    connect_timeout_task = None
                if disconnect_task is not None:
                    disconnect_task.cancel()
                    disconnect_task = None
            elif state == "disconnected":
                # ICE can recover from a transient network switch.  Immediate
                # closure made mobile clients appear unable to reconnect.
                if disconnect_task is None or disconnect_task.done():
                    disconnect_task = asyncio.create_task(
                        close_if_still_disconnected()
                    )
            elif state in {"failed", "closed"}:
                await release()

        async def negotiate() -> None:
            offer_desc = RTCSessionDescription(sdp=offer.sdp, type=offer.type)
            await pc.setRemoteDescription(offer_desc)
            answer_desc = await pc.createAnswer()
            await pc.setLocalDescription(answer_desc)

        await asyncio.wait_for(negotiate(), timeout=NEGOTIATION_TIMEOUT_SECONDS)

        if pc.localDescription is None:
            raise RuntimeError("WebRTC local description was not created")
        if pc.connectionState in {"new", "connecting"}:
            connect_timeout_task = asyncio.create_task(close_if_never_connected())
        return AnswerResponse(
            sdp=pc.localDescription.sdp,
            type=pc.localDescription.type,
        )
    except asyncio.CancelledError:
        await release()
        raise
    except Exception as error:
        logger.warning(
            "WebRTC negotiation failed for camera %s: %s",
            camera_id,
            error,
        )
        await release()
        raise HTTPException(status_code=400, detail="Invalid WebRTC offer") from error


async def close_all_peer_connections() -> None:
    """Close active peers before stopping their shared camera streams."""
    releases = list(_peer_releases.values())
    if releases:
        await asyncio.gather(*(release() for release in releases), return_exceptions=True)
