"""Measure real frame delivery from multiple local WebRTC camera endpoints."""

import argparse
import asyncio
import time
from dataclasses import dataclass

import httpx
from aiortc import RTCPeerConnection, RTCSessionDescription


@dataclass
class CameraResult:
    camera_id: str
    frames: int
    elapsed: float
    width: int
    height: int

    @property
    def fps(self) -> float:
        return self.frames / self.elapsed if self.elapsed > 0 else 0.0


async def measure_camera(
    client: httpx.AsyncClient,
    base_url: str,
    camera_id: str,
    duration: float,
) -> CameraResult:
    peer = RTCPeerConnection()
    peer.addTransceiver("video", direction="recvonly")
    frames = 0
    first_frame_at = 0.0
    last_frame_at = 0.0
    width = 0
    height = 0
    finished = asyncio.Event()

    @peer.on("track")
    async def on_track(track) -> None:
        nonlocal frames, first_frame_at, last_frame_at, width, height
        deadline = time.perf_counter() + duration
        try:
            while time.perf_counter() < deadline:
                frame = await asyncio.wait_for(track.recv(), timeout=5.0)
                now = time.perf_counter()
                if first_frame_at == 0.0:
                    first_frame_at = now
                last_frame_at = now
                frames += 1
                width = frame.width
                height = frame.height
        finally:
            finished.set()

    try:
        await peer.setLocalDescription(await peer.createOffer())
        response = await client.post(
            f"{base_url}/api/v1/live/{camera_id}/offer",
            json={
                "sdp": peer.localDescription.sdp,
                "type": peer.localDescription.type,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        answer = response.json()
        await peer.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )
        await asyncio.wait_for(finished.wait(), timeout=duration + 10.0)
    finally:
        await peer.close()

    elapsed = max(0.0, last_frame_at - first_frame_at)
    return CameraResult(camera_id, frames, elapsed, width, height)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("camera_ids", nargs="+", help="Camera IDs to subscribe")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--duration", type=float, default=15.0)
    args = parser.parse_args()

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *(
                measure_camera(
                    client,
                    args.base_url.rstrip("/"),
                    camera_id,
                    args.duration,
                )
                for camera_id in args.camera_ids
            )
        )

    for result in results:
        print(
            f"{result.camera_id}: frames={result.frames} "
            f"fps={result.fps:.2f} resolution={result.width}x{result.height}"
        )
    aggregate_fps = sum(result.fps for result in results)
    print(f"aggregate_fps={aggregate_fps:.2f} streams={len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
