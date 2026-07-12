import asyncio
import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np

from app.modules.agent.traffic_analyst import MAX_CONCURRENT_ANALYSES, TrafficAnalyst


class _FakeStream:
    def __init__(self):
        self._frame = np.zeros((4, 4, 3), dtype=np.uint8)

    def get_raw_frame(self) -> tuple[np.ndarray, int]:
        return self._frame.copy(), 1


class _FakeStreamManager:
    def __init__(self, stream_count: int):
        self._streams = {f"cam-{index:02d}": _FakeStream() for index in range(1, stream_count + 1)}

    def get_all_streams(self) -> dict[str, _FakeStream]:
        return dict(self._streams)


class TrafficAnalystConcurrencyTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.analyst = TrafficAnalyst()
        self.analyst._camera_tasks = {}
        self.analyst._next_analysis = {}
        self.analyst._analysis_semaphore = asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)

    async def asyncTearDown(self) -> None:
        await self.analyst._cancel_camera_tasks()
        self.analyst._next_analysis = {}
        self.analyst._analysis_semaphore = None

    async def test_scheduler_starts_thirty_camera_analyses_concurrently(self) -> None:
        manager = _FakeStreamManager(MAX_CONCURRENT_ANALYSES)
        started_camera_ids: set[str] = set()
        all_started = asyncio.Event()
        release = asyncio.Event()

        async def fake_analyze(stream: object, camera_id: str) -> None:
            started_camera_ids.add(camera_id)
            if len(started_camera_ids) == MAX_CONCURRENT_ANALYSES:
                all_started.set()
            await release.wait()

        with (
            patch("app.modules.agent.traffic_analyst.StreamManager", return_value=manager),
            patch.object(self.analyst, "_analyze_camera", side_effect=fake_analyze),
        ):
            scheduler = asyncio.create_task(self.analyst._analysis_loop())
            try:
                await asyncio.wait_for(all_started.wait(), timeout=1.0)
                self.assertEqual(len(started_camera_ids), MAX_CONCURRENT_ANALYSES)
                self.assertEqual(len(self.analyst._camera_tasks), MAX_CONCURRENT_ANALYSES)
            finally:
                scheduler.cancel()
                release.set()
                await asyncio.gather(scheduler, return_exceptions=True)

    async def test_thirty_vlm_calls_can_be_in_flight_together(self) -> None:
        manager = _FakeStreamManager(MAX_CONCURRENT_ANALYSES)
        active_calls = 0
        maximum_active_calls = 0
        all_vlm_calls_started = asyncio.Event()
        release = asyncio.Event()

        async def fake_call_multimodal(image_b64: str) -> str:
            nonlocal active_calls, maximum_active_calls
            active_calls += 1
            maximum_active_calls = max(maximum_active_calls, active_calls)
            if active_calls == MAX_CONCURRENT_ANALYSES:
                all_vlm_calls_started.set()
            try:
                await release.wait()
                return '{"incident_detected": false, "incident_type": "正常", "description": ""}'
            finally:
                active_calls -= 1

        with (
            patch("app.modules.agent.traffic_analyst.StreamManager", return_value=manager),
            patch.object(self.analyst, "_call_multimodal", side_effect=fake_call_multimodal),
        ):
            scheduler = asyncio.create_task(self.analyst._analysis_loop())
            try:
                await asyncio.wait_for(all_vlm_calls_started.wait(), timeout=1.0)
                self.assertEqual(maximum_active_calls, MAX_CONCURRENT_ANALYSES)
            finally:
                scheduler.cancel()
                release.set()
                await asyncio.gather(scheduler, return_exceptions=True)

    async def test_slow_camera_does_not_create_overlapping_calls(self) -> None:
        manager = _FakeStreamManager(MAX_CONCURRENT_ANALYSES)
        all_started = asyncio.Event()
        release = asyncio.Event()
        call_count = 0

        async def fake_analyze(stream: object, camera_id: str) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == MAX_CONCURRENT_ANALYSES:
                all_started.set()
            await release.wait()

        with (
            patch("app.modules.agent.traffic_analyst.StreamManager", return_value=manager),
            patch("app.modules.agent.traffic_analyst.ANALYSIS_INTERVAL", 0.02),
            patch.object(self.analyst, "_analyze_camera", side_effect=fake_analyze),
        ):
            scheduler = asyncio.create_task(self.analyst._analysis_loop())
            try:
                await asyncio.wait_for(all_started.wait(), timeout=1.0)
                await asyncio.sleep(0.06)
                self.assertEqual(call_count, MAX_CONCURRENT_ANALYSES)
            finally:
                scheduler.cancel()
                release.set()
                await asyncio.gather(scheduler, return_exceptions=True)

    async def test_completed_cameras_run_again_on_the_next_interval(self) -> None:
        manager = _FakeStreamManager(MAX_CONCURRENT_ANALYSES)
        call_counts: Counter[str] = Counter()
        all_repeated = asyncio.Event()

        async def fake_analyze(stream: object, camera_id: str) -> None:
            call_counts[camera_id] += 1
            if len(call_counts) == MAX_CONCURRENT_ANALYSES and min(call_counts.values()) >= 2:
                all_repeated.set()

        with (
            patch("app.modules.agent.traffic_analyst.StreamManager", return_value=manager),
            patch("app.modules.agent.traffic_analyst.ANALYSIS_INTERVAL", 0.02),
            patch.object(self.analyst, "_analyze_camera", side_effect=fake_analyze),
        ):
            scheduler = asyncio.create_task(self.analyst._analysis_loop())
            try:
                await asyncio.wait_for(all_repeated.wait(), timeout=1.0)
                self.assertEqual(len(call_counts), MAX_CONCURRENT_ANALYSES)
                self.assertTrue(all(count >= 2 for count in call_counts.values()))
            finally:
                scheduler.cancel()
                await asyncio.gather(scheduler, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
