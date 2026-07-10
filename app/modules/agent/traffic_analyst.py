import asyncio
import base64
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from openai import AsyncOpenAI

from app.config import llm_settings
from app.modules.camera_data import (
    CONSECUTIVE_NORMAL_THRESHOLD,
    CameraDataStore,
    TrafficIncidentResult,
    get_incident_rank,
)
from app.modules.stream.stream_manager import StreamManager
from app.repository.work_order_repository import WorkOrderRepository
from app.utils.camera_manager import CameraManager

logger = logging.getLogger(__name__)

ANALYSIS_INTERVAL = 5.0
MAX_CONCURRENT_ANALYSES = 30
SCHEDULER_MAX_SLEEP = 0.5
JPEG_QUALITY = 85


class TrafficAnalyst:
    _instance: Optional["TrafficAnalyst"] = None

    def __new__(cls) -> "TrafficAnalyst":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._client: Optional[AsyncOpenAI] = None
            cls._instance._prompt: str = ""
            cls._instance._task: Optional[asyncio.Task] = None
            cls._instance._camera_tasks: dict[str, asyncio.Task] = {}
            cls._instance._next_analysis: dict[str, float] = {}
            cls._instance._analysis_semaphore: Optional[asyncio.Semaphore] = None
        return cls._instance

    def _load_prompt(self) -> str:
        prompt_path = Path(__file__).resolve().parent / "prompts" / "traffic_incident_detection.md"
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _init_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=llm_settings.url,
            api_key=llm_settings.api_key,
        )

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._prompt = self._load_prompt()
        self._client = self._init_client()
        self._camera_tasks = {}
        self._next_analysis = {}
        self._analysis_semaphore = asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)
        logger.info(
            "TrafficAnalyst started, model=%s, interval=%ss, max_concurrency=%d",
            llm_settings.model_name,
            ANALYSIS_INTERVAL,
            MAX_CONCURRENT_ANALYSES,
        )
        self._task = asyncio.create_task(self._analysis_loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("TrafficAnalyst scheduler stopped unexpectedly")
            self._task = None
        await self._cancel_camera_tasks()
        self._next_analysis.clear()
        self._analysis_semaphore = None
        if self._client is not None:
            await self._client.close()
            self._client = None
        logger.info("TrafficAnalyst stopped")

    async def _analysis_loop(self) -> None:
        loop = asyncio.get_running_loop()

        try:
            while True:
                try:
                    streams = StreamManager().get_all_streams()
                    await self._remove_inactive_cameras(set(streams))
                    now = loop.time()
                    skipped_camera_ids: list[str] = []

                    for camera_id, stream in streams.items():
                        next_analysis = self._next_analysis.setdefault(camera_id, now)
                        if now < next_analysis:
                            continue

                        intervals_elapsed = int((now - next_analysis) // ANALYSIS_INTERVAL) + 1
                        self._next_analysis[camera_id] = next_analysis + intervals_elapsed * ANALYSIS_INTERVAL

                        current_task = self._camera_tasks.get(camera_id)
                        if current_task is not None and not current_task.done():
                            skipped_camera_ids.append(camera_id)
                            continue

                        self._camera_tasks[camera_id] = asyncio.create_task(
                            self._analyze_camera_safely(stream, camera_id),
                            name=f"traffic-analysis-{camera_id}",
                        )

                    if skipped_camera_ids:
                        logger.warning(
                            "Skipped this interval for %d camera(s) with analysis still running: %s",
                            len(skipped_camera_ids),
                            ", ".join(skipped_camera_ids),
                        )

                    await asyncio.sleep(self._scheduler_delay(loop.time()))
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Traffic analysis scheduler iteration failed")
                    await asyncio.sleep(SCHEDULER_MAX_SLEEP)
        finally:
            await self._cancel_camera_tasks()

    def _scheduler_delay(self, now: float) -> float:
        if not self._next_analysis:
            return SCHEDULER_MAX_SLEEP
        next_due = min(self._next_analysis.values())
        return min(SCHEDULER_MAX_SLEEP, max(0.0, next_due - now))

    async def _remove_inactive_cameras(self, active_camera_ids: set[str]) -> None:
        stale_camera_ids = set(self._next_analysis) - active_camera_ids
        cancelled_tasks: list[asyncio.Task] = []

        for camera_id in stale_camera_ids:
            self._next_analysis.pop(camera_id, None)
            task = self._camera_tasks.pop(camera_id, None)
            if task is not None and not task.done():
                task.cancel()
                cancelled_tasks.append(task)

        if cancelled_tasks:
            await asyncio.gather(*cancelled_tasks, return_exceptions=True)

    async def _cancel_camera_tasks(self) -> None:
        tasks = list(self._camera_tasks.values())
        self._camera_tasks.clear()
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _analyze_camera_safely(self, stream, camera_id: str) -> None:
        semaphore = self._analysis_semaphore
        if semaphore is None:
            return

        started_at = time.perf_counter()
        try:
            async with semaphore:
                await self._analyze_camera(stream, camera_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Analysis failed for camera %s", camera_id)
        finally:
            logger.debug(
                "Camera %s analysis task finished in %.3fs",
                camera_id,
                time.perf_counter() - started_at,
            )

    async def _analyze_camera(self, stream, camera_id: str) -> None:
        frame, frame_id = stream.get_raw_frame()
        if frame is None:
            logger.debug("No raw frame available for camera %s", camera_id)
            return

        image_b64 = await asyncio.to_thread(self._frame_to_base64, frame)
        result = await self._call_multimodal(image_b64)
        logger.info("Camera %s VLM raw response: %s", camera_id, result)
        incident = self._parse_response(camera_id, result)

        store = CameraDataStore()
        store.update_incident_result(camera_id, incident)

        logger.info(
            "Camera %s analysis: incident=%s, type=%s",
            camera_id,
            incident.incident_detected,
            incident.incident_type,
        )

        if incident.incident_detected:
            newly_active = store.activate_incident(camera_id)
            if newly_active:
                image_path = await asyncio.to_thread(self._save_incident_frame, frame, camera_id)
                await asyncio.to_thread(self._create_work_order, camera_id, incident, image_path)
        else:
            if store.is_incident_active(camera_id):
                cleared = store.register_normal_result(camera_id)
                if cleared:
                    logger.info(
                        "Camera %s incident cleared after %d consecutive normal frames",
                        camera_id,
                        CONSECUTIVE_NORMAL_THRESHOLD,
                    )

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return base64.b64encode(buffer).decode("utf-8")

    async def _call_multimodal(self, image_b64: str) -> str:
        response = await self._client.chat.completions.create(
            model=llm_settings.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self._prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                            },
                        },
                    ],
                },
            ],
            max_tokens=500,
            temperature=0.1,
        )
        return response.choices[0].message.content or "{}"

    def _parse_response(self, camera_id: str, raw_response: str) -> TrafficIncidentResult:
        text = raw_response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        default = TrafficIncidentResult(
            camera_id=camera_id,
            incident_detected=False,
            incident_type="正常",
            description="模型返回解析失败",
        )

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from model response: %s", raw_response[:200])
            return default

        if not isinstance(data, dict):
            return default

        incident_detected = bool(data.get("incident_detected", False))
        incident_type = str(data.get("incident_type", "正常"))
        description = str(data.get("description", ""))

        return TrafficIncidentResult(
            camera_id=camera_id,
            incident_detected=incident_detected,
            incident_type=incident_type,
            description=description,
        )

    def _save_incident_frame(self, frame: np.ndarray, camera_id: str) -> str:
        img_dir = Path(__file__).resolve().parent.parent.parent / "data" / "orderImg"
        img_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"incident_{camera_id}_{timestamp}.jpg"
        filepath = img_dir / filename
        cv2.imwrite(str(filepath), frame)
        logger.info("Incident frame saved: %s", filepath)
        return f"/orderImg/{filename}"

    def _create_work_order(self, camera_id: str, incident: TrafficIncidentResult, image_path: str) -> None:
        cam_config = CameraManager().get_by_id(camera_id)
        camera_name = cam_config.name if cam_config else camera_id
        rank = get_incident_rank(incident.incident_type)
        work_order = WorkOrderRepository.create_work_order(
            camera_id=camera_id,
            camera_name=camera_name,
            incident_type=incident.incident_type,
            description=incident.description,
            rank=rank,
            image_url=image_path,
        )
        if work_order:
            logger.info(
                "Work order created: %s for camera %s [type=%s, rank=%d]",
                work_order.work_order_id,
                camera_id,
                incident.incident_type,
                rank,
            )
        else:
            logger.error("Failed to create work order for camera %s", camera_id)
