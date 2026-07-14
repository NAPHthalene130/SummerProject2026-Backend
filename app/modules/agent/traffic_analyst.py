import asyncio
import base64
import json
import logging
import math
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
MIN_INCIDENT_CONFIDENCE = 0.85
FRAME_BUFFER_SIZE = 5
JPEG_QUALITY = 85

INCIDENT_CATEGORY_MAP: dict[str, str] = {
    "车辆碰撞": "traffic_police",
    "车辆起火": "emergency_fire",
    "交通拥堵": "traffic_coordination",
    "行人闯入": "traffic_police",
    "恶劣天气": "road_maintenance",
    "车辆抛锚": "vehicle_rescue",
    "异常停车": "traffic_police",
    "道路障碍": "road_maintenance",
    "其他事故": "traffic_police",
}


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
            cls._instance._frame_buffer: dict[str, list[str]] = {}
        return cls._instance

    def _load_prompt(self) -> str:
        prompt_path = Path(__file__).resolve().parent / "prompts" / "traffic_incident_detection.md"
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_verification_prompt(self) -> str:
        prompt_path = Path(__file__).resolve().parent / "prompts" / "traffic_incident_verification.md"
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
        self._verification_prompt: str = self._load_verification_prompt()
        self._client = self._init_client()
        self._camera_tasks = {}
        self._next_analysis = {}
        self._frame_buffer = {}
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
        self._frame_buffer.clear()
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
            self._frame_buffer.pop(camera_id, None)
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

        buffer = self._frame_buffer.setdefault(camera_id, [])
        buffer.append(image_b64)
        if len(buffer) > FRAME_BUFFER_SIZE:
            buffer.pop(0)

        images = list(buffer)
        result = await self._call_multimodal(images)
        incident = self._parse_response(camera_id, result)

        store = CameraDataStore()
        store.update_incident_result(camera_id, incident)

        if incident.incident_detected:
            logger.info(
                "Camera %s analysis: incident=%s, confidence=%.3f, type=%s",
                camera_id,
                incident.incident_detected,
                incident.confidence,
                incident.incident_type,
            )

            cleared_types = store.register_analysis_result(camera_id, incident.incident_type)
            self._log_cleared_incidents(camera_id, cleared_types)

            if store.is_incident_active(camera_id, incident.incident_type):
                return

            verified, verify_reason = await self._verify_incident(
                camera_id=camera_id,
                images=images,
                incident=incident,
            )

            if not verified:
                logger.warning(
                    "Camera %s incident REJECTED by secondary verification: %s (type=%s, confidence=%.3f)",
                    camera_id,
                    verify_reason,
                    incident.incident_type,
                    incident.confidence,
                )
                self._clear_frame_buffer(camera_id)
                return

            logger.info(
                "Camera %s incident CONFIRMED by secondary verification: %s",
                camera_id,
                verify_reason,
            )

            image_path = await asyncio.to_thread(self._save_incident_frame, frame, camera_id)
            work_order_created = await asyncio.to_thread(
                self._create_work_order,
                camera_id,
                incident,
                image_path,
            )
            if work_order_created:
                store.activate_incident(camera_id, incident.incident_type)
                self._clear_frame_buffer(camera_id)
            else:
                logger.warning(
                    "Camera %s incident remains pending because work order creation failed",
                    camera_id,
                )
        else:
            cleared_types = store.register_analysis_result(camera_id, None)
            self._log_cleared_incidents(camera_id, cleared_types)

    def _log_cleared_incidents(self, camera_id: str, incident_types: list[str]) -> None:
        for incident_type in incident_types:
            logger.info(
                "Camera %s incident type %s cleared after %d consecutive normal analyses",
                camera_id,
                incident_type,
                CONSECUTIVE_NORMAL_THRESHOLD,
            )

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return base64.b64encode(buffer).decode("utf-8")

    async def _call_multimodal(self, images_b64: list[str]) -> str:
        content: list[dict] = [{"type": "text", "text": self._prompt}]
        if len(images_b64) > 1:
            content.insert(0, {
                "type": "text",
                "text": f"以下是从该摄像头采集的连续 {len(images_b64)} 帧画面（按时间顺序，最早的在最前面，最新的在最后面），请结合帧间变化进行综合分析：\n\n",
            })
        for img in images_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"},
            })
        response = await self._client.chat.completions.create(
            model=llm_settings.model_name,
            messages=[{"role": "user", "content": content}],
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
            confidence=0.0,
        )

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from model response: %s", raw_response[:200])
            return default

        if not isinstance(data, dict):
            return default

        raw_incident_detected = data.get("incident_detected", False)
        if isinstance(raw_incident_detected, bool):
            incident_detected = raw_incident_detected
        elif isinstance(raw_incident_detected, str):
            normalized = raw_incident_detected.strip().lower()
            if normalized in {"true", "false"}:
                incident_detected = normalized == "true"
            else:
                logger.warning("Invalid incident_detected value: %r", raw_incident_detected)
                incident_detected = False
        else:
            logger.warning("Invalid incident_detected type: %r", raw_incident_detected)
            incident_detected = False

        raw_confidence = data.get("confidence", 0.0)
        try:
            if isinstance(raw_confidence, bool):
                raise ValueError
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            logger.warning("Invalid incident confidence: %r", raw_confidence)
            confidence = 0.0

        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            logger.warning("Incident confidence outside [0, 1]: %r", raw_confidence)
            confidence = 0.0

        if incident_detected and confidence < MIN_INCIDENT_CONFIDENCE:
            logger.info(
                "Incident downgraded to normal because confidence %.3f is below %.3f",
                confidence,
                MIN_INCIDENT_CONFIDENCE,
            )
            incident_detected = False

        incident_type = str(data.get("incident_type", "正常")).strip()
        if incident_detected and (not incident_type or incident_type == "正常"):
            incident_type = "其他事故"
        elif not incident_detected:
            incident_type = "正常"
        description = str(data.get("description", ""))

        return TrafficIncidentResult(
            camera_id=camera_id,
            incident_detected=incident_detected,
            incident_type=incident_type,
            description=description,
            confidence=confidence,
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

    async def _verify_incident(
        self,
        camera_id: str,
        images: list[str],
        incident: TrafficIncidentResult,
    ) -> tuple[bool, str]:
        """对初次检测到的事故进行严格的二次校验。

        返回 (confirmed, reason) 二元组。
        """
        try:
            detection_info = (
                f"初次检测结果:\n"
                f"- 事故类型: {incident.incident_type}\n"
                f"- 置信度: {incident.confidence:.3f}\n"
                f"- 描述: {incident.description}\n"
                f"- 摄像头ID: {camera_id}\n"
            )
            raw_response = await self._call_verification(images, detection_info)
            confirmed, reason = self._parse_verification_response(camera_id, raw_response)
            return confirmed, reason
        except Exception as exc:
            logger.exception("Secondary verification failed for camera %s", camera_id)
            return False, f"二次校验调用异常: {exc}"

    async def _call_verification(self, images_b64: list[str], detection_info: str) -> str:
        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    f"{self._verification_prompt}\n\n"
                    f"---\n\n"
                    f"以下是初次检测模型的判定结果，请对其进行严格的独立复核：\n\n"
                    f"{detection_info}\n\n"
                    f"---\n\n"
                    f"以下是从该摄像头采集的连续 {len(images_b64)} 帧画面"
                    f"（按时间顺序，最早的在最前面，最新的在最后面）。"
                    f"请逐一检查每一帧，判断事故是否确实成立：\n\n"
                ),
            },
        ]
        for img in images_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"},
            })
        response = await self._client.chat.completions.create(
            model=llm_settings.model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=800,
            temperature=0.05,
        )
        return response.choices[0].message.content or "{}"

    def _parse_verification_response(self, camera_id: str, raw_response: str) -> tuple[bool, str]:
        text = raw_response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse verification JSON for camera %s: %s",
                camera_id,
                raw_response[:200],
            )
            return False, f"校验结果JSON解析失败,原始响应前200字符: {raw_response[:200]}"

        if not isinstance(data, dict):
            return False, "校验结果格式无效(非JSON对象)"

        raw_confirmed = data.get("confirmed", False)
        if isinstance(raw_confirmed, bool):
            confirmed = raw_confirmed
        elif isinstance(raw_confirmed, str):
            confirmed = raw_confirmed.strip().lower() == "true"
        else:
            confirmed = False

        reason = str(data.get("reason", "")).strip()
        if not reason:
            reason = "校验模型未提供裁定理由"

        logger.info(
            "Camera %s verification result: confirmed=%s, reason=%s",
            camera_id,
            confirmed,
            reason,
        )
        return confirmed, reason

    def _clear_frame_buffer(self, camera_id: str) -> None:
        """清除指定摄像头的帧缓冲,释放内存。"""
        if camera_id in self._frame_buffer:
            buffer_len = len(self._frame_buffer[camera_id])
            del self._frame_buffer[camera_id]
            logger.debug(
                "Camera %s frame buffer cleared (%d frames released)",
                camera_id,
                buffer_len,
            )

    def _create_work_order(self, camera_id: str, incident: TrafficIncidentResult, image_path: str) -> bool:
        cam_config = CameraManager().get_by_id(camera_id)
        camera_name = cam_config.name if cam_config else camera_id
        rank = get_incident_rank(incident.incident_type)
        required_category = INCIDENT_CATEGORY_MAP.get(incident.incident_type, "traffic_police")
        work_order = WorkOrderRepository.create_work_order(
            camera_id=camera_id,
            camera_name=camera_name,
            incident_type=incident.incident_type,
            description=incident.description,
            rank=rank,
            image_url=image_path,
            required_category=required_category,
        )
        if work_order:
            logger.info(
                "Work order created: %s for camera %s [type=%s, rank=%d]",
                work_order.work_order_id,
                camera_id,
                incident.incident_type,
                rank,
            )
            return True
        else:
            logger.error("Failed to create work order for camera %s", camera_id)
            return False
