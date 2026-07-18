"""工具层 VLM / 图片加载 / Live API 扩展测试。"""

import base64
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.api.v1 import live as live_api
from app.modules.agent import tool
from app.repository.user_repository import UserNameAlreadyExistsError, UserRepository


class VLMClientTest(unittest.TestCase):
    def test_reuses_singleton_client(self) -> None:
        tool._vlm_client = None

        with patch("openai.OpenAI") as mock_openai:
            client_a = tool._get_vlm_client()
            client_b = tool._get_vlm_client()

        self.assertIs(client_a, client_b)
        mock_openai.assert_called_once()

    def test_call_vlm_handles_api_failure(self) -> None:
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda model, messages, max_tokens, temperature:
                    (_ for _ in ()).throw(RuntimeError("api error"))
                )
            )
        )
        tool._vlm_client = client

        result = tool._call_vlm_with_images(
            [{"data": "data:image/jpeg;base64,AA==", "filename": "t.jpg"}],
            "描述这张图",
        )

        self.assertIn("调用失败", result)

    def test_call_vlm_returns_analysis_result(self) -> None:
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kw: SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content="两车追尾"))]
                    )
                )
            )
        )
        tool._vlm_client = client

        result = tool._call_vlm_with_images(
            [{"data": "data:image/jpeg;base64,AA==", "filename": "t.jpg"}],
            "描述这张图",
        )

        self.assertEqual(result, "两车追尾")

    def test_call_vlm_handles_none_content(self) -> None:
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kw: SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
                    )
                )
            )
        )
        tool._vlm_client = client

        result = tool._call_vlm_with_images([{"data": "data:image/jpeg;base64,AA=="}], "desc")

        self.assertIn("无返回内容", result)

    def test_error_images_are_skipped_in_vlm_call(self) -> None:
        created_args = {}

        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kw: (
                        created_args.update(kw),
                        SimpleNamespace(
                            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
                        ),
                    )[-1]
                )
            )
        )
        tool._vlm_client = client

        # error image should be excluded from content array
        tool._call_vlm_with_images(
            [{"error": "file missing"}, {"data": "data:image/jpeg;base64,AA=="}],
            "desc",
        )

        content = created_args.get("messages")[0]["content"]
        image_count = sum(1 for item in content if item["type"] == "image_url")
        self.assertEqual(image_count, 1)


class UserRepositoryDTOTest(unittest.TestCase):
    def test_user_name_already_exists_is_value_error(self) -> None:
        exc = UserNameAlreadyExistsError("李明")

        self.assertIsInstance(exc, ValueError)

    def test_user_response_model_serialization(self) -> None:
        from app.models.user import UserResponse

        user = UserResponse(user_id=1, user_name="张三", user_type="巡检员")

        self.assertEqual(user.model_dump(), {"user_id": 1, "user_name": "张三", "user_type": "巡检员", "user_work_describe": None})

    def test_user_create_request_password_length(self) -> None:
        from app.models.user import UserCreateRequest
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            UserCreateRequest(user_name="张三", user_type="巡检员", password="1234567")  # too short

    def test_user_record_model(self) -> None:
        from app.models.user import UserRecord

        record = UserRecord(user_id=1, user_name="张三", user_password="hash", user_type="交警")

        self.assertEqual(record.user_password, "hash")

    def test_mobile_user_response_extra_fields(self) -> None:
        from app.models.user import MobileUserResponse

        user = MobileUserResponse(
            user_id=1, name="张三", phone="13800000000",
            personnel_category="traffic_police", role_name="交警执法",
            site="海淀区", extra_field="ignored",
        )

        self.assertEqual(user.role_name, "交警执法")

    def test_staff_response_model(self) -> None:
        from app.models.user import StaffResponse

        staff = StaffResponse(id="1", name="张三", role="交警执法", distance_km=0.5)

        self.assertEqual(staff.personnel_category, "traffic_police")
        self.assertIsNone(staff.user_work_describe)


class LiveApiExtrasTest(unittest.TestCase):
    def test_peer_cleanup_idempotent(self) -> None:
        import asyncio

        async def run():
            await live_api.close_all_peer_connections()
            await live_api.close_all_peer_connections()  # 第二次不应抛异常

        asyncio.run(run())


class MobileReportModelTest(unittest.TestCase):
    def test_mobile_report_create_model(self) -> None:
        from app.models.mobile_report import MobileReportCreate, MobileReportResponse, ConvertReportRequest, RejectReportRequest

        report = MobileReportCreate(
            reporter_user_id=5, title="测试", location="路口",
            detail="详情", severity="high", event_type="车辆碰撞",
            image_urls=["/img/1.jpg"],
        )
        self.assertEqual(report.severity, "high")

        convert = ConvertReportRequest(required_category="traffic_police")
        self.assertEqual(convert.required_category, "traffic_police")

        reject = RejectReportRequest(review_message="不实")
        self.assertEqual(reject.review_message, "不实")


if __name__ == "__main__":
    unittest.main()
