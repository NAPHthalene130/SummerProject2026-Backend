"""用户 API 扩展测试:登录(含旧明文兼容)、列表、删除及请求模型校验。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from app.api.v1.users import delete_user, list_users, login_user
from app.models.user import UserCreateRequest, UserLoginRequest, UserResponse
from app.utils.passwords import hash_password


def _user_record(password: str) -> SimpleNamespace:
    return SimpleNamespace(
        user_id=8,
        user_name="李明",
        user_password=password,
        user_type="巡检员",
        user_work_describe=None,
    )


class LoginApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_login_with_argon2_hash(self) -> None:
        record = _user_record(hash_password("safe-password-2026"))
        with patch("app.api.v1.users.UserRepository.get_user_by_name", return_value=record):
            response = await login_user(UserLoginRequest(user_name="李明", password="safe-password-2026"))

        self.assertEqual(response.user_id, 8)
        self.assertEqual(response.user_name, "李明")
        self.assertNotIn("password", response.model_dump())

    async def test_login_accepts_legacy_plaintext_password(self) -> None:
        record = _user_record("legacy-plain")
        with patch("app.api.v1.users.UserRepository.get_user_by_name", return_value=record):
            response = await login_user(UserLoginRequest(user_name="李明", password="legacy-plain"))

        self.assertEqual(response.user_id, 8)

    async def test_login_rejects_wrong_password(self) -> None:
        record = _user_record(hash_password("safe-password-2026"))
        with patch("app.api.v1.users.UserRepository.get_user_by_name", return_value=record):
            with self.assertRaises(HTTPException) as caught:
                await login_user(UserLoginRequest(user_name="李明", password="wrong-password"))

        self.assertEqual(caught.exception.status_code, 401)

    async def test_login_rejects_unknown_user(self) -> None:
        with patch("app.api.v1.users.UserRepository.get_user_by_name", return_value=None):
            with self.assertRaises(HTTPException) as caught:
                await login_user(UserLoginRequest(user_name="不存在", password="whatever-123"))

        self.assertEqual(caught.exception.status_code, 401)


class ListAndDeleteApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_list_users_passthrough(self) -> None:
        users = [UserResponse(user_id=1, user_name="李明", user_type="巡检员")]
        with patch("app.api.v1.users.UserRepository.list_users", return_value=users):
            result = await list_users()

        self.assertEqual(result, users)

    async def test_delete_existing_user_succeeds(self) -> None:
        with patch("app.api.v1.users.UserRepository.delete_user", return_value=True) as delete:
            result = await delete_user(8)

        self.assertIsNone(result)
        delete.assert_called_once_with(8)

    async def test_delete_missing_user_raises_404(self) -> None:
        with patch("app.api.v1.users.UserRepository.delete_user", return_value=False):
            with self.assertRaises(HTTPException) as caught:
                await delete_user(999)

        self.assertEqual(caught.exception.status_code, 404)


class UserRequestValidationTest(unittest.TestCase):
    def test_short_password_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            UserCreateRequest(user_name="李明", user_type="巡检员", password="short")

    def test_blank_identity_fields_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            UserCreateRequest(user_name="   ", user_type="巡检员", password="safe-password-2026")
        with self.assertRaises(ValidationError):
            UserCreateRequest(user_name="李明", user_type=" ", password="safe-password-2026")

    def test_login_request_trims_user_name(self) -> None:
        request = UserLoginRequest(user_name="  李明  ", password="x")

        self.assertEqual(request.user_name, "李明")

    def test_login_request_rejects_blank_name(self) -> None:
        with self.assertRaises(ValidationError):
            UserLoginRequest(user_name="   ", password="x")


if __name__ == "__main__":
    unittest.main()
