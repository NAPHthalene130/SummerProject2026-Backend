import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app.api.v1.users import create_user, delete_user, update_user
from app.models.user import UserCreateRequest, UserResponse, UserUpdateRequest
from app.repository.user_repository import UserNameAlreadyExistsError
from app.utils.passwords import hash_password, is_password_hash, verify_password


class PasswordHashTest(unittest.TestCase):
    def test_password_is_stored_as_argon2id_hash(self) -> None:
        password_hash = hash_password("safe-password-2026")

        self.assertNotEqual(password_hash, "safe-password-2026")
        self.assertTrue(is_password_hash(password_hash))
        self.assertTrue(verify_password("safe-password-2026", password_hash))
        self.assertFalse(verify_password("wrong-password", password_hash))

    def test_request_trims_identity_fields_but_preserves_password(self) -> None:
        request = UserCreateRequest(
            user_name="  李明  ",
            user_type="  巡检员  ",
            password="  pass word  ",
        )

        self.assertEqual(request.user_name, "李明")
        self.assertEqual(request.user_type, "巡检员")
        self.assertEqual(request.password, "  pass word  ")


class UserApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_create_hashes_password_and_never_returns_it(self) -> None:
        response = UserResponse(user_id=8, user_name="李明", user_type="巡检员")
        request = UserCreateRequest(
            user_name="李明",
            user_type="巡检员",
            password="safe-password-2026",
        )

        with patch("app.api.v1.users.UserRepository.create_user", return_value=response) as create:
            result = await create_user(request)

        call = create.call_args.kwargs
        self.assertEqual(call["user_name"], "李明")
        self.assertEqual(call["user_type"], "巡检员")
        self.assertTrue(verify_password("safe-password-2026", call["user_password"]))
        self.assertNotIn("password", result.model_dump())

    async def test_update_without_password_keeps_existing_hash(self) -> None:
        response = UserResponse(user_id=8, user_name="李明", user_type="道路管理员")
        request = UserUpdateRequest(user_name="李明", user_type="道路管理员")

        with patch("app.api.v1.users.UserRepository.update_user", return_value=response) as update:
            result = await update_user(8, request)

        self.assertEqual(result.user_type, "道路管理员")
        self.assertIsNone(update.call_args.kwargs["user_password"])

    async def test_duplicate_name_returns_conflict(self) -> None:
        request = UserCreateRequest(
            user_name="李明",
            user_type="巡检员",
            password="safe-password-2026",
        )

        with (
            patch(
                "app.api.v1.users.UserRepository.create_user",
                side_effect=UserNameAlreadyExistsError("李明"),
            ),
            self.assertRaises(HTTPException) as raised,
        ):
            await create_user(request)

        self.assertEqual(raised.exception.status_code, 409)

    async def test_delete_missing_user_returns_not_found(self) -> None:
        with (
            patch("app.api.v1.users.UserRepository.delete_user", return_value=False),
            self.assertRaises(HTTPException) as raised,
        ):
            await delete_user(404)

        self.assertEqual(raised.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
