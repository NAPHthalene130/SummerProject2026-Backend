"""Android 端移动端 API 测试:注册/登录/用户管理/上报管理(DB mock)。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from app.api.v1 import mobile
from app.models.user import (
    MobileUserLoginRequest,
    MobileUserRegisterRequest,
    MobileUserUpdateRequest,
)
from app.utils.passwords import hash_password, verify_password

# patch mobile.ensure_schema in every test since it performs schema migration
patch_ensure = patch.object(mobile, "ensure_schema")


class _FakeCursor:
    def __init__(self, fetchone=None, fetchall=None, lastrowid=1, rowcount=1):
        self._fetchone = fetchone and list(fetchone) or []
        self._fetchall = fetchall and list(fetchall) or []
        self.lastrowid = lastrowid
        self.rowcount = rowcount
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, args=None):
        self.executed.append((sql, args))

    def fetchone(self):
        return self._fetchone.pop(0) if self._fetchone else None

    def fetchall(self):
        result = self._fetchall.pop(0) if self._fetchall else []
        return result


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return self._cursor


def _user_row(user_id=1, name="张三", phone="13800000001", category="traffic_police",
              site="海淀区", password=None):
    return {
        "user_id": user_id,
        "user_name": name,
        "user_type": "交警执法",
        "user_password": password or hash_password("password-2026"),
        "phone": phone,
        "personnel_category": category,
        "site": site,
    }


class CategoriesTest(unittest.TestCase):
    def test_returns_six_categories(self) -> None:
        with patch_ensure, patch("app.api.v1.mobile.mysql_connection"):
            result = mobile.categories()

        codes = [c["code"] for c in result]
        self.assertIn("traffic_police", codes)
        self.assertIn("emergency_fire", codes)
        self.assertEqual(len(codes), 6)


class RegisterTest(unittest.TestCase):
    def test_invalid_category(self) -> None:
        request = MobileUserRegisterRequest(
            name="张三", phone="138", password="pwd-2026!", personnel_category="invalid"
        )
        with patch_ensure:
            with self.assertRaises(HTTPException) as caught:
                mobile.register(request)

        self.assertEqual(caught.exception.status_code, 400)

    def test_phone_conflict(self) -> None:
        request = MobileUserRegisterRequest(
            name="张三", phone="13800000000", password="pwd-2026!", personnel_category="traffic_police"
        )
        cursor = _FakeCursor(fetchone=[{"user_id": 1}])
        with patch_ensure, patch.object(mobile, "mysql_connection", return_value=_FakeConnection(cursor)):
            with self.assertRaises(HTTPException) as caught:
                mobile.register(request)

        self.assertEqual(caught.exception.status_code, 409)

    def test_success(self) -> None:
        request = MobileUserRegisterRequest(
            name="张三", phone="13800000001", password="pwd-2026!", personnel_category="traffic_police"
        )
        user = _user_row()
        cursor = _FakeCursor(
            fetchone=[None, user],
            lastrowid=user["user_id"],
        )
        with patch_ensure, patch.object(mobile, "mysql_connection", return_value=_FakeConnection(cursor)):
            result = mobile.register(request)

        self.assertEqual(result.user_id, 1)
        self.assertEqual(result.name, "张三")

        # password stored as argon2 hash
        insert_sql = cursor.executed[1][0]
        stored_pw = cursor.executed[1][1][1]
        self.assertTrue(stored_pw.startswith("$argon2id$"))


class LoginTest(unittest.TestCase):
    def test_unknown_phone(self) -> None:
        request = MobileUserLoginRequest(phone="13800000000", password="pwd")
        cursor = _FakeCursor(fetchone=[None])
        with patch_ensure, patch.object(mobile, "mysql_connection", return_value=_FakeConnection(cursor)):
            with self.assertRaises(HTTPException) as caught:
                mobile.login(request)

        self.assertEqual(caught.exception.status_code, 401)

    def test_wrong_password(self) -> None:
        request = MobileUserLoginRequest(phone="13800000001", password="wrong")
        cursor = _FakeCursor(fetchone=[_user_row()])
        with patch_ensure, patch.object(mobile, "mysql_connection", return_value=_FakeConnection(cursor)):
            with self.assertRaises(HTTPException) as caught:
                mobile.login(request)

        self.assertEqual(caught.exception.status_code, 401)

    def test_accepts_legacy_plaintext_password(self) -> None:
        request = MobileUserLoginRequest(phone="13800000001", password="old-plain")
        cursor = _FakeCursor(fetchone=[_user_row(password="old-plain")])
        with patch_ensure, patch.object(mobile, "mysql_connection", return_value=_FakeConnection(cursor)):
            result = mobile.login(request)

        self.assertEqual(result.name, "张三")


class UpdateMobileUserTest(unittest.TestCase):
    def test_invalid_category(self) -> None:
        request = MobileUserUpdateRequest(
            name="张三", phone="138", personnel_category="bad", site=""
        )
        with patch_ensure:
            with self.assertRaises(HTTPException) as caught:
                mobile.update_mobile_user(1, request)

        self.assertEqual(caught.exception.status_code, 400)

    def test_phone_conflict_with_other_user(self) -> None:
        request = MobileUserUpdateRequest(
            name="李四", phone="13800000002", personnel_category="traffic_police", site=""
        )
        cursor = _FakeCursor(fetchone=[None, _user_row(user_id=1)], rowcount=1)
        with patch_ensure, patch.object(mobile, "mysql_connection", return_value=_FakeConnection(cursor)):
            result = mobile.update_mobile_user(1, request)

        self.assertEqual(result.name, "张三")

    def test_user_not_found(self) -> None:
        request = MobileUserUpdateRequest(
            name="李四", phone="13800000002", personnel_category="traffic_police", site=""
        )
        cursor = _FakeCursor(fetchone=[None, None], rowcount=0)
        with patch_ensure, patch.object(mobile, "mysql_connection", return_value=_FakeConnection(cursor)):
            with self.assertRaises(HTTPException) as caught:
                mobile.update_mobile_user(1, request)

        self.assertEqual(caught.exception.status_code, 404)


class DeleteMobileUserTest(unittest.TestCase):
    def test_success(self) -> None:
        cursor = _FakeCursor()
        with patch_ensure, patch.object(mobile, "mysql_connection", return_value=_FakeConnection(cursor)):
            self.assertIsNone(mobile.delete_mobile_user(1))

    def test_not_found(self) -> None:
        cursor = _FakeCursor(rowcount=0)
        with patch_ensure, patch.object(mobile, "mysql_connection", return_value=_FakeConnection(cursor)):
            with self.assertRaises(HTTPException) as caught:
                mobile.delete_mobile_user(1)

        self.assertEqual(caught.exception.status_code, 404)


class ReportResponseTest(unittest.TestCase):
    def test_formatting(self) -> None:
        row = {
            "report_id": 1,
            "reporter_user_id": 5,
            "reporter_name": "用户A",
            "title": "测试上报",
            "location": "路口",
            "detail": "详情",
            "severity": "high",
            "event_type": "车辆碰撞",
            "image_urls": "/a.jpg,/b.jpg",
            "status": "pending",
            "created_at": "2026-07-12 10:00:00",
            "work_order_id": 0,
            "review_message": None,
            "reviewed_at": None,
        }
        response = mobile.report_response(row)

        self.assertEqual(response.report_id, 1)
        self.assertEqual(response.reporter_name, "用户A")
        self.assertEqual(response.image_urls, ["/a.jpg", "/b.jpg"])
        self.assertIsNone(response.work_order_id)


class ConvertReportTest(unittest.TestCase):
    def test_invalid_category(self) -> None:
        from app.models.mobile_report import ConvertReportRequest
        request = ConvertReportRequest(required_category="bad")
        with patch_ensure:
            with self.assertRaises(HTTPException) as caught:
                mobile.convert_report(1, request)
        self.assertEqual(caught.exception.status_code, 400)

    def test_report_not_found(self) -> None:
        from app.models.mobile_report import ConvertReportRequest
        request = ConvertReportRequest(required_category="traffic_police")
        cursor = _FakeCursor(fetchone=[None])
        with patch_ensure, patch.object(mobile, "mysql_connection", return_value=_FakeConnection(cursor)):
            with self.assertRaises(HTTPException) as caught:
                mobile.convert_report(1, request)
        self.assertEqual(caught.exception.status_code, 404)

    def test_already_processed_report(self) -> None:
        from app.models.mobile_report import ConvertReportRequest
        request = ConvertReportRequest(required_category="traffic_police")
        cursor = _FakeCursor(fetchone=[{"report_id": 1, "status": "converted"}])
        with patch_ensure, patch.object(mobile, "mysql_connection", return_value=_FakeConnection(cursor)):
            with self.assertRaises(HTTPException) as caught:
                mobile.convert_report(1, request)
        self.assertEqual(caught.exception.status_code, 409)


class RejectReportTest(unittest.TestCase):
    def test_already_processed_raises_409(self) -> None:
        from app.models.mobile_report import RejectReportRequest
        request = RejectReportRequest(review_message="重复上报")
        cursor = _FakeCursor(rowcount=0)
        with patch_ensure, patch.object(mobile, "mysql_connection", return_value=_FakeConnection(cursor)):
            with self.assertRaises(HTTPException) as caught:
                mobile.reject_report(1, request)
        self.assertEqual(caught.exception.status_code, 409)


if __name__ == "__main__":
    unittest.main()
