"""移动端 API 扩展测试:列表/上报/转工单路径补充。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.api.v1 import mobile
from app.models.mobile_report import ConvertReportRequest, MobileReportCreate, RejectReportRequest


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
        return self._fetchall.pop(0) if self._fetchall else []


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return self._cursor


def _user_row(user_id=1, name="张三", phone="13800138001", category="traffic_police"):
    return {
        "user_id": user_id, "user_name": name, "user_type": "交警执法",
        "user_password": "$argon2id$hash", "phone": phone,
        "personnel_category": category, "site": "海淀区",
    }


def _report_row(report_id=1, status="pending", image_urls="", severity="high",
                work_order_id=None):
    return {
        "report_id": report_id, "reporter_user_id": 5, "reporter_name": "用户A",
        "title": "测试上报", "location": "路口", "detail": "详情",
        "severity": severity, "event_type": "车辆碰撞",
        "image_urls": image_urls,
        "status": status, "created_at": "2026-07-12 10:00:00",
        "work_order_id": work_order_id,
        "review_message": None, "reviewed_at": None,
    }


class ListUsersTest(unittest.TestCase):
    def test_list_returns_users(self) -> None:
        row = _user_row()
        cursor = _FakeCursor(fetchall=[[row]])
        with (
            patch.object(mobile, "ensure_schema"),
            patch.object(mobile, "mysql_connection", return_value=_FakeConnection(cursor)),
        ):
            result = mobile.list_mobile_users()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "张三")


class CreateReportTest(unittest.TestCase):
    def test_success(self) -> None:
        request = MobileReportCreate(
            reporter_user_id=5, title="上报标题", location="海淀路",
            detail="事故详情", severity="high", event_type="车辆碰撞",
            image_urls=["/img/1.jpg", "/img/2.jpg"],
        )
        row = _report_row(image_urls="/img/1.jpg,/img/2.jpg")
        cursor = _FakeCursor(fetchone=[row], lastrowid=1)
        with (
            patch.object(mobile, "ensure_schema"),
            patch.object(mobile, "mysql_connection", return_value=_FakeConnection(cursor)),
        ):
            result = mobile.create_report(request)

        self.assertEqual(result.report_id, 1)
        self.assertEqual(result.title, "测试上报")
        self.assertEqual(result.image_urls, ["/img/1.jpg", "/img/2.jpg"])


class ListReportsTest(unittest.TestCase):
    def test_list_all(self) -> None:
        cursor = _FakeCursor(fetchall=[[_report_row()]])
        with (
            patch.object(mobile, "ensure_schema"),
            patch.object(mobile, "mysql_connection", return_value=_FakeConnection(cursor)),
        ):
            result = mobile.list_reports()

        self.assertEqual(len(result), 1)

    def test_list_with_filters(self) -> None:
        cursor = _FakeCursor(fetchall=[[]])
        with (
            patch.object(mobile, "ensure_schema"),
            patch.object(mobile, "mysql_connection", return_value=_FakeConnection(cursor)),
        ):
            result = mobile.list_reports(status="pending", reporter_user_id=5)

        self.assertEqual(result, [])


class RejectReportTest(unittest.TestCase):
    def test_success(self) -> None:
        request = RejectReportRequest(review_message="内容不实")
        pending_row = _report_row(status="pending")
        rejected_row = _report_row(status="rejected")
        cursor = _FakeCursor(fetchone=[rejected_row], rowcount=1)
        with (
            patch.object(mobile, "ensure_schema"),
            patch.object(mobile, "mysql_connection", return_value=_FakeConnection(cursor)),
        ):
            result = mobile.reject_report(1, request)

        self.assertEqual(result.status, "rejected")
        # UPDATE sets review_message and reviewed_at
        update_sql = cursor.executed[0][0]
        self.assertIn("review_message", update_sql)


class ConvertReportSuccessTest(unittest.TestCase):
    def test_creates_work_order_and_updates_report(self) -> None:
        request = ConvertReportRequest(required_category="traffic_police")
        pending_row = {"report_id": 1, "status": "pending", "work_order_id": None,
                       "title": "上报", "detail": "详情", "severity": "high",
                       "image_urls": "/a.jpg", "location": "路口"}

        order = SimpleNamespace(work_order_id="WO-20260712-099")

        conn1 = _FakeConnection(_FakeCursor(fetchone=[pending_row]))
        conn2 = _FakeConnection(_FakeCursor(fetchone=[_report_row(work_order_id=99, status="converted")]))
        conns = iter([conn1, conn2])

        with (
            patch.object(mobile, "ensure_schema"),
            patch.object(mobile, "mysql_connection", side_effect=lambda **kw: next(conns)),
            patch(
                "app.api.v1.mobile.WorkOrderRepository.create_work_order",
                return_value=order,
            ) as create_order,
        ):
            result = mobile.convert_report(1, request)

        create_order.assert_called_once()
        self.assertEqual(result.status, "converted")
        self.assertEqual(result.work_order_id, "WO-20260709-099")


if __name__ == "__main__":
    unittest.main()
