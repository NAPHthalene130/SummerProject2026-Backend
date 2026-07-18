"""数据库连接工厂测试:mocked PyMySQL 验证连接参数与事务控制。"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app import database as db_module


class GetMySqlConnectionTest(unittest.TestCase):
    def test_raises_when_pymysql_is_missing(self) -> None:
        with patch.object(db_module, "pymysql", None):
            with self.assertRaises(RuntimeError) as caught:
                db_module.get_mysql_connection()

            self.assertIn("PyMySQL is required", str(caught.exception))

    def test_connection_args_include_database_when_use_database_is_true(self) -> None:
        mock_pymysql = MagicMock()
        mock_pymysql.cursors = SimpleNamespace(DictCursor=object)

        with (
            patch.object(db_module, "pymysql", mock_pymysql),
            patch.object(db_module, "database_settings") as ds,
        ):
            ds.host = "127.0.0.1"
            ds.port = 3306
            ds.username = "root"
            ds.password = "pw"
            ds.charset = "utf8mb4"
            ds.name = "mydb"

            db_module.get_mysql_connection(use_database=True)
            kwargs = mock_pymysql.connect.call_args.kwargs

            self.assertEqual(kwargs["host"], "127.0.0.1")
            self.assertEqual(kwargs["port"], 3306)
            self.assertEqual(kwargs["database"], "mydb")
            self.assertEqual(kwargs["connect_timeout"], 5)

    def test_connection_args_exclude_database_when_use_database_is_false(self) -> None:
        mock_pymysql = MagicMock()
        mock_pymysql.cursors = SimpleNamespace(DictCursor=object)

        with (
            patch.object(db_module, "pymysql", mock_pymysql),
            patch.object(db_module, "database_settings") as ds,
        ):
            ds.host = "127.0.0.1"; ds.port = 3306; ds.username = "u"; ds.password = "p"
            ds.charset = "utf8mb4"; ds.name = "mydb"

            db_module.get_mysql_connection(use_database=False)
            kwargs = mock_pymysql.connect.call_args.kwargs

            self.assertNotIn("database", kwargs)


class MysqlConnectionContextManagerTest(unittest.TestCase):
    def test_commits_on_success_and_closes_connection(self) -> None:
        connection = MagicMock()
        with patch.object(db_module, "get_mysql_connection", return_value=connection):
            with db_module.mysql_connection():
                pass

        connection.commit.assert_called_once()
        connection.close.assert_called_once()
        connection.rollback.assert_not_called()

    def test_rolls_back_on_exception_and_re_raises(self) -> None:
        connection = MagicMock()
        with patch.object(db_module, "get_mysql_connection", return_value=connection):
            with self.assertRaises(ValueError):
                with db_module.mysql_connection():
                    raise ValueError("oops")

        connection.rollback.assert_called_once()
        connection.close.assert_called_once()

    def test_closes_connection_even_when_commit_fails(self) -> None:
        connection = MagicMock()
        connection.commit.side_effect = RuntimeError("commit failed")
        with patch.object(db_module, "get_mysql_connection", return_value=connection):
            with self.assertRaises(RuntimeError):
                with db_module.mysql_connection():
                    pass

        connection.rollback.assert_called_once()
        connection.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
