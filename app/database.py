from contextlib import contextmanager
from typing import Any, Iterator

from app.config import database_settings

try:
    import pymysql
except ModuleNotFoundError:
    pymysql = None


def get_mysql_connection(use_database: bool = True) -> Any:
    if pymysql is None:
        raise RuntimeError("PyMySQL is required. Run `pip install PyMySQL` first.")

    connection_args: dict[str, Any] = {
        "host": database_settings.host,
        "port": database_settings.port,
        "user": database_settings.username,
        "password": database_settings.password,
        "charset": database_settings.charset,
        "cursorclass": pymysql.cursors.DictCursor,
        "connect_timeout": 5,
        "read_timeout": 15,
        "write_timeout": 15,
    }
    if use_database:
        connection_args["database"] = database_settings.name

    return pymysql.connect(**connection_args)


@contextmanager
def mysql_connection(use_database: bool = True) -> Iterator[Any]:
    connection = get_mysql_connection(use_database=use_database)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
