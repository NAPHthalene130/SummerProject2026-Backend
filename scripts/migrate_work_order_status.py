from __future__ import annotations

from init_mysql_db import (
    escape_identifier,
    load_database_config,
    load_project_config,
    migrate_work_order_status_schema,
    require_pymysql,
)


def main() -> None:
    pymysql = require_pymysql()
    database_config = load_database_config(load_project_config())
    database_name = escape_identifier(str(database_config["name"]))

    connection = pymysql.connect(
        host=database_config["host"],
        port=database_config["port"],
        user=database_config["username"],
        password=str(database_config["password"]),
        database=database_config["name"],
        charset=database_config["charset"],
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )
    try:
        with connection.cursor() as cursor:
            migrate_work_order_status_schema(cursor)
            cursor.execute(
                """
                SELECT column_name, data_type, column_default
                FROM information_schema.columns
                WHERE table_schema = DATABASE()
                  AND table_name = 'work_orders'
                  AND column_name IN ('work_order_stage', 'work_order_status', 'work_order_is_solve')
                ORDER BY ordinal_position
                """
            )
            schema = cursor.fetchall()
            cursor.execute(
                "SELECT work_order_status, COUNT(*) AS count "
                "FROM work_orders GROUP BY work_order_status ORDER BY work_order_status"
            )
            status_counts = cursor.fetchall()
        connection.commit()
        print(
            f"Migrated work-order status schema in database '{database_name}': "
            f"columns={schema}, status_counts={status_counts}"
        )
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()
