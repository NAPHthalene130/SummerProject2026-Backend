from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

from app.utils.passwords import hash_password, is_password_hash


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.yaml"
SQL_PATH = Path(__file__).with_name("init_mysql.sql")

DEFAULT_DATABASE_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "username": "root",
    "password": "123456",
    "name": "summer_project_2026",
    "charset": "utf8mb4",
}

UPSERT_CAMERA_SQL = """
INSERT INTO cameras (id, name, url, longitude, latitude)
VALUES (%s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
  name = VALUES(name),
  url = VALUES(url),
  longitude = VALUES(longitude),
  latitude = VALUES(latitude)
"""

INSERT_CAMERA_STATS_SQL = """
INSERT IGNORE INTO camera_stats (camera_id, total_vehicle_count)
VALUES (%s, 0)
"""

REQUIRED_COLUMNS = (
    ("work_orders", "event_id", "ADD COLUMN `event_id` VARCHAR(64) NULL AFTER `work_order_id`"),
    ("work_orders", "camera_id", "ADD COLUMN `camera_id` VARCHAR(64) NULL AFTER `event_id`"),
    ("work_orders", "camera_name", "ADD COLUMN `camera_name` VARCHAR(255) NULL AFTER `camera_id`"),
    ("work_orders", "segment_id", "ADD COLUMN `segment_id` VARCHAR(64) NULL AFTER `camera_name`"),
    ("work_orders", "segment_name", "ADD COLUMN `segment_name` VARCHAR(255) NULL AFTER `segment_id`"),
    (
        "work_orders",
        "monitor_address",
        "ADD COLUMN `monitor_address` VARCHAR(255) NULL AFTER `segment_name`",
    ),
    (
        "work_orders",
        "work_order_time",
        "ADD COLUMN `work_order_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP "
        "AFTER `work_order_rank`",
    ),
    (
        "work_orders",
        "work_order_stage",
        "ADD COLUMN `work_order_stage` VARCHAR(64) NOT NULL DEFAULT 'unassigned' "
        "AFTER `work_order_time`",
    ),
    (
        "work_orders",
        "work_order_status",
        "ADD COLUMN `work_order_status` INT NOT NULL DEFAULT 0 AFTER `work_order_stage`",
    ),
    ("work_orders", "ai_suggestion", "ADD COLUMN `ai_suggestion` TEXT NULL AFTER `work_order_status`"),
    ("work_orders", "scene_info", "ADD COLUMN `scene_info` TEXT NULL AFTER `ai_suggestion`"),
    ("work_orders", "completed_at", "ADD COLUMN `completed_at` DATETIME NULL AFTER `scene_info`"),
    (
        "work_order_replies",
        "work_order_reply_time",
        "ADD COLUMN `work_order_reply_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP "
        "AFTER `work_order_reply_msg`",
    ),
)

SEED_USERS = (
    (1, "人员A", "123456", "道路管理员"),
    (2, "人员B", "123456", "巡检员"),
    (3, "人员C", "123456", "巡检员"),
    (4, "人员D", "123456", "应急处置员"),
)

SEED_WORK_ORDERS = (
    {
        "work_order_id": 1,
        "event_id": "evt_front_001",
        "camera_id": "cam_006",
        "camera_name": "高新一路事故高发点",
        "segment_id": "seg_gaoxin_1",
        "segment_name": "高新一路事故高发段",
        "monitor_address": "科技园区东侧高新一路与创新路交会区域",
        "work_order_type": "疑似追尾事故，占用主干路右侧车道",
        "work_order_describe": "AI 检测到车辆异常停滞，后方车流排队长度快速增加。",
        "work_order_img_url": "https://placehold.co/640x360/172033/f4f8ff?text=Gaoxin+Accident",
        "work_order_rank": 3,
        "work_order_time": "2026-07-09 09:15:00",
        "work_order_stage": "unassigned",
        "work_order_status": 0,
        "ai_suggestion": "建议优先确认现场人员安全，临时封控右侧车道，并联动交警与清障车辆。",
        "scene_info": "画面显示 2 辆小客车低速接触，后方车辆出现连续变道。",
        "completed_at": None,
    },
    {
        "work_order_id": 2,
        "event_id": "evt_front_002",
        "camera_id": "cam_003",
        "camera_name": "创新路西段摄像头",
        "segment_id": "seg_innov_e",
        "segment_name": "创新路施工拥堵段",
        "monitor_address": "创新路东段施工围挡区域",
        "work_order_type": "施工占道导致拥堵",
        "work_order_describe": "施工围挡附近车辆排队明显，通行效率下降。",
        "work_order_img_url": "https://placehold.co/640x360/2c2f39/f4f8ff?text=Construction+Queue",
        "work_order_rank": 2,
        "work_order_time": "2026-07-09 09:22:00",
        "work_order_stage": "pending",
        "work_order_status": 0,
        "ai_suggestion": "建议核查施工占道范围，补充临时警示牌，调整高峰绕行提示。",
        "scene_info": "施工区外侧车道通行能力下降，排队影响创新路东向西车流。",
        "completed_at": None,
    },
    {
        "work_order_id": 3,
        "event_id": "evt_front_003",
        "camera_id": "cam_004",
        "camera_name": "学校入口摄像头",
        "segment_id": "seg_academy_w",
        "segment_name": "学院路园区入口西段",
        "monitor_address": "学院路学校/园区入口",
        "work_order_type": "早高峰入口拥堵",
        "work_order_describe": "园区入口车辆集中进入，短时拥堵并影响学院路通行。",
        "work_order_img_url": "https://placehold.co/640x360/1b3044/f4f8ff?text=Campus+Entrance",
        "work_order_rank": 2,
        "work_order_time": "2026-07-09 09:30:00",
        "work_order_stage": "processing",
        "work_order_status": 0,
        "ai_suggestion": "建议安排现场疏导，开放临停区，并优化入口排队动线。",
        "scene_info": "入口等待车辆约 18 辆，非机动车与机动车有短时交织。",
        "completed_at": None,
    },
    {
        "work_order_id": 4,
        "event_id": "evt_front_004",
        "camera_id": "cam_002",
        "camera_name": "核心十字路口鹰眼",
        "segment_id": "seg_tech_e",
        "segment_name": "科技大道核心路口东段",
        "monitor_address": "科技大道核心十字路口",
        "work_order_type": "短时违停影响右转车道",
        "work_order_describe": "核心路口附近检测到短时违停，影响右转车辆通行。",
        "work_order_img_url": "https://placehold.co/640x360/14263a/f4f8ff?text=Illegal+Parking",
        "work_order_rank": 1,
        "work_order_time": "2026-07-09 08:40:00",
        "work_order_stage": "completed",
        "work_order_status": 1,
        "ai_suggestion": "建议巡检提醒驶离，并纳入重点观察点。",
        "scene_info": "违停车辆停靠约 4 分钟，未造成持续拥堵。",
        "completed_at": "2026-07-09 08:58:00",
    },
    {
        "work_order_id": 5,
        "event_id": "evt_front_005",
        "camera_id": "cam_001",
        "camera_name": "科技大道西段卡口",
        "segment_id": "seg_data_s",
        "segment_name": "数据南路南段",
        "monitor_address": "数据南路南段市政井盖点位",
        "work_order_type": "井盖疑似偏移",
        "work_order_describe": "巡检图像提示井盖疑似偏移，需现场复核。",
        "work_order_img_url": "https://placehold.co/640x360/253142/f4f8ff?text=Manhole+Check",
        "work_order_rank": 1,
        "work_order_time": "2026-07-09 08:10:00",
        "work_order_stage": "completed",
        "work_order_status": 1,
        "ai_suggestion": "建议复核井盖状态，如存在松动及时安排市政维修。",
        "scene_info": "疑似偏移区域位于慢行车道边缘。",
        "completed_at": "2026-07-09 08:35:00",
    },
    {
        "work_order_id": 6,
        "event_id": "evt_front_006",
        "camera_id": "cam_005",
        "camera_name": "云计算路北段摄像头",
        "segment_id": "seg_cloud_n",
        "segment_name": "云计算路核心北段",
        "monitor_address": "云计算路北段公交港湾附近",
        "work_order_type": "AI 误报行人闯入机动车道",
        "work_order_describe": "模型将路侧工作人员误判为行人闯入机动车道。",
        "work_order_img_url": "https://placehold.co/640x360/202b3b/f4f8ff?text=False+Alarm",
        "work_order_rank": 1,
        "work_order_time": "2026-07-09 07:52:00",
        "work_order_stage": "ignored",
        "work_order_status": 2,
        "ai_suggestion": "建议将该样本加入误报样本库，优化施工人员识别标签。",
        "scene_info": "现场为路侧保洁人员在隔离区域作业。",
        "completed_at": "2026-07-09 08:05:00",
    },
)

SEED_ORDER_USERS = (
    (1, 2, 2, "2026-07-09 09:24:00", "pending"),
    (2, 3, 3, "2026-07-09 09:31:00", "processing"),
    (3, 4, 1, "2026-07-09 08:45:00", "completed"),
    (4, 5, 4, "2026-07-09 08:12:00", "completed"),
    (5, 6, 1, "2026-07-09 07:54:00", "ignored"),
)

SEED_WORK_ORDER_REPLIES = (
    (
        1,
        4,
        "https://placehold.co/640x360/123826/f4f8ff?text=Feedback+Resolved",
        "现场已完成劝离，路口通行恢复正常。",
        "2026-07-09 08:58:00",
        1,
    ),
    (
        2,
        5,
        "https://placehold.co/640x360/263d32/f4f8ff?text=Feedback+Checked",
        "现场复核为误报，井盖状态正常。",
        "2026-07-09 08:35:00",
        1,
    ),
    (
        3,
        6,
        "https://placehold.co/640x360/343434/f4f8ff?text=False+Alarm+Closed",
        "确认误报，已关闭并加入样本复盘。",
        "2026-07-09 08:05:00",
        1,
    ),
)


def require_pymysql() -> Any:
    try:
        import pymysql
    except ModuleNotFoundError:
        print("PyMySQL is not installed. Run `pip install PyMySQL` first.", file=sys.stderr)
        raise SystemExit(2)
    return pymysql


def load_project_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"{CONFIG_PATH} must contain a YAML mapping")

    return data


def load_database_config(config: dict[str, Any]) -> dict[str, Any]:
    database_config = dict(DEFAULT_DATABASE_CONFIG)
    database_config.update(config.get("database") or {})
    database_config["port"] = int(database_config["port"])
    return database_config


def escape_identifier(identifier: str) -> str:
    return identifier.replace("`", "``")


def split_sql(sql: str) -> list[str]:
    lines = []
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") or not stripped:
            continue
        lines.append(line)

    return [statement.strip() for statement in "\n".join(lines).split(";") if statement.strip()]


def column_exists(cursor: Any, table_name: str, column_name: str) -> bool:
    cursor.execute(
        """
        SELECT COUNT(*) AS count
        FROM information_schema.columns
        WHERE table_schema = DATABASE()
          AND table_name = %s
          AND column_name = %s
        """,
        (table_name, column_name),
    )
    row = cursor.fetchone()
    count = row["count"] if isinstance(row, dict) else row[0]
    return count > 0


def column_data_type(cursor: Any, table_name: str, column_name: str) -> str | None:
    cursor.execute(
        """
        SELECT data_type
        FROM information_schema.columns
        WHERE table_schema = DATABASE()
          AND table_name = %s
          AND column_name = %s
        """,
        (table_name, column_name),
    )
    row = cursor.fetchone()
    if not row:
        return None
    if isinstance(row, dict):
        value = next(
            (value for key, value in row.items() if str(key).lower() == "data_type"),
            None,
        )
    else:
        value = row[0]
    return str(value).lower() if value is not None else None


def index_columns(cursor: Any, table_name: str, index_name: str) -> list[str]:
    cursor.execute(
        """
        SELECT column_name
        FROM information_schema.statistics
        WHERE table_schema = DATABASE()
          AND table_name = %s
          AND index_name = %s
        ORDER BY seq_in_index
        """,
        (table_name, index_name),
    )
    rows = cursor.fetchall()
    columns: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            value = next(
                (value for key, value in row.items() if str(key).lower() == "column_name"),
                None,
            )
        else:
            value = row[0]
        if value is not None:
            columns.append(str(value))
    return columns


def ensure_single_column_index(cursor: Any, table_name: str, index_name: str, column_name: str) -> None:
    existing_columns = index_columns(cursor, table_name, index_name)
    if existing_columns == [column_name]:
        return
    if existing_columns:
        cursor.execute(f"ALTER TABLE `{table_name}` DROP INDEX `{index_name}`")
    cursor.execute(f"ALTER TABLE `{table_name}` ADD INDEX `{index_name}` (`{column_name}`)")


def migrate_work_order_status_schema(cursor: Any) -> None:
    """Migrate legacy boolean/string status columns to status-code + stage."""
    integer_types = {"tinyint", "smallint", "mediumint", "int", "integer", "bigint"}
    status_type = column_data_type(cursor, "work_orders", "work_order_status")

    if status_type is not None and status_type not in integer_types:
        if column_exists(cursor, "work_orders", "work_order_stage"):
            cursor.execute(
                "UPDATE work_orders SET work_order_stage = work_order_status "
                "WHERE work_order_status IS NOT NULL"
            )
            cursor.execute("ALTER TABLE work_orders DROP COLUMN work_order_status")
        else:
            cursor.execute(
                "ALTER TABLE work_orders CHANGE COLUMN work_order_status work_order_stage "
                "VARCHAR(64) NOT NULL DEFAULT 'unassigned'"
            )

    if not column_exists(cursor, "work_orders", "work_order_stage"):
        cursor.execute(
            "ALTER TABLE work_orders ADD COLUMN work_order_stage "
            "VARCHAR(64) NOT NULL DEFAULT 'unassigned' AFTER work_order_time"
        )

    if not column_exists(cursor, "work_orders", "work_order_status"):
        if column_exists(cursor, "work_orders", "work_order_is_solve"):
            cursor.execute(
                "ALTER TABLE work_orders CHANGE COLUMN work_order_is_solve work_order_status "
                "INT NOT NULL DEFAULT 0"
            )
        else:
            cursor.execute(
                "ALTER TABLE work_orders ADD COLUMN work_order_status "
                "INT NOT NULL DEFAULT 0 AFTER work_order_stage"
            )

    cursor.execute(
        """
        UPDATE work_orders
        SET work_order_status = CASE
          WHEN work_order_stage IN ('ignored', 'false_alarm') THEN 2
          WHEN work_order_stage = 'completed' THEN 1
          ELSE 0
        END,
        work_order_stage = CASE
          WHEN work_order_stage = 'false_alarm' THEN 'ignored'
          ELSE work_order_stage
        END
        """
    )

    if column_exists(cursor, "work_orders", "work_order_is_solve"):
        old_index_columns = index_columns(cursor, "work_orders", "idx_work_orders_is_solve")
        if old_index_columns:
            cursor.execute("ALTER TABLE work_orders DROP INDEX idx_work_orders_is_solve")
        cursor.execute("ALTER TABLE work_orders DROP COLUMN work_order_is_solve")

    cursor.execute(
        "ALTER TABLE work_orders MODIFY COLUMN work_order_stage "
        "VARCHAR(64) NOT NULL DEFAULT 'unassigned'"
    )
    cursor.execute(
        "ALTER TABLE work_orders MODIFY COLUMN work_order_status "
        "INT NOT NULL DEFAULT 0 COMMENT '0=unresolved, 1=resolved, 2=ignored'"
    )
    ensure_single_column_index(cursor, "work_orders", "idx_work_orders_status", "work_order_status")
    ensure_single_column_index(cursor, "work_orders", "idx_work_orders_stage", "work_order_stage")


def ensure_required_columns(cursor: Any) -> None:
    for table_name, column_name, alter_clause in REQUIRED_COLUMNS:
        if not column_exists(cursor, table_name, column_name):
            cursor.execute(f"ALTER TABLE `{table_name}` {alter_clause}")


def seed_users(cursor: Any) -> int:
    for user in SEED_USERS:
        user_id, user_name, password, user_type = user
        cursor.execute(
            """
            INSERT INTO users (user_id, user_name, user_password, user_type)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
              user_name = VALUES(user_name),
              user_type = VALUES(user_type)
            """,
            (user_id, user_name, hash_password(password), user_type),
        )
    return len(SEED_USERS)


def migrate_legacy_user_passwords(cursor: Any) -> int:
    cursor.execute("SELECT user_id, user_password FROM users")
    legacy_passwords: list[tuple[int, str]] = []
    for row in cursor.fetchall():
        if isinstance(row, dict):
            user_id, password = row["user_id"], row["user_password"]
        else:
            user_id, password = row[0], row[1]
        if not is_password_hash(str(password)):
            legacy_passwords.append((int(user_id), str(password)))
    for user_id, password in legacy_passwords:
        cursor.execute(
            "UPDATE users SET user_password = %s WHERE user_id = %s",
            (hash_password(str(password)), user_id),
        )
    return len(legacy_passwords)


def seed_work_orders(cursor: Any) -> int:
    for order in SEED_WORK_ORDERS:
        cursor.execute(
            """
            INSERT INTO work_orders (
              work_order_id,
              event_id,
              camera_id,
              camera_name,
              segment_id,
              segment_name,
              monitor_address,
              work_order_type,
              work_order_describe,
              work_order_img_url,
              work_order_rank,
              work_order_time,
              work_order_stage,
              work_order_status,
              ai_suggestion,
              scene_info,
              completed_at
            )
            VALUES (
              %(work_order_id)s,
              %(event_id)s,
              %(camera_id)s,
              %(camera_name)s,
              %(segment_id)s,
              %(segment_name)s,
              %(monitor_address)s,
              %(work_order_type)s,
              %(work_order_describe)s,
              %(work_order_img_url)s,
              %(work_order_rank)s,
              %(work_order_time)s,
              %(work_order_stage)s,
              %(work_order_status)s,
              %(ai_suggestion)s,
              %(scene_info)s,
              %(completed_at)s
            )
            ON DUPLICATE KEY UPDATE
              event_id = VALUES(event_id),
              camera_id = VALUES(camera_id),
              camera_name = VALUES(camera_name),
              segment_id = VALUES(segment_id),
              segment_name = VALUES(segment_name),
              monitor_address = VALUES(monitor_address),
              work_order_type = VALUES(work_order_type),
              work_order_describe = VALUES(work_order_describe),
              work_order_img_url = VALUES(work_order_img_url),
              work_order_rank = VALUES(work_order_rank),
              work_order_time = VALUES(work_order_time),
              work_order_stage = VALUES(work_order_stage),
              work_order_status = VALUES(work_order_status),
              ai_suggestion = VALUES(ai_suggestion),
              scene_info = VALUES(scene_info),
              completed_at = VALUES(completed_at)
            """,
            order,
        )
    return len(SEED_WORK_ORDERS)


def seed_order_users(cursor: Any) -> int:
    for order_user in SEED_ORDER_USERS:
        cursor.execute(
            """
            INSERT INTO order_user (
              order_user_id,
              work_order_id,
              user_id,
              order_user_time,
              order_user_status
            )
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
              work_order_id = VALUES(work_order_id),
              user_id = VALUES(user_id),
              order_user_time = VALUES(order_user_time),
              order_user_status = VALUES(order_user_status)
            """,
            order_user,
        )
    return len(SEED_ORDER_USERS)


def seed_work_order_replies(cursor: Any) -> int:
    for reply in SEED_WORK_ORDER_REPLIES:
        cursor.execute(
            """
            INSERT INTO work_order_replies (
              work_order_reply_id,
              work_order_id,
              work_order_reply_img_url,
              work_order_reply_msg,
              work_order_reply_time,
              work_order_reply_status
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
              work_order_id = VALUES(work_order_id),
              work_order_reply_img_url = VALUES(work_order_reply_img_url),
              work_order_reply_msg = VALUES(work_order_reply_msg),
              work_order_reply_time = VALUES(work_order_reply_time),
              work_order_reply_status = VALUES(work_order_reply_status)
            """,
            reply,
        )
    return len(SEED_WORK_ORDER_REPLIES)


def sync_cameras(cursor: Any, cameras: list[dict[str, Any]]) -> int:
    synced = 0
    for camera in cameras:
        cursor.execute(
            UPSERT_CAMERA_SQL,
            (
                camera["id"],
                camera["name"],
                camera["url"],
                float(camera["longitude"]),
                float(camera["latitude"]),
            ),
        )
        cursor.execute(INSERT_CAMERA_STATS_SQL, (camera["id"],))
        synced += 1
    return synced


def main() -> None:
    pymysql = require_pymysql()
    project_config = load_project_config()
    database_config = load_database_config(project_config)
    database_name = escape_identifier(str(database_config["name"]))

    schema_sql = SQL_PATH.read_text(encoding="utf-8")
    schema_sql = schema_sql.replace("`summer_project_2026`", f"`{database_name}`")

    connection = pymysql.connect(
        host=database_config["host"],
        port=database_config["port"],
        user=database_config["username"],
        password=str(database_config["password"]),
        charset=database_config["charset"],
        autocommit=True,
    )
    try:
        with connection.cursor() as cursor:
            for statement in split_sql(schema_sql):
                cursor.execute(statement)

            migrate_work_order_status_schema(cursor)
            ensure_required_columns(cursor)
            user_count = seed_users(cursor)
            migrated_password_count = migrate_legacy_user_passwords(cursor)
            work_order_count = seed_work_orders(cursor)
            order_user_count = seed_order_users(cursor)
            reply_count = seed_work_order_replies(cursor)
            camera_count = sync_cameras(cursor, project_config.get("cameras") or [])

        print(
            f"Initialized MySQL database '{database_config['name']}' "
            f"and synced {camera_count} cameras, {user_count} users, "
            f"{migrated_password_count} legacy passwords, "
            f"{work_order_count} work orders, {order_user_count} dispatches, "
            f"{reply_count} replies."
        )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
