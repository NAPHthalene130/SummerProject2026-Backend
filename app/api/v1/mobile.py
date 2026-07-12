from fastapi import APIRouter, HTTPException
from app.database import mysql_connection
from app.models.mobile_report import MobileReportCreate, MobileReportResponse, ConvertReportRequest
from app.models.user import MobileUserLoginRequest, MobileUserRegisterRequest, MobileUserResponse
from app.repository.work_order_repository import WorkOrderRepository, format_work_order_code

router = APIRouter()

CATEGORIES = {
    "traffic_police": "交警执法",
    "road_maintenance": "道路养护",
    "municipal_facilities": "市政设施",
    "vehicle_rescue": "清障救援",
    "traffic_coordination": "交通疏导",
    "emergency_fire": "应急消防",
}


def ensure_schema(cursor):
    def add_column(table, column, definition):
        cursor.execute("SELECT COUNT(*) AS count FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name=%s AND column_name=%s", (table, column))
        if int(cursor.fetchone()["count"]) == 0:
            cursor.execute(f"ALTER TABLE `{table}` ADD COLUMN `{column}` {definition}")
    add_column("users", "phone", "VARCHAR(32) NULL")
    add_column("users", "personnel_category", "VARCHAR(64) NOT NULL DEFAULT 'traffic_police'")
    add_column("users", "site", "VARCHAR(255) NULL")
    add_column("work_orders", "required_category", "VARCHAR(64) NOT NULL DEFAULT 'traffic_police'")
    legacy_groups = (
        (1, "交警执法一组", "traffic_police"),
        (2, "道路养护一组", "road_maintenance"),
        (3, "交通疏导一组", "traffic_coordination"),
        (4, "市政设施一组", "municipal_facilities"),
    )
    for user_id, group_name, category in legacy_groups:
        cursor.execute(
            "UPDATE users SET user_name=%s, user_type=%s, personnel_category=%s "
            "WHERE user_id=%s AND (phone IS NULL OR phone='')",
            (group_name, CATEGORIES[category], category, user_id),
        )
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS mobile_reports (
        report_id INT AUTO_INCREMENT PRIMARY KEY, reporter_user_id INT NOT NULL,
        title VARCHAR(255) NOT NULL, location VARCHAR(255) NOT NULL, detail TEXT NOT NULL,
        severity VARCHAR(16) NOT NULL, event_type VARCHAR(64) NOT NULL,
        image_urls TEXT NULL, status VARCHAR(16) NOT NULL DEFAULT 'pending',
        work_order_id INT NULL, created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_mobile_reports_status(status), INDEX idx_mobile_reports_user(reporter_user_id)
      ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    repaired_reports = (
        (
            1,
            "解放路北口疑似追尾",
            "解放路-人民大道北进口",
            "多模态模型识别到两车短距离碰撞，YOLO 轨迹显示后车急停，风险评分 87。",
        ),
        (
            2,
            "实验小学西门疑似违规停车",
            "文昌街实验小学西门",
            "车辆在禁停区停留超过 3 分钟，遮挡非机动车通行路径。",
        ),
    )
    for report_id, title, location, detail in repaired_reports:
        cursor.execute(
            "UPDATE mobile_reports SET title=%s, location=%s, detail=%s "
            "WHERE report_id=%s AND title LIKE '%%?%%'",
            (title, location, detail, report_id),
        )
    cursor.execute("SELECT work_order_id FROM mobile_reports WHERE report_id=1")
    repaired_order = cursor.fetchone()
    if repaired_order and repaired_order.get("work_order_id"):
        cursor.execute(
            "UPDATE work_orders SET work_order_type=%s, work_order_describe=%s, "
            "monitor_address=%s WHERE work_order_id=%s AND work_order_type LIKE '%%?%%'",
            (repaired_reports[0][1], repaired_reports[0][3], repaired_reports[0][2], repaired_order["work_order_id"]),
        )


def user_response(row):
    category = row.get("personnel_category") or "traffic_police"
    return MobileUserResponse(user_id=row["user_id"], name=row["user_name"], phone=row.get("phone") or "",
        personnel_category=category, role_name=CATEGORIES.get(category, category), site=row.get("site") or "")


@router.get("/personnel-categories")
def categories():
    return [{"code": code, "name": name} for code, name in CATEGORIES.items()]


@router.post("/mobile-users/register", response_model=MobileUserResponse)
def register(request: MobileUserRegisterRequest):
    if request.personnel_category not in CATEGORIES:
        raise HTTPException(400, "无效的人员类别")
    with mysql_connection() as connection:
        with connection.cursor() as cursor:
            ensure_schema(cursor)
            cursor.execute("SELECT user_id FROM users WHERE phone=%s", (request.phone,))
            if cursor.fetchone(): raise HTTPException(409, "手机号已注册")
            cursor.execute("INSERT INTO users(user_name,user_password,user_type,phone,personnel_category,site) VALUES(%s,%s,%s,%s,%s,%s)",
                (request.name, request.password, CATEGORIES[request.personnel_category], request.phone, request.personnel_category, request.site))
            user_id = cursor.lastrowid
            cursor.execute("SELECT * FROM users WHERE user_id=%s", (user_id,))
            return user_response(cursor.fetchone())


@router.post("/mobile-users/login", response_model=MobileUserResponse)
def login(request: MobileUserLoginRequest):
    with mysql_connection() as connection:
        with connection.cursor() as cursor:
            ensure_schema(cursor)
            cursor.execute("SELECT * FROM users WHERE phone=%s AND user_password=%s", (request.phone, request.password))
            row = cursor.fetchone()
            if not row: raise HTTPException(401, "手机号或密码错误")
            return user_response(row)


def report_response(row):
    return MobileReportResponse(report_id=row["report_id"], reporter_user_id=row["reporter_user_id"],
        reporter_name=row.get("reporter_name") or "", title=row["title"], location=row["location"], detail=row["detail"],
        severity=row["severity"], event_type=row["event_type"], image_urls=[x for x in (row.get("image_urls") or "").split(",") if x],
        status=row["status"], created_at=str(row["created_at"]),
        work_order_id=format_work_order_code(row["work_order_id"], row["created_at"]) if row.get("work_order_id") else None)


@router.post("/mobile-reports", response_model=MobileReportResponse)
def create_report(request: MobileReportCreate):
    with mysql_connection() as connection:
        with connection.cursor() as cursor:
            ensure_schema(cursor)
            cursor.execute("INSERT INTO mobile_reports(reporter_user_id,title,location,detail,severity,event_type,image_urls) VALUES(%s,%s,%s,%s,%s,%s,%s)",
                (request.reporter_user_id, request.title, request.location, request.detail, request.severity, request.event_type, ",".join(request.image_urls)))
            report_id = cursor.lastrowid
            cursor.execute("SELECT r.*,u.user_name reporter_name FROM mobile_reports r LEFT JOIN users u ON u.user_id=r.reporter_user_id WHERE report_id=%s", (report_id,))
            return report_response(cursor.fetchone())


@router.get("/mobile-reports", response_model=list[MobileReportResponse])
def list_reports(status: str | None = None, reporter_user_id: int | None = None):
    with mysql_connection() as connection:
        with connection.cursor() as cursor:
            ensure_schema(cursor)
            sql="SELECT r.*,u.user_name reporter_name FROM mobile_reports r LEFT JOIN users u ON u.user_id=r.reporter_user_id"
            conditions = []
            args: tuple = ()
            if status:
                conditions.append("r.status=%s"); args += (status,)
            if reporter_user_id is not None:
                conditions.append("r.reporter_user_id=%s"); args += (reporter_user_id,)
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            cursor.execute(sql + " ORDER BY r.created_at DESC", args)
            return [report_response(row) for row in cursor.fetchall()]


@router.post("/mobile-reports/{report_id}/convert", response_model=MobileReportResponse)
def convert_report(report_id: int, request: ConvertReportRequest):
    if request.required_category not in CATEGORIES: raise HTTPException(400, "无效的人员类别")
    with mysql_connection() as connection:
        with connection.cursor() as cursor:
            ensure_schema(cursor)
            cursor.execute("SELECT * FROM mobile_reports WHERE report_id=%s", (report_id,)); report=cursor.fetchone()
            if not report: raise HTTPException(404, "上报不存在")
            if report["status"] != "pending": raise HTTPException(409, "该上报已处理")
    order = WorkOrderRepository.create_work_order("mobile", "Android现场上报", report["title"], report["detail"],
        {"low":1,"medium":2,"high":3}[report["severity"]], report.get("image_urls") or "", report["location"], request.required_category)
    with mysql_connection() as connection:
        with connection.cursor() as cursor:
            numeric_id=int(order.work_order_id.rsplit("-",1)[-1])
            cursor.execute("UPDATE mobile_reports SET status='converted',work_order_id=%s WHERE report_id=%s", (numeric_id, report_id))
            cursor.execute("SELECT r.*,u.user_name reporter_name FROM mobile_reports r LEFT JOIN users u ON u.user_id=r.reporter_user_id WHERE report_id=%s", (report_id,))
            return report_response(cursor.fetchone())
