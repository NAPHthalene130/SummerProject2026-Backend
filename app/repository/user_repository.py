from typing import Optional

from app.database import mysql_connection
from app.models.user import UserRecord, UserResponse


class UserNameAlreadyExistsError(ValueError):
    pass


class UserRepository:
    @staticmethod
    def ensure_schema(cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS admin_users (
                admin_user_id INT NOT NULL AUTO_INCREMENT,
                user_name VARCHAR(255) NOT NULL,
                user_password VARCHAR(255) NOT NULL,
                user_type VARCHAR(64) NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY (admin_user_id),
                UNIQUE KEY uk_admin_users_user_name (user_name),
                INDEX idx_admin_users_user_type (user_type)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )

    @staticmethod
    def create_user(user_name: str, user_password: str, user_type: str) -> UserResponse:
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                UserRepository.ensure_schema(cursor)
                try:
                    cursor.execute(
                        """
                        INSERT INTO admin_users (user_name, user_password, user_type)
                        VALUES (%s, %s, %s)
                        """,
                        (user_name, user_password, user_type),
                    )
                except Exception as exc:
                    if getattr(exc, "args", (None,))[0] == 1062:
                        raise UserNameAlreadyExistsError(user_name) from exc
                    raise
                return UserResponse(
                    user_id=cursor.lastrowid,
                    user_name=user_name,
                    user_type=user_type,
                )

    @staticmethod
    def get_user_by_id(user_id: int) -> Optional[UserRecord]:
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                UserRepository.ensure_schema(cursor)
                cursor.execute(
                    """
                    SELECT admin_user_id AS user_id, user_name, user_password, user_type
                    FROM admin_users
                    WHERE admin_user_id = %s
                    """,
                    (user_id,),
                )
                row = cursor.fetchone()
                return UserRecord.model_validate(row) if row else None

    @staticmethod
    def get_user_by_name(user_name: str) -> Optional[UserRecord]:
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                UserRepository.ensure_schema(cursor)
                cursor.execute(
                    """
                    SELECT admin_user_id AS user_id, user_name, user_password, user_type
                    FROM admin_users
                    WHERE user_name = %s
                    """,
                    (user_name,),
                )
                row = cursor.fetchone()
                return UserRecord.model_validate(row) if row else None

    @staticmethod
    def list_users() -> list[UserResponse]:
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                UserRepository.ensure_schema(cursor)
                cursor.execute(
                    """
                    SELECT admin_user_id AS user_id, user_name, user_type
                    FROM admin_users
                    ORDER BY admin_user_id
                    """
                )
                return [UserResponse.model_validate(row) for row in cursor.fetchall()]

    @staticmethod
    def update_user(
        user_id: int,
        user_name: str,
        user_type: str,
        user_password: str | None = None,
    ) -> UserResponse | None:
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                UserRepository.ensure_schema(cursor)
                fields = ["user_name = %s", "user_type = %s"]
                values: list[object] = [user_name, user_type]
                if user_password is not None:
                    fields.append("user_password = %s")
                    values.append(user_password)
                values.append(user_id)

                try:
                    cursor.execute(
                        f"UPDATE admin_users SET {', '.join(fields)} WHERE admin_user_id = %s",
                        tuple(values),
                    )
                except Exception as exc:
                    if getattr(exc, "args", (None,))[0] == 1062:
                        raise UserNameAlreadyExistsError(user_name) from exc
                    raise

                cursor.execute(
                    "SELECT admin_user_id AS user_id, user_name, user_type FROM admin_users WHERE admin_user_id = %s",
                    (user_id,),
                )
                row = cursor.fetchone()
                return UserResponse.model_validate(row) if row else None

    @staticmethod
    def delete_user(user_id: int) -> bool:
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                UserRepository.ensure_schema(cursor)
                cursor.execute("DELETE FROM admin_users WHERE admin_user_id = %s", (user_id,))
                return cursor.rowcount > 0
