from typing import Optional

from app.database import mysql_connection
from app.models.user import UserRecord, UserResponse


class UserNameAlreadyExistsError(ValueError):
    pass


class UserRepository:
    @staticmethod
    def create_user(user_name: str, user_password: str, user_type: str, user_work_describe: Optional[str] = None) -> UserResponse:
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                try:
                    cursor.execute(
                        """
                        INSERT INTO users (user_name, user_password, user_type, user_work_describe)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (user_name, user_password, user_type, user_work_describe),
                    )
                except Exception as exc:
                    if getattr(exc, "args", (None,))[0] == 1062:
                        raise UserNameAlreadyExistsError(user_name) from exc
                    raise
                return UserResponse(
                    user_id=cursor.lastrowid,
                    user_name=user_name,
                    user_type=user_type,
                    user_work_describe=user_work_describe,
                )

    @staticmethod
    def get_user_by_id(user_id: int) -> Optional[UserRecord]:
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT user_id, user_name, user_password, user_type, user_work_describe
                    FROM users
                    WHERE user_id = %s
                    """,
                    (user_id,),
                )
                row = cursor.fetchone()
                return UserRecord.model_validate(row) if row else None

    @staticmethod
    def get_user_by_name(user_name: str) -> Optional[UserRecord]:
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT user_id, user_name, user_password, user_type, user_work_describe
                    FROM users
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
                cursor.execute(
                    """
                    SELECT user_id, user_name, user_type, user_work_describe
                    FROM users
                    ORDER BY user_id
                    """
                )
                return [UserResponse.model_validate(row) for row in cursor.fetchall()]

    @staticmethod
    def update_user(
        user_id: int,
        user_name: str,
        user_type: str,
        user_password: Optional[str] = None,
        user_work_describe: Optional[str] = None,
    ) -> Optional["UserResponse"]:
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                fields = ["user_name = %s", "user_type = %s"]
                values: list[object] = [user_name, user_type]
                if user_password is not None:
                    fields.append("user_password = %s")
                    values.append(user_password)
                if user_work_describe is not None:
                    fields.append("user_work_describe = %s")
                    values.append(user_work_describe)
                values.append(user_id)

                try:
                    cursor.execute(
                        f"UPDATE users SET {', '.join(fields)} WHERE user_id = %s",
                        tuple(values),
                    )
                except Exception as exc:
                    if getattr(exc, "args", (None,))[0] == 1062:
                        raise UserNameAlreadyExistsError(user_name) from exc
                    raise

                cursor.execute(
                    "SELECT user_id, user_name, user_type, user_work_describe FROM users WHERE user_id = %s",
                    (user_id,),
                )
                row = cursor.fetchone()
                return UserResponse.model_validate(row) if row else None

    @staticmethod
    def delete_user(user_id: int) -> bool:
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
                return cursor.rowcount > 0
