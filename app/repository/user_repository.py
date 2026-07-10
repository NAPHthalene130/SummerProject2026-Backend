from typing import Optional

from app.database import mysql_connection
from app.models.user import User


class UserRepository:
    @staticmethod
    def create_user(user_name: str, user_password: str, user_type: str) -> int:
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO users (user_name, user_password, user_type)
                    VALUES (%s, %s, %s)
                    """,
                    (user_name, user_password, user_type),
                )
                return cursor.lastrowid

    @staticmethod
    def get_user_by_id(user_id: int) -> Optional[User]:
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT user_id, user_name, user_password, user_type
                    FROM users
                    WHERE user_id = %s
                    """,
                    (user_id,),
                )
                row = cursor.fetchone()
                return User.model_validate(row) if row else None

    @staticmethod
    def get_user_by_name(user_name: str) -> Optional[User]:
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT user_id, user_name, user_password, user_type
                    FROM users
                    WHERE user_name = %s
                    """,
                    (user_name,),
                )
                row = cursor.fetchone()
                return User.model_validate(row) if row else None

    @staticmethod
    def list_users() -> list[User]:
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT user_id, user_name, user_password, user_type
                    FROM users
                    ORDER BY user_id
                    """
                )
                return [User.model_validate(row) for row in cursor.fetchall()]

    @staticmethod
    def update_user(user: User) -> bool:
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE users
                    SET user_name = %s,
                        user_password = %s,
                        user_type = %s
                    WHERE user_id = %s
                    """,
                    (
                        user.user_name,
                        user.user_password,
                        user.user_type,
                        user.user_id,
                    ),
                )
                return cursor.rowcount > 0

    @staticmethod
    def delete_user(user_id: int) -> bool:
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
                return cursor.rowcount > 0
