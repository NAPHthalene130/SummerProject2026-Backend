from typing import Literal

from pydantic import BaseModel, Field, field_validator


class UserRecord(BaseModel):
    user_id: int
    user_name: str
    user_password: str
    user_type: str


class UserResponse(BaseModel):
    user_id: int
    user_name: str
    user_type: str


class UserCreateRequest(BaseModel):
    user_name: str = Field(min_length=1, max_length=255)
    user_type: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("user_name", "user_type")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("字段不能为空")
        return value.strip()


class UserUpdateRequest(BaseModel):
    user_name: str = Field(min_length=1, max_length=255)
    user_type: str = Field(min_length=1, max_length=64)
    password: str | None = Field(default=None, min_length=8, max_length=128)

    @field_validator("user_name", "user_type")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("字段不能为空")
        return value.strip()


class StaffResponse(BaseModel):
    id: str
    name: str
    role: str
    status: Literal["idle", "busy"]
    distance_km: float
    personnel_category: str = "traffic_police"


class MobileUserRegisterRequest(BaseModel):
    name: str
    phone: str
    password: str
    personnel_category: str
    site: str = ""


class MobileUserLoginRequest(BaseModel):
    phone: str
    password: str


class MobileUserResponse(BaseModel):
    user_id: int
    name: str
    phone: str
    personnel_category: str
    role_name: str
    site: str = ""
