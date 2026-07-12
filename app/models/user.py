from pydantic import BaseModel
from typing import Literal


class User(BaseModel):
    user_id: int
    user_name: str
    user_password: str
    user_type: str


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
