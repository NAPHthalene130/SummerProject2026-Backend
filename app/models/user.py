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
