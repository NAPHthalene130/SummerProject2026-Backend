from datetime import datetime

from pydantic import BaseModel


class OrderUser(BaseModel):
    order_user_id: int
    work_order_id: int
    user_id: int
    order_user_time: datetime
    order_user_status: str
