from datetime import datetime

from pydantic import BaseModel


class WorkOrderReply(BaseModel):
    work_order_reply_id: int
    work_order_id: int
    work_order_reply_img_url: str
    work_order_reply_msg: str
    work_order_reply_time: datetime
    work_order_reply_status: bool
