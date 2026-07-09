from pydantic import BaseModel


class WorkOrder(BaseModel):
    work_order_id: int
    work_order_type: str
    work_order_describe: str
    work_order_img_url: str
    work_order_rank: int
    work_order_is_solve: bool
