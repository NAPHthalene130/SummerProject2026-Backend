from typing import Literal, Optional
from pydantic import BaseModel


ReportStatus = Literal["pending", "converted", "rejected"]


class MobileReportCreate(BaseModel):
    reporter_user_id: int
    title: str
    location: str
    detail: str
    severity: Literal["low", "medium", "high"] = "medium"
    event_type: str = "other"
    image_urls: list[str] = []


class MobileReportResponse(MobileReportCreate):
    report_id: int
    reporter_name: str
    status: ReportStatus
    created_at: str
    work_order_id: Optional[str] = None


class ConvertReportRequest(BaseModel):
    required_category: str
