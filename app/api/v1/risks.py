from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.modules.lstm.predictor import risk_predictor
from app.modules.risk_prediction import live_risk_prediction_service
from app.utils.camera_manager import CameraManager

router = APIRouter()


class RoadPredictionInput(BaseModel):
    segment_id: str
    name: str = ""
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    road_type: str = "unknown"
    lane_count: int = Field(default=2, ge=1, le=20)
    speed_limit: float = Field(default=40, ge=0, le=200)
    camera_ids: list[str] = Field(default_factory=list)
    traffic_flow: float = Field(default=60, ge=0)
    avg_speed: float = Field(default=40, ge=0)
    historical_accidents_24h: int = Field(default=0, ge=0)
    historical_accidents_7d: int = Field(default=0, ge=0)


class RiskPredictionRequest(BaseModel):
    segments: list[RoadPredictionInput]
    selected_segment_id: str | None = None


@router.get("/")
async def get_all_risks():
    risks = risk_predictor.get_all_risks()
    all_cams = CameraManager().get_all()

    # 按 config.yaml 中的顺序映射: cam_index → C{index+1:02d}
    camera_risks_mapped: dict[str, float] = {}
    for i, cam in enumerate(all_cams):
        frontend_id = f"C{i + 1:02d}"
        if cam.id in risks:
            camera_risks_mapped[frontend_id] = risks[cam.id]

    return {
        "camera_risks": camera_risks_mapped,
        "detailed": risk_predictor.get_detailed_risks(),
    }


@router.post("/prediction")
async def predict_road_risks(request: RiskPredictionRequest):
    return await live_risk_prediction_service.predict(
        [item.model_dump() for item in request.segments],
        selected_segment_id=request.selected_segment_id,
    )


@router.get("/{camera_id}")
async def get_camera_risk(camera_id: str):
    risk = risk_predictor.get_camera_risk(camera_id)
    if risk is None:
        return {"camera_id": camera_id, "risk_score": None, "message": "no data yet"}
    return {"camera_id": camera_id, "risk_score": risk}
