from fastapi import APIRouter

from app.modules.lstm.predictor import risk_predictor
from app.utils.camera_manager import CameraManager

router = APIRouter()


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


@router.get("/{camera_id}")
async def get_camera_risk(camera_id: str):
    risk = risk_predictor.get_camera_risk(camera_id)
    if risk is None:
        return {"camera_id": camera_id, "risk_score": None, "message": "no data yet"}
    return {"camera_id": camera_id, "risk_score": risk}
