from pydantic import BaseModel


class CameraResponse(BaseModel):
    id: str
    name: str
    longitude: float
    latitude: float
