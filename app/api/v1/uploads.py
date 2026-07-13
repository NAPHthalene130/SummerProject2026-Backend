from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile


uploads_router = APIRouter()

_IMAGE_DIR = Path(__file__).resolve().parents[2] / "data" / "orderImg"
_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

_ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
_MAX_IMAGE_SIZE = 10 * 1024 * 1024


@uploads_router.post("/images")
async def upload_image(file: UploadFile = File(...)) -> dict[str, str]:
    extension = _ALLOWED_IMAGE_TYPES.get(file.content_type or "")
    if extension is None:
        raise HTTPException(status_code=415, detail="仅支持 JPG、PNG、WebP 或 GIF 图片")

    filename = f"mobile-{uuid4().hex}{extension}"
    destination = _IMAGE_DIR / filename
    size = 0

    try:
        with destination.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > _MAX_IMAGE_SIZE:
                    raise HTTPException(status_code=413, detail="图片不能超过 10 MB")
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    if size == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="上传的图片为空")

    return {"url": f"/orderImg/{filename}"}
