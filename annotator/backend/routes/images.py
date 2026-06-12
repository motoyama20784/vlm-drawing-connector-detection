from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from annotator.backend.config import get_config, Config

router = APIRouter()

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


@router.get("/images")
def list_images(config: Config = Depends(get_config)):
    samples_dir = config.data_dir / "samples"
    if not samples_dir.exists():
        return {"images": []}
    images = [
        f.name for f in sorted(samples_dir.iterdir())
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return {"images": images}


@router.get("/images/{filename}")
def get_image(filename: str, config: Config = Depends(get_config)):
    path = config.data_dir / "samples" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(path))
