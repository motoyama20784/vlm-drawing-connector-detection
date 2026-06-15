import json
from pathlib import Path
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional
from annotator.backend.config import get_config, Config

router = APIRouter()


class BboxItem(BaseModel):
    id: str
    x_center: float
    y_center: float
    width: float
    height: float
    category: str
    vlm_text: Optional[str] = None
    vlm_shape: Optional[str] = None


class AnnotationData(BaseModel):
    image: str
    connectors: list[BboxItem]
    completed: bool = False


@router.get("/annotations/{filename}")
def get_annotation(filename: str, config: Config = Depends(get_config)):
    path = config.data_dir / "ground_truth" / f"{Path(filename).stem}.json"
    if not path.exists():
        return {"image": filename, "connectors": [], "completed": False}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.post("/annotations/{filename}")
def save_annotation(
    filename: str,
    data: AnnotationData,
    config: Config = Depends(get_config),
):
    gt_dir = config.data_dir / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    path = gt_dir / f"{Path(filename).stem}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data.model_dump(), f, ensure_ascii=False, indent=2)
    return {"saved": str(path)}
