import json
from fastapi import APIRouter, Depends
from annotator.backend.config import get_config, Config

router = APIRouter()

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


@router.get("/dirs")
def list_dirs(config: Config = Depends(get_config)):
    dirs = []
    for d in sorted(config.data_dir.iterdir()):
        if not d.is_dir():
            continue
        if any(f.suffix.lower() in IMAGE_EXTENSIONS for f in d.iterdir() if f.is_file()):
            dirs.append(d.name)
    return {"dirs": dirs}


@router.get("/status")
def get_status(dir: str = "samples", config: Config = Depends(get_config)):
    target_dir = config.data_dir / dir
    gt_dir = config.data_dir / "ground_truth"

    if not target_dir.exists():
        return {"images": []}

    images = sorted(
        f for f in target_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    result = []
    for img in images:
        gt_path = gt_dir / f"{img.stem}.json"
        if gt_path.exists():
            try:
                data = json.loads(gt_path.read_text(encoding="utf-8"))
                count = len(data.get("connectors", []))
                completed = data.get("completed", False)
            except Exception:
                count = 0
                completed = False
        else:
            count = 0
            completed = False
        result.append({
            "filename": img.name,
            "completed": completed,
            "bbox_count": count,
        })
    return {"images": result, "dir": dir}
