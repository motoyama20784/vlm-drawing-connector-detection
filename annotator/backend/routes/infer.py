import base64
import io
import json
import re
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException
from PIL import Image
from pydantic import BaseModel

from annotator.backend.config import get_config, Config

router = APIRouter()


class BboxCoords(BaseModel):
    x_center: float
    y_center: float
    width: float
    height: float


class InferRequest(BaseModel):
    image: str
    dir: str = "samples"
    bbox: BboxCoords


def _crop_to_base64(image_path: Path, bbox: BboxCoords) -> str:
    with Image.open(image_path) as img:
        w, h = img.size
        x1 = int((bbox.x_center - bbox.width / 2) * w)
        y1 = int((bbox.y_center - bbox.height / 2) * h)
        x2 = int((bbox.x_center + bbox.width / 2) * w)
        y2 = int((bbox.y_center + bbox.height / 2) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        cropped = img.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()


def _parse_vlm_response(text: str) -> dict:
    match = re.search(r'\{[^{}]*"vlm_text"[^{}]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"vlm_text": "", "vlm_shape": ""}


@router.post("/infer")
def infer_bbox(req: InferRequest, config: Config = Depends(get_config)):
    image_path = config.inputs_dir / req.dir / req.image
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    image_b64 = _crop_to_base64(image_path, req.bbox)

    prompt_path = Path(config.prompt_file)
    prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""

    with httpx.Client() as http_client:
        resp = http_client.post(
            f"{config.ollama_base_url}/api/generate",
            json={
                "model": config.model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
            },
            timeout=60.0,
        )
        resp.raise_for_status()

    result = _parse_vlm_response(resp.json().get("response", ""))
    return result
