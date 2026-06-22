from __future__ import annotations

import random
import string
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

from annotator.backend.config import Config, get_config

router = APIRouter()

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
FONT_DIRS = [
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path("/app/fonts"),
]
FONT_EXTENSIONS = {".ttf", ".otf"}
_font_cache: Optional[list] = None


def find_fonts() -> list:
    global _font_cache
    if _font_cache is not None:
        return _font_cache
    result = []
    for d in FONT_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in FONT_EXTENSIONS:
                result.append({"name": p.stem, "path": str(p)})
    _font_cache = result
    return result


def get_font_path(name: str) -> Optional[str]:
    for f in find_fonts():
        if f["name"] == name:
            return f["path"]
    return None


def sample_background(img: Image.Image, x1: int, y1: int, x2: int, y2: int) -> tuple:
    """Median color sampled from border pixels of the bbox region."""
    iw, ih = img.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(iw - 1, x2), min(ih - 1, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        px = img.getpixel((max(0, min(x1, iw - 1)), max(0, min(y1, ih - 1))))
        return px if isinstance(px, tuple) else (px,)

    border = max(1, min(5, bw // 5, bh // 5))
    sx = max(1, bw // 15)
    sy = max(1, bh // 15)
    samples: list[tuple] = []

    for x in range(x1, x2 + 1, sx):
        for y in range(y1, min(y1 + border + 1, y2 + 1)):
            p = img.getpixel((x, y))
            samples.append(p if isinstance(p, tuple) else (p,))
        for y in range(max(y1, y2 - border), y2 + 1):
            p = img.getpixel((x, y))
            samples.append(p if isinstance(p, tuple) else (p,))
    for y in range(y1 + border, y2 - border + 1, sy):
        for x in range(x1, min(x1 + border + 1, x2 + 1)):
            p = img.getpixel((x, y))
            samples.append(p if isinstance(p, tuple) else (p,))
        for x in range(max(x1, x2 - border), x2 + 1):
            p = img.getpixel((x, y))
            samples.append(p if isinstance(p, tuple) else (p,))

    if not samples:
        p = img.getpixel((x1, y1))
        return p if isinstance(p, tuple) else (p,)

    n_ch = len(samples[0])
    return tuple(sorted(s[c] for s in samples)[len(samples) // 2] for c in range(n_ch))


def contrasting_color(bg: tuple) -> tuple:
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    base = (20, 20, 20) if lum > 128 else (235, 235, 235)
    if len(bg) == 4:
        return base + (255,)
    return base


def random_text(n: int) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


# ---------- Pydantic schemas ----------

class BboxItem(BaseModel):
    x_center: float
    y_center: float
    width: float
    height: float


class MaskRequest(BaseModel):
    filename: str
    dir: str
    bboxes: list[BboxItem]
    font_name: Optional[str] = None


# ---------- Endpoints ----------

@router.get("/masking/fonts")
def list_fonts():
    return {"fonts": find_fonts()}


@router.get("/masking/status")
def masking_status(dir: str = "samples", config: Config = Depends(get_config)):
    src_dir = config.inputs_dir / dir
    masked_dir = config.inputs_dir / f"{dir}_masked"
    if not src_dir.exists():
        return {"images": []}
    images = []
    for f in sorted(src_dir.iterdir()):
        if f.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        images.append({
            "filename": f.name,
            "masked": (masked_dir / f.name).exists(),
        })
    return {"images": images}


@router.post("/masking/apply")
def apply_masking(req: MaskRequest, config: Config = Depends(get_config)):
    src_path = config.inputs_dir / req.dir / req.filename
    if not src_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    out_dir_name = f"{req.dir}_masked"
    out_dir = config.inputs_dir / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / req.filename

    img = Image.open(src_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    iw, ih = img.size

    font_path = get_font_path(req.font_name) if req.font_name else None

    for bbox in req.bboxes:
        x1 = int((bbox.x_center - bbox.width / 2) * iw)
        y1 = int((bbox.y_center - bbox.height / 2) * ih)
        x2 = int((bbox.x_center + bbox.width / 2) * iw)
        y2 = int((bbox.y_center + bbox.height / 2) * ih)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(iw, x2), min(ih, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        bg = sample_background(img, x1, y1, x2, y2)
        fg = contrasting_color(bg)

        draw.rectangle([x1, y1, x2, y2], fill=bg)

        font_size = max(8, int((y2 - y1) * 0.70))
        font = None
        if font_path:
            try:
                font = ImageFont.truetype(font_path, font_size)
            except Exception:
                font = None
        if font is None:
            try:
                font = ImageFont.load_default(size=font_size)
            except TypeError:
                font = ImageFont.load_default()

        n_chars = max(1, int((x2 - x1) / (font_size * 0.55)))
        text = random_text(n_chars)

        try:
            tb = font.getbbox(text)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except AttributeError:
            tw = len(text) * max(6, font_size // 2)
            th = font_size

        tx = x1 + max(0, ((x2 - x1) - tw) // 2)
        ty = y1 + max(0, ((y2 - y1) - th) // 2)
        draw.text((tx, ty), text, fill=fg, font=font)

    if src_path.suffix.lower() in (".jpg", ".jpeg"):
        img = img.convert("RGB")
        img.save(str(out_path), quality=95)
    else:
        img.save(str(out_path))

    return {
        "output_dir": out_dir_name,
        "output_filename": req.filename,
        "message": f"Saved to inputs/{out_dir_name}/{req.filename}",
    }
