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


# ---------- Font / text helpers ----------

def _load_font(font_path: Optional[str], font_size: int) -> ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=font_size)
    except TypeError:
        return ImageFont.load_default()


def _measure(font, text: str) -> tuple:
    try:
        tb = font.getbbox(text)
        return tb[2] - tb[0], tb[3] - tb[1]
    except AttributeError:
        return len(text) * 8, 13


def _draw_text_in_bbox(
    img: Image.Image,
    x1: int, y1: int, x2: int, y2: int,
    fg: tuple,
    font_path: Optional[str],
) -> None:
    """
    bbox 内にランダムテキストを描画する。
    縦長 bbox (bh > bw) のときは横書きを90°回転して縦書き風にレンダリング。
    テキストが bbox をはみ出さないようフォントサイズ縮小・文字数切り詰めを行う。
    """
    bw, bh = x2 - x1, y2 - y1
    margin = 4
    is_vertical = bh > bw

    if is_vertical:
        # フォントサイズ = bbox 幅に収まるサイズ（回転後の文字高さ）
        font_size = max(8, int(bw * 0.80))
        font = _load_font(font_path, font_size)

        # 文字数 = bbox 高さに収まる文字列幅を想定
        n_chars = max(1, int(bh / (font_size * 0.65)))
        text = random_text(n_chars)
        tw, th = _measure(font, text)

        # 回転後に縦方向へはみ出さないよう文字数を切り詰め (tw → 回転後の高さ)
        while tw > bh - margin and len(text) > 1:
            text = text[:-1]
            tw, th = _measure(font, text)

        # 水平テキストを一時 RGBA に描いて 90° 時計回りに回転 → 上から下へ読む縦書き
        pad = 2
        tmp = Image.new("RGBA", (max(1, tw + pad * 2), max(1, th + pad * 2)), (0, 0, 0, 0))
        ImageDraw.Draw(tmp).text((pad, pad), text, fill=fg, font=font)
        rotated = tmp.rotate(-90, expand=True)   # -90 = CW
        rw, rh = rotated.size

        px = x1 + max(0, (bw - rw) // 2)
        py = y1 + max(0, (bh - rh) // 2)
        img.paste(rotated, (px, py), rotated)

    else:
        # フォントサイズ = bbox 高さに収まるサイズ
        font_size = max(8, int(bh * 0.75))
        font = _load_font(font_path, font_size)
        n_chars = max(1, int(bw / (font_size * 0.55)))
        text = random_text(n_chars)
        tw, th = _measure(font, text)

        # bbox 幅に収まるまでフォントサイズを縮小
        while tw > bw - margin and font_size > 8:
            font_size = max(8, font_size - 2)
            font = _load_font(font_path, font_size)
            tw, th = _measure(font, text)

        # それでも収まらなければ文字数を切り詰め
        while tw > bw - margin and len(text) > 1:
            text = text[:-1]
            tw, th = _measure(font, text)

        tx = x1 + max(0, (bw - tw) // 2)
        ty = y1 + max(0, (bh - th) // 2)
        ImageDraw.Draw(img).text((tx, ty), text, fill=fg, font=font)


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

        # 背景色で塗りつぶし
        ImageDraw.Draw(img).rectangle([x1, y1, x2, y2], fill=bg)
        # テキスト描画（縦長なら縦書き風、横長なら横書き）
        _draw_text_in_bbox(img, x1, y1, x2, y2, fg, font_path)

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
