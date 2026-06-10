from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def draw_bboxes(image_path: str, boxes: list, output_path: str) -> None:
    """正規化座標のbboxリストを画像に描画して保存する。

    boxes: [{"x_center": float, "y_center": float, "width": float, "height": float}, ...]
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", max(12, h // 50))
    except (IOError, OSError):
        font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        cx = box["x_center"] * w
        cy = box["y_center"] * h
        bw = box["width"] * w
        bh = box["height"] * h

        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1 + 2, y1 + 2), str(i + 1), fill="red", font=font)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
