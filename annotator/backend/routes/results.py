from fastapi import APIRouter
from pathlib import Path
import json
import re

from annotator.backend.config import get_config

router = APIRouter()

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
_TIMESTAMP_RE = re.compile(r'^\d{8}_\d{6}_')


def _stem_from_result_name(name: str) -> str:
    return _TIMESTAMP_RE.sub('', name)


def _compute_iou(b1: dict, b2: dict) -> float:
    x1 = max(b1["x_center"] - b1["width"] / 2, b2["x_center"] - b2["width"] / 2)
    y1 = max(b1["y_center"] - b1["height"] / 2, b2["y_center"] - b2["height"] / 2)
    x2 = min(b1["x_center"] + b1["width"] / 2, b2["x_center"] + b2["width"] / 2)
    y2 = min(b1["y_center"] + b1["height"] / 2, b2["y_center"] + b2["height"] / 2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = b1["width"] * b1["height"] + b2["width"] * b2["height"] - inter
    return inter / union if union > 0 else 0.0


def _match(pred_boxes: list, gt_boxes: list, iou_threshold: float = 0.3, near_iou_min: float = 0.1) -> dict:
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)

    if not pred_boxes and not gt_boxes:
        return {"tp": [], "near_fp": [], "ghost_fp": [], "fn": [],
                "metrics": {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                            "tp": 0, "fp": 0, "fn": 0, "miss_rate": 0.0, "avg_matched_iou": 0.0}}

    if not gt_boxes:
        return {"tp": [], "near_fp": [],
                "ghost_fp": [{"pred_idx": i, "max_iou_to_gt": 0.0} for i in range(n_pred)],
                "fn": [],
                "metrics": {"precision": 0.0, "recall": 1.0, "f1": 0.0,
                            "tp": 0, "fp": n_pred, "fn": 0, "miss_rate": 0.0, "avg_matched_iou": 0.0}}

    if not pred_boxes:
        return {"tp": [], "near_fp": [], "ghost_fp": [],
                "fn": [{"gt_idx": j} for j in range(n_gt)],
                "metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                            "tp": 0, "fp": 0, "fn": n_gt, "miss_rate": 1.0, "avg_matched_iou": 0.0}}

    iou_mat = [[_compute_iou(pred_boxes[i], gt_boxes[j]) for j in range(n_gt)] for i in range(n_pred)]

    matched_gt, matched_pred = set(), set()
    tp_pairs = []
    for i, row in enumerate(iou_mat):
        best_iou, best_j = 0.0, -1
        for j, iou in enumerate(row):
            if j not in matched_gt and iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_threshold and best_j != -1:
            tp_pairs.append({"pred_idx": i, "gt_idx": best_j, "iou": round(best_iou, 4)})
            matched_gt.add(best_j)
            matched_pred.add(i)

    near_fp, ghost_fp = [], []
    for i in range(n_pred):
        if i in matched_pred:
            continue
        max_iou = max(iou_mat[i])
        (near_fp if max_iou >= near_iou_min else ghost_fp).append(
            {"pred_idx": i, "max_iou_to_gt": round(max_iou, 4)}
        )

    fn = [{"gt_idx": j} for j in range(n_gt) if j not in matched_gt]
    tp_n, fp_n, fn_n = len(tp_pairs), len(near_fp) + len(ghost_fp), len(fn)
    precision = tp_n / (tp_n + fp_n) if (tp_n + fp_n) > 0 else 0.0
    recall = tp_n / (tp_n + fn_n) if (tp_n + fn_n) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp_pairs, "near_fp": near_fp, "ghost_fp": ghost_fp, "fn": fn,
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp_n, "fp": fp_n, "fn": fn_n,
            "miss_rate": round(fn_n / n_gt, 4),
            "avg_matched_iou": round(sum(p["iou"] for p in tp_pairs) / tp_n if tp_n else 0.0, 4),
        },
    }


def _result_files_by_stem(outputs_dir: Path) -> dict:
    result = {}
    if not outputs_dir.exists():
        return result
    for f in outputs_dir.glob("*.json"):
        if not _TIMESTAMP_RE.match(f.stem):
            continue
        stem = _stem_from_result_name(f.stem)
        result.setdefault(stem, []).append(f.name)
    return result


@router.get("/results/list")
def list_results(dir: str = "samples"):
    cfg = get_config()
    inputs_dir = cfg.original_dir / dir
    gt_dir = cfg.data_dir / "ground_truth"
    outputs_dir = cfg.data_dir / "outputs"

    if not inputs_dir.exists():
        return {"dir": dir, "images": []}

    files_by_stem = _result_files_by_stem(outputs_dir)
    images = []

    for img_file in sorted(inputs_dir.iterdir()):
        if img_file.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = img_file.stem
        result_files = sorted(files_by_stem.get(stem, []))
        has_gt = (gt_dir / f"{stem}.json").exists()

        latest_metrics = None
        if result_files and has_gt:
            latest = result_files[-1]
            try:
                pred_boxes = json.loads((outputs_dir / latest).read_text()).get("connectors", [])
                gt_boxes = json.loads((gt_dir / f"{stem}.json").read_text()).get("connectors", [])
                latest_metrics = _match(pred_boxes, gt_boxes)["metrics"]
            except Exception:
                pass

        images.append({
            "filename": img_file.name,
            "has_gt": has_gt,
            "result_files": result_files,
            "latest_metrics": latest_metrics,
        })

    return {"dir": dir, "images": images}


@router.get("/results/evaluate")
def evaluate_results(dir: str, image: str, result_file: str = None, iou_threshold: float = 0.3):
    cfg = get_config()
    stem = Path(image).stem
    gt_dir = cfg.data_dir / "ground_truth"
    outputs_dir = cfg.data_dir / "outputs"

    gt_boxes, completed = [], False
    gt_path = gt_dir / f"{stem}.json"
    if gt_path.exists():
        gt_data = json.loads(gt_path.read_text())
        gt_boxes = gt_data.get("connectors", [])
        completed = gt_data.get("completed", False)

    all_result_files = sorted(_result_files_by_stem(outputs_dir).get(stem, []))
    selected = result_file if result_file in all_result_files else (all_result_files[-1] if all_result_files else None)

    pred_boxes = []
    if selected:
        try:
            pred_boxes = json.loads((outputs_dir / selected).read_text()).get("connectors", [])
        except Exception:
            pass

    return {
        "image": image,
        "dir": dir,
        "gt_boxes": gt_boxes,
        "pred_boxes": pred_boxes,
        "matches": _match(pred_boxes, gt_boxes, iou_threshold=iou_threshold),
        "result_files": all_result_files,
        "selected_result_file": selected,
        "completed": completed,
    }
