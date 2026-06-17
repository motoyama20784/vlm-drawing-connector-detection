def compute_iou(box1: dict, box2: dict) -> float:
    x1_1 = box1["x_center"] - box1["width"] / 2
    y1_1 = box1["y_center"] - box1["height"] / 2
    x2_1 = box1["x_center"] + box1["width"] / 2
    y2_1 = box1["y_center"] + box1["height"] / 2

    x1_2 = box2["x_center"] - box2["width"] / 2
    y1_2 = box2["y_center"] - box2["height"] / 2
    x2_2 = box2["x_center"] + box2["width"] / 2
    y2_2 = box2["y_center"] + box2["height"] / 2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = box1["width"] * box1["height"]
    area2 = box2["width"] * box2["height"]
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def _match_boxes(pred_boxes: list, gt_boxes: list, iou_threshold: float) -> dict:
    """
    Greedy matching of pred boxes to GT boxes (1-to-1).

    Returns:
        matched_pairs:   [{"pred_idx": int, "gt_idx": int, "iou": float}, ...]
        unmatched_preds: [{"pred_idx": int, "max_iou_to_gt": float}, ...]
                         max_iou_to_gt は B/D 分類に使う（マッチ済みGTも含む全GTとの最大IoU）
        unmatched_gts:   [{"gt_idx": int}, ...]
    """
    n_gt = len(gt_boxes)

    iou_matrix = [
        [compute_iou(pred_boxes[i], gt_boxes[j]) for j in range(n_gt)]
        for i in range(len(pred_boxes))
    ]

    matched_gt = set()
    matched_pred = set()
    matched_pairs = []

    for i, row in enumerate(iou_matrix):
        best_iou, best_j = 0.0, -1
        for j, iou in enumerate(row):
            if j not in matched_gt and iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_threshold and best_j != -1:
            matched_pairs.append({"pred_idx": i, "gt_idx": best_j, "iou": best_iou})
            matched_gt.add(best_j)
            matched_pred.add(i)

    unmatched_preds = [
        {"pred_idx": i, "max_iou_to_gt": max(iou_matrix[i]) if n_gt > 0 else 0.0}
        for i in range(len(pred_boxes))
        if i not in matched_pred
    ]

    unmatched_gts = [
        {"gt_idx": j}
        for j in range(n_gt)
        if j not in matched_gt
    ]

    return {
        "matched_pairs": matched_pairs,
        "unmatched_preds": unmatched_preds,
        "unmatched_gts": unmatched_gts,
    }


def evaluate(
    pred_boxes: list,
    gt_boxes: list,
    iou_threshold: float = 0.5,
    near_iou_min: float = 0.1,
) -> dict:
    """
    pred_boxes と gt_boxes を照合し、検出精度の各種指標を返す。

    Args:
        iou_threshold : TP と判定する IoU の下限（デフォルト 0.5）
        near_iou_min  : [B] near FP と判定する IoU の下限（デフォルト 0.1）
                        near_iou_min ≤ max_iou_to_gt < iou_threshold を near FP とする

    Returns (常に以下のキーを含む):
        precision, recall, f1  : 基本指標
        tp, fp, fn             : マッチング件数
        miss_rate              : [A] FN / GT総数（見逃し率）
        near_fp_count          : [B] near FP の件数
        near_fp_rate           : [B] near FP / pred総数
        matched_pairs          : TP の (pred_idx, gt_idx, iou) リスト  ← C/E/F で使用
        unmatched_preds        : FP の (pred_idx, max_iou_to_gt) リスト ← B/D で使用
        unmatched_gts          : FN の (gt_idx,) リスト               ← A/F で使用
    """
    if not gt_boxes and not pred_boxes:
        return {
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "tp": 0, "fp": 0, "fn": 0,
            "miss_rate": 0.0,
            "near_fp_count": 0, "near_fp_rate": 0.0,
            "matched_pairs": [],
            "unmatched_preds": [],
            "unmatched_gts": [],
        }

    if not gt_boxes:
        return {
            "precision": 0.0, "recall": 1.0, "f1": 0.0,
            "tp": 0, "fp": len(pred_boxes), "fn": 0,
            "miss_rate": 0.0,
            "near_fp_count": 0, "near_fp_rate": 0.0,
            "matched_pairs": [],
            "unmatched_preds": [
                {"pred_idx": i, "max_iou_to_gt": 0.0}
                for i in range(len(pred_boxes))
            ],
            "unmatched_gts": [],
        }

    if not pred_boxes:
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "tp": 0, "fp": 0, "fn": len(gt_boxes),
            "miss_rate": 1.0,
            "near_fp_count": 0, "near_fp_rate": 0.0,
            "matched_pairs": [],
            "unmatched_preds": [],
            "unmatched_gts": [{"gt_idx": j} for j in range(len(gt_boxes))],
        }

    matching = _match_boxes(pred_boxes, gt_boxes, iou_threshold)

    tp = len(matching["matched_pairs"])
    fp = len(matching["unmatched_preds"])
    fn = len(matching["unmatched_gts"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    miss_rate = fn / len(gt_boxes)

    near_fp_count = sum(
        1 for p in matching["unmatched_preds"]
        if p["max_iou_to_gt"] >= near_iou_min
    )
    near_fp_rate = near_fp_count / len(pred_boxes)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "miss_rate": miss_rate,
        "near_fp_count": near_fp_count,
        "near_fp_rate": near_fp_rate,
        "matched_pairs": matching["matched_pairs"],
        "unmatched_preds": matching["unmatched_preds"],
        "unmatched_gts": matching["unmatched_gts"],
    }
