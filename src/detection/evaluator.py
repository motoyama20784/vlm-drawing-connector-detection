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


def evaluate(pred_boxes: list, gt_boxes: list, iou_threshold: float = 0.5) -> dict:
    if not gt_boxes and not pred_boxes:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    if not gt_boxes:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}

    if not pred_boxes:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    matched_gt = set()
    tp = 0

    for pred in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}
