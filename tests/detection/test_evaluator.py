import pytest
from src.detection.evaluator import compute_iou, evaluate


# ---------------------------------------------------------------------------
# compute_iou
# ---------------------------------------------------------------------------

def test_iou_perfect_match():
    box = {"x_center": 0.5, "y_center": 0.5, "width": 0.1, "height": 0.1}
    assert compute_iou(box, box) == pytest.approx(1.0)


def test_iou_no_overlap():
    box1 = {"x_center": 0.1, "y_center": 0.1, "width": 0.1, "height": 0.1}
    box2 = {"x_center": 0.9, "y_center": 0.9, "width": 0.1, "height": 0.1}
    assert compute_iou(box1, box2) == 0.0


def test_iou_partial_overlap():
    box1 = {"x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.2}
    box2 = {"x_center": 0.6, "y_center": 0.5, "width": 0.2, "height": 0.2}
    iou = compute_iou(box1, box2)
    assert 0.0 < iou < 1.0


# ---------------------------------------------------------------------------
# evaluate: 基本指標（既存テスト）
# ---------------------------------------------------------------------------

def test_evaluate_perfect_detection():
    box = {"x_center": 0.5, "y_center": 0.5, "width": 0.1, "height": 0.1}
    metrics = evaluate([box], [box], iou_threshold=0.5)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_evaluate_no_prediction():
    gt = [{"x_center": 0.5, "y_center": 0.5, "width": 0.1, "height": 0.1}]
    metrics = evaluate([], gt, iou_threshold=0.5)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_evaluate_false_positive():
    pred = [{"x_center": 0.5, "y_center": 0.5, "width": 0.1, "height": 0.1}]
    metrics = evaluate(pred, [], iou_threshold=0.5)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 0.0


def test_evaluate_both_empty():
    metrics = evaluate([], [], iou_threshold=0.5)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


# ---------------------------------------------------------------------------
# evaluate: tp / fp / fn カウント
# ---------------------------------------------------------------------------

def test_counts_perfect():
    box = {"x_center": 0.5, "y_center": 0.5, "width": 0.1, "height": 0.1}
    m = evaluate([box], [box])
    assert m["tp"] == 1
    assert m["fp"] == 0
    assert m["fn"] == 0


def test_counts_all_fp():
    pred = {"x_center": 0.9, "y_center": 0.9, "width": 0.1, "height": 0.1}
    gt   = {"x_center": 0.1, "y_center": 0.1, "width": 0.1, "height": 0.1}
    m = evaluate([pred], [gt])
    assert m["tp"] == 0
    assert m["fp"] == 1
    assert m["fn"] == 1


def test_counts_partial():
    gt1 = {"x_center": 0.3, "y_center": 0.5, "width": 0.1, "height": 0.1}
    gt2 = {"x_center": 0.7, "y_center": 0.5, "width": 0.1, "height": 0.1}
    m = evaluate([gt1], [gt1, gt2])
    assert m["tp"] == 1
    assert m["fp"] == 0
    assert m["fn"] == 1


# ---------------------------------------------------------------------------
# [A] miss_rate
# ---------------------------------------------------------------------------

def test_miss_rate_all_missed():
    gt = [{"x_center": 0.5, "y_center": 0.5, "width": 0.1, "height": 0.1}]
    m = evaluate([], gt)
    assert m["miss_rate"] == pytest.approx(1.0)


def test_miss_rate_none_missed():
    box = {"x_center": 0.5, "y_center": 0.5, "width": 0.1, "height": 0.1}
    m = evaluate([box], [box])
    assert m["miss_rate"] == pytest.approx(0.0)


def test_miss_rate_partial():
    gt1 = {"x_center": 0.3, "y_center": 0.5, "width": 0.1, "height": 0.1}
    gt2 = {"x_center": 0.7, "y_center": 0.5, "width": 0.1, "height": 0.1}
    m = evaluate([gt1], [gt1, gt2])
    assert m["miss_rate"] == pytest.approx(0.5)


def test_miss_rate_no_gt():
    pred = {"x_center": 0.5, "y_center": 0.5, "width": 0.1, "height": 0.1}
    m = evaluate([pred], [])
    assert m["miss_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# matched_pairs の構造確認（後続 C/E/F の足場）
# ---------------------------------------------------------------------------

def test_matched_pairs_perfect():
    box = {"x_center": 0.5, "y_center": 0.5, "width": 0.1, "height": 0.1}
    m = evaluate([box], [box])
    assert len(m["matched_pairs"]) == 1
    pair = m["matched_pairs"][0]
    assert pair["pred_idx"] == 0
    assert pair["gt_idx"] == 0
    assert pair["iou"] == pytest.approx(1.0)


def test_matched_pairs_empty_when_no_match():
    pred = {"x_center": 0.9, "y_center": 0.9, "width": 0.1, "height": 0.1}
    gt   = {"x_center": 0.1, "y_center": 0.1, "width": 0.1, "height": 0.1}
    m = evaluate([pred], [gt])
    assert m["matched_pairs"] == []


# ---------------------------------------------------------------------------
# unmatched_preds の構造確認（後続 B/D の足場）
# ---------------------------------------------------------------------------

def test_unmatched_preds_ghost_fp():
    pred = {"x_center": 0.9, "y_center": 0.9, "width": 0.1, "height": 0.1}
    gt   = {"x_center": 0.1, "y_center": 0.1, "width": 0.1, "height": 0.1}
    m = evaluate([pred], [gt])
    assert len(m["unmatched_preds"]) == 1
    assert m["unmatched_preds"][0]["pred_idx"] == 0
    assert m["unmatched_preds"][0]["max_iou_to_gt"] == pytest.approx(0.0)


def test_unmatched_preds_near_fp():
    # pred が GT に近いが IoU < 0.5（near FP）
    pred = {"x_center": 0.55, "y_center": 0.5, "width": 0.1, "height": 0.1}
    gt   = {"x_center": 0.5,  "y_center": 0.5, "width": 0.1, "height": 0.1}
    iou = compute_iou(pred, gt)
    assert 0.0 < iou < 0.5  # near FP の前提確認

    m = evaluate([pred], [gt], iou_threshold=0.5)
    assert len(m["unmatched_preds"]) == 1
    assert m["unmatched_preds"][0]["max_iou_to_gt"] == pytest.approx(iou)


# ---------------------------------------------------------------------------
# unmatched_gts の構造確認（後続 A/F の足場）
# ---------------------------------------------------------------------------

def test_unmatched_gts_all_missed():
    gt1 = {"x_center": 0.3, "y_center": 0.5, "width": 0.1, "height": 0.1}
    gt2 = {"x_center": 0.7, "y_center": 0.5, "width": 0.1, "height": 0.1}
    m = evaluate([], [gt1, gt2])
    assert len(m["unmatched_gts"]) == 2
    assert {e["gt_idx"] for e in m["unmatched_gts"]} == {0, 1}


def test_unmatched_gts_empty_when_all_matched():
    box = {"x_center": 0.5, "y_center": 0.5, "width": 0.1, "height": 0.1}
    m = evaluate([box], [box])
    assert m["unmatched_gts"] == []


# ---------------------------------------------------------------------------
# [B] near_fp_count / near_fp_rate
# ---------------------------------------------------------------------------

def test_near_fp_is_zero_when_perfect():
    box = {"x_center": 0.5, "y_center": 0.5, "width": 0.1, "height": 0.1}
    m = evaluate([box], [box])
    assert m["near_fp_count"] == 0
    assert m["near_fp_rate"] == pytest.approx(0.0)


def test_near_fp_ghost_is_not_counted():
    # GT と全く重ならない pred → ghost FP（near FP ではない）
    pred = {"x_center": 0.9, "y_center": 0.9, "width": 0.1, "height": 0.1}
    gt   = {"x_center": 0.1, "y_center": 0.1, "width": 0.1, "height": 0.1}
    m = evaluate([pred], [gt])
    assert m["near_fp_count"] == 0


def test_near_fp_detected():
    # pred が GT に近いが IoU < 0.5（near FP）
    pred = {"x_center": 0.55, "y_center": 0.5, "width": 0.1, "height": 0.1}
    gt   = {"x_center": 0.5,  "y_center": 0.5, "width": 0.1, "height": 0.1}
    iou = compute_iou(pred, gt)
    assert 0.1 <= iou < 0.5  # near FP の前提確認

    m = evaluate([pred], [gt], iou_threshold=0.5)
    assert m["near_fp_count"] == 1
    assert m["near_fp_rate"] == pytest.approx(1.0)


def test_near_fp_rate_with_mixed_preds():
    # pred1: TP、pred2: near FP、pred3: ghost FP
    gt1  = {"x_center": 0.2, "y_center": 0.5, "width": 0.1, "height": 0.1}
    gt2  = {"x_center": 0.8, "y_center": 0.5, "width": 0.1, "height": 0.1}
    tp   = gt1
    near = {"x_center": 0.85, "y_center": 0.5, "width": 0.1, "height": 0.1}
    ghost = {"x_center": 0.5,  "y_center": 0.0, "width": 0.05, "height": 0.05}

    iou_near  = compute_iou(near,  gt2)
    iou_ghost = compute_iou(ghost, gt1)
    assert 0.1 <= iou_near < 0.5
    assert iou_ghost < 0.1

    m = evaluate([tp, near, ghost], [gt1, gt2], iou_threshold=0.5)
    assert m["tp"] == 1
    assert m["near_fp_count"] == 1
    assert m["near_fp_rate"] == pytest.approx(1 / 3)


def test_near_fp_custom_near_iou_min():
    # near_iou_min を変えると分類が変わることを確認
    pred = {"x_center": 0.55, "y_center": 0.5, "width": 0.1, "height": 0.1}
    gt   = {"x_center": 0.5,  "y_center": 0.5, "width": 0.1, "height": 0.1}
    iou = compute_iou(pred, gt)

    # iou より大きい near_iou_min を設定 → near FP にならない
    m = evaluate([pred], [gt], iou_threshold=0.5, near_iou_min=iou + 0.01)
    assert m["near_fp_count"] == 0

    # iou 以下の near_iou_min を設定 → near FP になる
    m = evaluate([pred], [gt], iou_threshold=0.5, near_iou_min=iou - 0.01)
    assert m["near_fp_count"] == 1


def test_near_fp_no_gt():
    pred = {"x_center": 0.5, "y_center": 0.5, "width": 0.1, "height": 0.1}
    m = evaluate([pred], [])
    assert m["near_fp_count"] == 0
    assert m["near_fp_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# [D] ghost_fp_count / ghost_fp_rate
# ---------------------------------------------------------------------------

def test_ghost_fp_is_zero_when_perfect():
    box = {"x_center": 0.5, "y_center": 0.5, "width": 0.1, "height": 0.1}
    m = evaluate([box], [box])
    assert m["ghost_fp_count"] == 0
    assert m["ghost_fp_rate"] == pytest.approx(0.0)


def test_ghost_fp_detected():
    # GT と全く重ならない pred → ghost FP
    pred = {"x_center": 0.9, "y_center": 0.9, "width": 0.1, "height": 0.1}
    gt   = {"x_center": 0.1, "y_center": 0.1, "width": 0.1, "height": 0.1}
    assert compute_iou(pred, gt) < 0.1  # ghost FP の前提確認

    m = evaluate([pred], [gt])
    assert m["ghost_fp_count"] == 1
    assert m["ghost_fp_rate"] == pytest.approx(1.0)


def test_ghost_fp_near_fp_not_counted():
    # GT に近い pred（near FP）は ghost FP に含まれない
    pred = {"x_center": 0.55, "y_center": 0.5, "width": 0.1, "height": 0.1}
    gt   = {"x_center": 0.5,  "y_center": 0.5, "width": 0.1, "height": 0.1}
    iou = compute_iou(pred, gt)
    assert 0.1 <= iou < 0.5  # near FP の前提確認

    m = evaluate([pred], [gt], iou_threshold=0.5)
    assert m["ghost_fp_count"] == 0


def test_ghost_fp_rate_with_mixed_preds():
    # pred1: TP、pred2: near FP、pred3: ghost FP
    gt1   = {"x_center": 0.2, "y_center": 0.5, "width": 0.1, "height": 0.1}
    gt2   = {"x_center": 0.8, "y_center": 0.5, "width": 0.1, "height": 0.1}
    tp    = gt1
    near  = {"x_center": 0.85, "y_center": 0.5, "width": 0.1, "height": 0.1}
    ghost = {"x_center": 0.5,  "y_center": 0.0, "width": 0.05, "height": 0.05}

    assert compute_iou(ghost, gt1) < 0.1
    assert compute_iou(ghost, gt2) < 0.1

    m = evaluate([tp, near, ghost], [gt1, gt2], iou_threshold=0.5)
    assert m["ghost_fp_count"] == 1
    assert m["ghost_fp_rate"] == pytest.approx(1 / 3)


def test_ghost_fp_and_near_fp_are_exhaustive():
    # near_fp + ghost_fp = fp（全 FP の内訳として過不足ない）
    gt1   = {"x_center": 0.2, "y_center": 0.5, "width": 0.1, "height": 0.1}
    gt2   = {"x_center": 0.8, "y_center": 0.5, "width": 0.1, "height": 0.1}
    tp    = gt1
    near  = {"x_center": 0.85, "y_center": 0.5, "width": 0.1, "height": 0.1}
    ghost = {"x_center": 0.5,  "y_center": 0.0, "width": 0.05, "height": 0.05}

    m = evaluate([tp, near, ghost], [gt1, gt2], iou_threshold=0.5)
    assert m["near_fp_count"] + m["ghost_fp_count"] == m["fp"]


def test_ghost_fp_no_gt():
    # GT なし → 全 pred が ghost FP
    pred = {"x_center": 0.5, "y_center": 0.5, "width": 0.1, "height": 0.1}
    m = evaluate([pred], [])
    assert m["ghost_fp_count"] == 1
    assert m["ghost_fp_rate"] == pytest.approx(1.0)


def test_ghost_fp_no_pred():
    m = evaluate([], [{"x_center": 0.5, "y_center": 0.5, "width": 0.1, "height": 0.1}])
    assert m["ghost_fp_count"] == 0
    assert m["ghost_fp_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# [C] avg_matched_iou
# ---------------------------------------------------------------------------

def test_avg_matched_iou_perfect():
    box = {"x_center": 0.5, "y_center": 0.5, "width": 0.1, "height": 0.1}
    m = evaluate([box], [box])
    assert m["avg_matched_iou"] == pytest.approx(1.0)


def test_avg_matched_iou_no_tp():
    # TP なし → 0.0
    pred = {"x_center": 0.9, "y_center": 0.9, "width": 0.1, "height": 0.1}
    gt   = {"x_center": 0.1, "y_center": 0.1, "width": 0.1, "height": 0.1}
    m = evaluate([pred], [gt])
    assert m["avg_matched_iou"] == pytest.approx(0.0)


def test_avg_matched_iou_no_pred():
    gt = {"x_center": 0.5, "y_center": 0.5, "width": 0.1, "height": 0.1}
    m = evaluate([], [gt])
    assert m["avg_matched_iou"] == pytest.approx(0.0)


def test_avg_matched_iou_partial_overlap():
    # pred が GT より少しズレている場合、IoU < 1.0 になる
    pred = {"x_center": 0.52, "y_center": 0.5, "width": 0.1, "height": 0.1}
    gt   = {"x_center": 0.5,  "y_center": 0.5, "width": 0.1, "height": 0.1}
    iou = compute_iou(pred, gt)
    assert iou >= 0.5  # TP になる前提確認

    m = evaluate([pred], [gt])
    assert m["tp"] == 1
    assert m["avg_matched_iou"] == pytest.approx(iou)


def test_avg_matched_iou_is_mean_of_all_tps():
    # 複数 TP の平均になっていることを確認
    box1 = {"x_center": 0.2, "y_center": 0.5, "width": 0.1, "height": 0.1}
    box2 = {"x_center": 0.8, "y_center": 0.5, "width": 0.1, "height": 0.1}
    pred1 = {"x_center": 0.22, "y_center": 0.5, "width": 0.1, "height": 0.1}
    pred2 = {"x_center": 0.82, "y_center": 0.5, "width": 0.1, "height": 0.1}

    iou1 = compute_iou(pred1, box1)
    iou2 = compute_iou(pred2, box2)
    assert iou1 >= 0.5 and iou2 >= 0.5  # 両方 TP になる前提確認

    m = evaluate([pred1, pred2], [box1, box2])
    assert m["tp"] == 2
    assert m["avg_matched_iou"] == pytest.approx((iou1 + iou2) / 2)
