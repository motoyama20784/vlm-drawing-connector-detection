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
