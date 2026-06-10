import pytest
from src.detection.evaluator import compute_iou, evaluate


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
