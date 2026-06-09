import json
import pytest
from src.experiment.runner import load_ground_truth


def test_load_ground_truth_found(tmp_path):
    gt_data = {
        "connectors": [
            {"x_center": 0.5, "y_center": 0.5, "width": 0.1, "height": 0.1}
        ]
    }
    gt_file = tmp_path / "drawing_001.json"
    gt_file.write_text(json.dumps(gt_data))

    result = load_ground_truth(str(tmp_path), "drawing_001.png")
    assert len(result) == 1
    assert result[0]["x_center"] == 0.5


def test_load_ground_truth_not_found(tmp_path):
    result = load_ground_truth(str(tmp_path), "nonexistent.png")
    assert result == []


def test_load_ground_truth_empty(tmp_path):
    gt_data = {"connectors": []}
    gt_file = tmp_path / "drawing_empty.json"
    gt_file.write_text(json.dumps(gt_data))

    result = load_ground_truth(str(tmp_path), "drawing_empty.png")
    assert result == []
