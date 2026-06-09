import pytest
from src.detection.parser import parse_connectors


def test_parse_valid_json():
    text = '{"connectors": [{"x_center": 0.25, "y_center": 0.40, "width": 0.10, "height": 0.06}]}'
    result = parse_connectors(text)
    assert len(result) == 1
    assert result[0] == {"x_center": 0.25, "y_center": 0.40, "width": 0.10, "height": 0.06}


def test_parse_json_embedded_in_text():
    text = 'コネクタを検出しました。\n{"connectors": [{"x_center": 0.25, "y_center": 0.40, "width": 0.10, "height": 0.06}]}'
    result = parse_connectors(text)
    assert len(result) == 1


def test_parse_no_connectors():
    text = '{"connectors": []}'
    result = parse_connectors(text)
    assert result == []


def test_parse_invalid_output_returns_empty():
    text = "コネクタは見つかりませんでした。"
    result = parse_connectors(text)
    assert result == []


def test_parse_multiple_connectors():
    text = '{"connectors": [{"x_center": 0.25, "y_center": 0.40, "width": 0.10, "height": 0.06}, {"x_center": 0.60, "y_center": 0.30, "width": 0.08, "height": 0.05}]}'
    result = parse_connectors(text)
    assert len(result) == 2
