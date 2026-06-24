import pytest
from src.detection.parser import parse_connectors


def test_parse_valid_json():
    text = '{"connectors": [{"x_center": 0.25, "y_center": 0.40, "width": 0.10, "height": 0.06}]}'
    connectors, parse_failed = parse_connectors(text)
    assert len(connectors) == 1
    assert connectors[0] == {"x_center": 0.25, "y_center": 0.40, "width": 0.10, "height": 0.06}
    assert parse_failed is False


def test_parse_json_embedded_in_text():
    text = 'コネクタを検出しました。\n{"connectors": [{"x_center": 0.25, "y_center": 0.40, "width": 0.10, "height": 0.06}]}'
    connectors, parse_failed = parse_connectors(text)
    assert len(connectors) == 1
    assert parse_failed is False


def test_parse_no_connectors():
    text = '{"connectors": []}'
    connectors, parse_failed = parse_connectors(text)
    assert connectors == []
    assert parse_failed is False


def test_parse_invalid_output_returns_empty_and_failed():
    text = "コネクタは見つかりませんでした。"
    connectors, parse_failed = parse_connectors(text)
    assert connectors == []
    assert parse_failed is True


def test_parse_multiple_connectors():
    text = '{"connectors": [{"x_center": 0.25, "y_center": 0.40, "width": 0.10, "height": 0.06}, {"x_center": 0.60, "y_center": 0.30, "width": 0.08, "height": 0.05}]}'
    connectors, parse_failed = parse_connectors(text)
    assert len(connectors) == 2
    assert parse_failed is False
