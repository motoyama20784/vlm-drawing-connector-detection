import pytest
from unittest.mock import patch, MagicMock
from src.detection.detector import detect


def test_detect_returns_response(tmp_path):
    img_path = tmp_path / "test.png"
    img_path.write_bytes(b"fake_image_data")

    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "コネクタが検出されました"}
    mock_response.raise_for_status = MagicMock()

    with patch("src.detection.detector.requests.post", return_value=mock_response):
        result = detect(
            str(img_path),
            "テストプロンプト",
            "gemma4:26b",
            {"temperature": 0.0, "top_p": 0.9, "top_k": 40},
        )

    assert result == "コネクタが検出されました"


def test_detect_raises_on_http_error(tmp_path):
    img_path = tmp_path / "test.png"
    img_path.write_bytes(b"fake_image_data")

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("HTTP Error")

    with patch("src.detection.detector.requests.post", return_value=mock_response):
        with pytest.raises(Exception, match="HTTP Error"):
            detect(
                str(img_path),
                "テストプロンプト",
                "gemma4:26b",
                {"temperature": 0.0, "top_p": 0.9, "top_k": 40},
            )
