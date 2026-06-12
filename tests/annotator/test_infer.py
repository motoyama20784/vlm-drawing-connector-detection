import pytest
import respx
import httpx
from pathlib import Path


MOCK_OLLAMA_RESPONSE = {
    "response": '{"vlm_text": "CN1", "vlm_shape": "rectangular"}',
    "done": True,
}


@respx.mock
def test_infer_bbox(client, tmp_data):
    prompt_dir = tmp_data / "prompts" / "annotator"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "v1.txt").write_text(
        'この画像を見て {"vlm_text": "テキスト", "vlm_shape": "形状"} で答えてください'
    )

    respx.post("http://mock-ollama:11434/api/generate").mock(
        return_value=httpx.Response(200, json=MOCK_OLLAMA_RESPONSE)
    )

    response = client.post(
        "/api/infer",
        json={
            "image": "test.png",
            "bbox": {"x_center": 0.5, "y_center": 0.5, "width": 0.4, "height": 0.4},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["vlm_text"] == "CN1"
    assert data["vlm_shape"] == "rectangular"


@respx.mock
def test_infer_bbox_parse_failure(client, tmp_data):
    prompt_dir = tmp_data / "prompts" / "annotator"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "v1.txt").write_text("テスト")

    respx.post("http://mock-ollama:11434/api/generate").mock(
        return_value=httpx.Response(200, json={"response": "解析できないテキスト", "done": True})
    )

    response = client.post(
        "/api/infer",
        json={
            "image": "test.png",
            "bbox": {"x_center": 0.5, "y_center": 0.5, "width": 0.4, "height": 0.4},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["vlm_text"] == ""
    assert data["vlm_shape"] == ""


def test_infer_image_not_found(client):
    response = client.post(
        "/api/infer",
        json={
            "image": "notexist.png",
            "bbox": {"x_center": 0.5, "y_center": 0.5, "width": 0.1, "height": 0.1},
        },
    )
    assert response.status_code == 404
