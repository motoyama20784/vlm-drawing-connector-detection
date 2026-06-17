import json


def test_get_annotation_not_found(client):
    response = client.get("/api/annotations/test.png")
    assert response.status_code == 200
    data = response.json()
    assert data["image"] == "test.png"
    assert data["connectors"] == []


def test_save_and_load_annotation(client):
    payload = {
        "image": "test.png",
        "connectors": [
            {
                "id": "bbox-1",
                "x_center": 0.25,
                "y_center": 0.40,
                "width": 0.10,
                "height": 0.06,
                "category": "terminal",
                "vlm_text": None,
                "vlm_shape": None,
            }
        ],
    }
    save_response = client.post("/api/annotations/test.png", json=payload)
    assert save_response.status_code == 200

    load_response = client.get("/api/annotations/test.png")
    assert load_response.status_code == 200
    data = load_response.json()
    assert len(data["connectors"]) == 1
    assert data["connectors"][0]["id"] == "bbox-1"
    assert data["connectors"][0]["x_center"] == 0.25


def test_save_annotation_overwrites(client):
    payload_v1 = {
        "image": "test.png",
        "connectors": [{"id": "bbox-1", "x_center": 0.1, "y_center": 0.1,
                        "width": 0.05, "height": 0.05, "category": "",
                        "vlm_text": None, "vlm_shape": None}],
    }
    payload_v2 = {
        "image": "test.png",
        "connectors": [{"id": "bbox-2", "x_center": 0.5, "y_center": 0.5,
                        "width": 0.1, "height": 0.1, "category": "A",
                        "vlm_text": None, "vlm_shape": None}],
    }
    client.post("/api/annotations/test.png", json=payload_v1)
    client.post("/api/annotations/test.png", json=payload_v2)

    response = client.get("/api/annotations/test.png")
    data = response.json()
    assert len(data["connectors"]) == 1
    assert data["connectors"][0]["id"] == "bbox-2"
