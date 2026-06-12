def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_images(client):
    response = client.get("/api/images")
    assert response.status_code == 200
    data = response.json()
    assert "images" in data
    assert "test.png" in data["images"]


def test_get_image(client):
    response = client.get("/api/images/test.png")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")


def test_get_image_not_found(client):
    response = client.get("/api/images/notexist.png")
    assert response.status_code == 404
