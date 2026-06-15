import pytest
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient
from annotator.backend.main import app
from annotator.backend.config import get_config, Config


@pytest.fixture
def tmp_data(tmp_path):
    inputs = tmp_path / "data" / "inputs"
    ground_truth = tmp_path / "data" / "ground_truth"
    inputs.mkdir(parents=True)
    ground_truth.mkdir(parents=True)
    img = Image.new("RGB", (200, 100), color=(128, 128, 128))
    img.save(inputs / "test.png")
    return tmp_path


@pytest.fixture
def client(tmp_data):
    def override():
        return Config(
            data_dir=tmp_data / "data",
            ollama_base_url="http://mock-ollama:11434",
            model="test-model",
            prompt_file=str(tmp_data / "prompts" / "annotator" / "v1.txt"),
        )
    app.dependency_overrides[get_config] = override
    yield TestClient(app)
    app.dependency_overrides.clear()
