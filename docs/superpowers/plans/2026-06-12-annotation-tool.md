# アノテーションツール実装プラン

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** FastAPI + React で bbox アノテーションツールを構築し、VLM (ollama) で各 bbox 内のテキスト・形状を自動抽出して JSON として保存できる Web アプリを完成させる

**Architecture:** annotator コンテナ（マルチステージ Dockerfile: Node.js で React をビルド → Python/FastAPI イメージに組み込む）を既存 docker-compose.yml に追加する。FastAPI が `/api/*` でバックエンド処理、`/` で React SPA を静的配信する。テストは python コンテナから `pytest tests/annotator/` で実行する。

**Tech Stack:** Python 3.11 / FastAPI / uvicorn / httpx / Pillow / React 18 / Vite / axios

---

## ファイルマップ

### 新規作成
| ファイル | 役割 |
|---------|------|
| `annotator/backend/main.py` | FastAPI アプリ本体・静的ファイル配信 |
| `annotator/backend/config.py` | 環境変数ラッパー（DI 対応） |
| `annotator/backend/routes/images.py` | 画像一覧・配信エンドポイント |
| `annotator/backend/routes/annotations.py` | アノテーション読み書きエンドポイント |
| `annotator/backend/routes/infer.py` | bbox クロップ→ollama 推論エンドポイント |
| `annotator/frontend/package.json` | React + Vite 依存定義 |
| `annotator/frontend/vite.config.js` | ビルド設定・開発用 API プロキシ |
| `annotator/frontend/index.html` | HTML エントリポイント |
| `annotator/frontend/src/main.jsx` | React エントリポイント |
| `annotator/frontend/src/App.jsx` | ルートコンポーネント（状態管理） |
| `annotator/frontend/src/api.js` | axios API ラッパー |
| `annotator/frontend/src/components/ImageSelector.jsx` | 画像選択ドロップダウン |
| `annotator/frontend/src/components/AnnotationCanvas.jsx` | Canvas bbox 描画 |
| `annotator/frontend/src/components/BboxList.jsx` | bbox リスト・VLM推論ボタン |
| `docker/annotator/Dockerfile` | マルチステージビルド定義 |
| `docker/annotator/requirements.txt` | annotator コンテナの Python 依存 |
| `prompts/annotator/v1.txt` | VLM 推論用プロンプト |
| `tests/annotator/__init__.py` | テストパッケージ |
| `tests/annotator/conftest.py` | テスト用フィクスチャ |
| `tests/annotator/test_images.py` | 画像 API テスト |
| `tests/annotator/test_annotations.py` | アノテーション API テスト |
| `tests/annotator/test_infer.py` | 推論 API テスト |

### 変更
| ファイル | 変更内容 |
|---------|---------|
| `docker-compose.yml` | annotator サービス追加 |
| `docker/python/requirements.txt` | fastapi / httpx / respx / python-multipart 追加 |

---

## Task 1: ディレクトリ構造・プロンプトファイルの作成

**Files:**
- Create: `annotator/__init__.py`
- Create: `annotator/backend/__init__.py`
- Create: `annotator/backend/routes/__init__.py`
- Create: `annotator/frontend/src/components/.gitkeep`
- Create: `prompts/annotator/v1.txt`
- Create: `tests/annotator/__init__.py`

- [ ] **Step 1: ディレクトリと空ファイルを作成**

```bash
mkdir -p annotator/backend/routes
mkdir -p annotator/frontend/src/components
mkdir -p prompts/annotator
mkdir -p tests/annotator
touch annotator/__init__.py
touch annotator/backend/__init__.py
touch annotator/backend/routes/__init__.py
touch tests/annotator/__init__.py
```

- [ ] **Step 2: プロンプトファイルを作成**

`prompts/annotator/v1.txt`:
```
この図面の一部を見てください。
以下のJSON形式のみで回答してください：
{"vlm_text": "画像内のテキストや記号", "vlm_shape": "形状の種類（例: rectangular, circular, line など）"}
テキストがなければ vlm_text は空文字にしてください。
```

- [ ] **Step 3: コミット**

```bash
git add annotator/ prompts/annotator/ tests/annotator/__init__.py
git commit -m "feat: add annotator directory structure and prompt file"
```

---

## Task 2: Config モジュールと FastAPI 基盤

**Files:**
- Create: `annotator/backend/config.py`
- Create: `annotator/backend/main.py`
- Create: `tests/annotator/conftest.py`

- [ ] **Step 1: テストの conftest.py を書く**

`tests/annotator/conftest.py`:
```python
import pytest
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient
from annotator.backend.main import app
from annotator.backend.config import get_config, Config


@pytest.fixture
def tmp_data(tmp_path):
    samples = tmp_path / "data" / "samples"
    ground_truth = tmp_path / "data" / "ground_truth"
    samples.mkdir(parents=True)
    ground_truth.mkdir(parents=True)
    img = Image.new("RGB", (200, 100), color=(128, 128, 128))
    img.save(samples / "test.png")
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
```

- [ ] **Step 2: config.py を作成**

`annotator/backend/config.py`:
```python
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    data_dir: Path
    ollama_base_url: str
    model: str
    prompt_file: str


def get_config() -> Config:
    return Config(
        data_dir=Path(os.getenv("DATA_DIR", "/app/data")),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
        model=os.getenv("MODEL", "gemma4:26b"),
        prompt_file=os.getenv("ANNOTATOR_PROMPT_FILE", "prompts/annotator/v1.txt"),
    )
```

- [ ] **Step 3: health エンドポイントのテストを書く**

`tests/annotator/test_images.py` の先頭に追加（ファイルをこれで作成）:
```python
def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 4: テストが失敗することを確認**

```bash
docker compose exec python pytest tests/annotator/test_images.py::test_health -v
```

Expected: `ModuleNotFoundError` or `ImportError`（main.py 未作成のため）

- [ ] **Step 5: main.py を作成**

`annotator/backend/main.py`:
```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from annotator.backend.routes import images, annotations, infer

app = FastAPI(title="Annotation Tool")

app.include_router(images.router, prefix="/api")
app.include_router(annotations.router, prefix="/api")
app.include_router(infer.router, prefix="/api")

@app.get("/api/health")
def health():
    return {"status": "ok"}

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
```

- [ ] **Step 6: テストが通ることを確認**

```bash
docker compose exec python pytest tests/annotator/test_images.py::test_health -v
```

Expected: PASSED

- [ ] **Step 7: コミット**

```bash
git add annotator/backend/config.py annotator/backend/main.py tests/annotator/conftest.py tests/annotator/test_images.py
git commit -m "feat: add FastAPI app foundation with health endpoint"
```

---

## Task 3: 画像一覧・配信 API

**Files:**
- Create: `annotator/backend/routes/images.py`
- Modify: `tests/annotator/test_images.py`

- [ ] **Step 1: テストを書く**

`tests/annotator/test_images.py`（上書き）:
```python
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
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
docker compose exec python pytest tests/annotator/test_images.py -v
```

Expected: health PASSED, others FAILED（routes/images.py 未作成のため）

- [ ] **Step 3: routes/images.py を作成**

`annotator/backend/routes/images.py`:
```python
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from annotator.backend.config import get_config, Config

router = APIRouter()

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


@router.get("/images")
def list_images(config: Config = Depends(get_config)):
    samples_dir = config.data_dir / "samples"
    if not samples_dir.exists():
        return {"images": []}
    images = [
        f.name for f in sorted(samples_dir.iterdir())
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return {"images": images}


@router.get("/images/{filename}")
def get_image(filename: str, config: Config = Depends(get_config)):
    path = config.data_dir / "samples" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(path))
```

- [ ] **Step 4: テストが通ることを確認**

```bash
docker compose exec python pytest tests/annotator/test_images.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: コミット**

```bash
git add annotator/backend/routes/images.py tests/annotator/test_images.py
git commit -m "feat: add image list and serve endpoints"
```

---

## Task 4: アノテーション読み書き API

**Files:**
- Create: `annotator/backend/routes/annotations.py`
- Create: `tests/annotator/test_annotations.py`

- [ ] **Step 1: テストを書く**

`tests/annotator/test_annotations.py`:
```python
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
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
docker compose exec python pytest tests/annotator/test_annotations.py -v
```

Expected: 3 FAILED

- [ ] **Step 3: routes/annotations.py を作成**

`annotator/backend/routes/annotations.py`:
```python
import json
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional
from annotator.backend.config import get_config, Config

router = APIRouter()


class BboxItem(BaseModel):
    id: str
    x_center: float
    y_center: float
    width: float
    height: float
    category: str
    vlm_text: Optional[str] = None
    vlm_shape: Optional[str] = None


class AnnotationData(BaseModel):
    image: str
    connectors: list[BboxItem]


@router.get("/annotations/{filename}")
def get_annotation(filename: str, config: Config = Depends(get_config)):
    path = config.data_dir / "ground_truth" / f"{filename}.json"
    if not path.exists():
        return {"image": filename, "connectors": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.post("/annotations/{filename}")
def save_annotation(
    filename: str,
    data: AnnotationData,
    config: Config = Depends(get_config),
):
    gt_dir = config.data_dir / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    path = gt_dir / f"{filename}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data.model_dump(), f, ensure_ascii=False, indent=2)
    return {"saved": str(path)}
```

- [ ] **Step 4: テストが通ることを確認**

```bash
docker compose exec python pytest tests/annotator/test_annotations.py -v
```

Expected: 3 PASSED

- [ ] **Step 5: コミット**

```bash
git add annotator/backend/routes/annotations.py tests/annotator/test_annotations.py
git commit -m "feat: add annotation read/write endpoints"
```

---

## Task 5: VLM 推論 API

**Files:**
- Create: `annotator/backend/routes/infer.py`
- Create: `tests/annotator/test_infer.py`

- [ ] **Step 1: テストを書く**

`tests/annotator/test_infer.py`:
```python
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
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
docker compose exec python pytest tests/annotator/test_infer.py -v
```

Expected: FAILED または ImportError

- [ ] **Step 3: routes/infer.py を作成**

`annotator/backend/routes/infer.py`:
```python
import base64
import io
import json
import re
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException
from PIL import Image
from pydantic import BaseModel

from annotator.backend.config import get_config, Config

router = APIRouter()


class BboxCoords(BaseModel):
    x_center: float
    y_center: float
    width: float
    height: float


class InferRequest(BaseModel):
    image: str
    bbox: BboxCoords


def _crop_to_base64(image_path: Path, bbox: BboxCoords) -> str:
    with Image.open(image_path) as img:
        w, h = img.size
        x1 = int((bbox.x_center - bbox.width / 2) * w)
        y1 = int((bbox.y_center - bbox.height / 2) * h)
        x2 = int((bbox.x_center + bbox.width / 2) * w)
        y2 = int((bbox.y_center + bbox.height / 2) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        cropped = img.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()


def _parse_vlm_response(text: str) -> dict:
    match = re.search(r'\{[^{}]*"vlm_text"[^{}]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"vlm_text": "", "vlm_shape": ""}


@router.post("/infer")
def infer_bbox(req: InferRequest, config: Config = Depends(get_config)):
    image_path = config.data_dir / "samples" / req.image
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    image_b64 = _crop_to_base64(image_path, req.bbox)

    prompt_path = Path(config.prompt_file)
    prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""

    with httpx.Client() as client:
        resp = client.post(
            f"{config.ollama_base_url}/api/generate",
            json={
                "model": config.model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
            },
            timeout=60.0,
        )
        resp.raise_for_status()

    result = _parse_vlm_response(resp.json().get("response", ""))
    return result
```

- [ ] **Step 4: `respx` を python コンテナに追加**

`docker/python/requirements.txt` に追記:
```
fastapi>=0.111.0
httpx>=0.27.0
respx>=0.21.0
python-multipart>=0.0.9
```

その後コンテナを再ビルド:
```bash
docker compose build python
```

- [ ] **Step 5: テストが通ることを確認**

```bash
docker compose exec python pytest tests/annotator/test_infer.py -v
```

Expected: 3 PASSED

- [ ] **Step 6: バックエンド全テストが通ることを確認**

```bash
docker compose exec python pytest tests/annotator/ -v
```

Expected: 全 PASSED

- [ ] **Step 7: コミット**

```bash
git add annotator/backend/routes/infer.py tests/annotator/test_infer.py docker/python/requirements.txt
git commit -m "feat: add VLM inference endpoint with bbox crop"
```

---

## Task 6: Docker セットアップ

**Files:**
- Create: `docker/annotator/Dockerfile`
- Create: `docker/annotator/requirements.txt`
- Modify: `docker-compose.yml`

- [ ] **Step 1: annotator コンテナの requirements.txt を作成**

`docker/annotator/requirements.txt`:
```
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
httpx>=0.27.0
pillow>=10.0.0
python-multipart>=0.0.9
```

- [ ] **Step 2: Dockerfile を作成**

`docker/annotator/Dockerfile`:
```dockerfile
# Stage 1: React ビルド
FROM node:20-slim AS frontend-builder
WORKDIR /frontend
COPY annotator/frontend/package*.json ./
RUN npm ci
COPY annotator/frontend/ ./
RUN npm run build

# Stage 2: FastAPI サーバー
FROM python:3.11-slim
WORKDIR /app
COPY docker/annotator/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY annotator/ ./annotator/
COPY --from=frontend-builder /frontend/dist/ ./annotator/backend/static/
ENV PYTHONPATH=/app
CMD ["uvicorn", "annotator.backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

- [ ] **Step 3: docker-compose.yml に annotator サービスを追加**

`docker-compose.yml` の `volumes:` セクションの直前に追加:
```yaml
  annotator:
    build:
      context: .
      dockerfile: docker/annotator/Dockerfile
    container_name: annotator
    restart: unless-stopped
    ports:
      - "${ANNOTATOR_PORT:-8000}:8000"
    volumes:
      - ./data:/app/data
      - ./prompts:/app/prompts
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - MODEL=${MODEL:-gemma4:26b}
      - DATA_DIR=/app/data
      - ANNOTATOR_PROMPT_FILE=/app/prompts/annotator/v1.txt
    depends_on:
      - ollama
```

- [ ] **Step 4: フロントエンドが存在しない状態でもビルドが通るか確認するため、ダミーの dist を作成してビルドテスト**

```bash
mkdir -p annotator/frontend
echo '{"name":"annotator","version":"0.0.0","scripts":{"build":"echo build skipped","dev":"echo"}}' > annotator/frontend/package.json
docker compose build annotator
```

Expected: ビルド成功（後の Task でフロントエンドを実装する）

- [ ] **Step 5: コミット**

```bash
git add docker/annotator/ docker-compose.yml
git commit -m "feat: add annotator Docker service to docker-compose"
```

---

## Task 7: React プロジェクト初期化

**Files:**
- Modify: `annotator/frontend/package.json`
- Create: `annotator/frontend/vite.config.js`
- Create: `annotator/frontend/index.html`
- Create: `annotator/frontend/src/main.jsx`

- [ ] **Step 1: package.json を作成（Task 6 のダミーを上書き）**

`annotator/frontend/package.json`:
```json
{
  "name": "annotator-frontend",
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "axios": "^1.7.0",
    "react": "^18.3.0",
    "react-dom": "^18.3.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.3.0",
    "vite": "^5.3.0"
  }
}
```

- [ ] **Step 2: vite.config.js を作成**

`annotator/frontend/vite.config.js`:
```js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000'
    }
  }
})
```

- [ ] **Step 3: index.html を作成**

`annotator/frontend/index.html`:
```html
<!doctype html>
<html lang="ja">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Annotation Tool</title>
    <style>
      * { box-sizing: border-box; margin: 0; padding: 0; }
      body { font-family: sans-serif; background: #1a1a1a; color: #eee; }
    </style>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

- [ ] **Step 4: main.jsx を作成**

`annotator/frontend/src/main.jsx`:
```jsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>
)
```

- [ ] **Step 5: Ubuntu で npm install を実行してビルドが通ることを確認**

```bash
cd annotator/frontend
npm install
npm run build
cd ../..
```

Expected: `dist/` が生成される

- [ ] **Step 6: コミット**

```bash
git add annotator/frontend/package.json annotator/frontend/vite.config.js annotator/frontend/index.html annotator/frontend/src/main.jsx annotator/frontend/package-lock.json
git commit -m "feat: initialize React/Vite frontend project"
```

---

## Task 8: API クライアント

**Files:**
- Create: `annotator/frontend/src/api.js`

- [ ] **Step 1: api.js を作成**

`annotator/frontend/src/api.js`:
```js
import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export const fetchImages = () =>
  api.get('/images').then(r => r.data.images)

export const fetchImageUrl = (filename) =>
  `/api/images/${encodeURIComponent(filename)}`

export const fetchAnnotation = (filename) =>
  api.get(`/annotations/${encodeURIComponent(filename)}`).then(r => r.data)

export const saveAnnotation = (filename, data) =>
  api.post(`/annotations/${encodeURIComponent(filename)}`, data).then(r => r.data)

export const inferBbox = (image, bbox) =>
  api.post('/infer', { image, bbox }).then(r => r.data)
```

- [ ] **Step 2: コミット**

```bash
git add annotator/frontend/src/api.js
git commit -m "feat: add API client module"
```

---

## Task 9: ImageSelector コンポーネント

**Files:**
- Create: `annotator/frontend/src/components/ImageSelector.jsx`

- [ ] **Step 1: ImageSelector.jsx を作成**

`annotator/frontend/src/components/ImageSelector.jsx`:
```jsx
export default function ImageSelector({ images, selected, onChange }) {
  return (
    <div style={{ padding: '8px', background: '#2a2a2a', display: 'flex', alignItems: 'center', gap: '8px' }}>
      <label style={{ fontSize: '14px' }}>画像:</label>
      <select
        value={selected}
        onChange={e => onChange(e.target.value)}
        style={{ padding: '4px 8px', background: '#333', color: '#eee', border: '1px solid #555', borderRadius: '4px' }}
      >
        <option value="">-- 選択してください --</option>
        {images.map(img => (
          <option key={img} value={img}>{img}</option>
        ))}
      </select>
    </div>
  )
}
```

- [ ] **Step 2: コミット**

```bash
git add annotator/frontend/src/components/ImageSelector.jsx
git commit -m "feat: add ImageSelector component"
```

---

## Task 10: AnnotationCanvas コンポーネント

**Files:**
- Create: `annotator/frontend/src/components/AnnotationCanvas.jsx`

- [ ] **Step 1: AnnotationCanvas.jsx を作成**

`annotator/frontend/src/components/AnnotationCanvas.jsx`:
```jsx
import { useRef, useEffect, useState, useCallback } from 'react'

export default function AnnotationCanvas({ imageSrc, bboxes, onBboxAdd }) {
  const canvasRef = useRef(null)
  const imageRef = useRef(null)
  const [drawing, setDrawing] = useState(false)
  const [startPos, setStartPos] = useState(null)
  const [currentPos, setCurrentPos] = useState(null)

  const redraw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || !imageRef.current) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(imageRef.current, 0, 0, canvas.width, canvas.height)

    bboxes.forEach((bbox, i) => {
      const x = (bbox.x_center - bbox.width / 2) * canvas.width
      const y = (bbox.y_center - bbox.height / 2) * canvas.height
      const w = bbox.width * canvas.width
      const h = bbox.height * canvas.height
      ctx.strokeStyle = '#00e676'
      ctx.lineWidth = 2
      ctx.strokeRect(x, y, w, h)
      ctx.fillStyle = '#00e676'
      ctx.font = 'bold 13px sans-serif'
      ctx.fillText(String(i + 1), x + 3, y + 15)
    })

    if (drawing && startPos && currentPos) {
      ctx.strokeStyle = '#ff5252'
      ctx.lineWidth = 2
      ctx.setLineDash([5, 3])
      const rx = Math.min(startPos.x, currentPos.x)
      const ry = Math.min(startPos.y, currentPos.y)
      const rw = Math.abs(currentPos.x - startPos.x)
      const rh = Math.abs(currentPos.y - startPos.y)
      ctx.strokeRect(rx, ry, rw, rh)
      ctx.setLineDash([])
    }
  }, [bboxes, drawing, startPos, currentPos])

  useEffect(() => {
    if (!imageSrc) return
    const img = new Image()
    img.onload = () => {
      imageRef.current = img
      const canvas = canvasRef.current
      if (!canvas) return
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight
      redraw()
    }
    img.src = imageSrc
  }, [imageSrc])

  useEffect(() => { redraw() }, [redraw])

  const getPos = (e) => {
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    return {
      x: (e.clientX - rect.left) * (canvas.width / rect.width),
      y: (e.clientY - rect.top) * (canvas.height / rect.height),
    }
  }

  const handleMouseDown = (e) => {
    setDrawing(true)
    const pos = getPos(e)
    setStartPos(pos)
    setCurrentPos(pos)
  }

  const handleMouseMove = (e) => {
    if (!drawing) return
    setCurrentPos(getPos(e))
  }

  const handleMouseUp = (e) => {
    if (!drawing || !startPos) return
    const endPos = getPos(e)
    const canvas = canvasRef.current
    const x1 = Math.min(startPos.x, endPos.x)
    const y1 = Math.min(startPos.y, endPos.y)
    const x2 = Math.max(startPos.x, endPos.x)
    const y2 = Math.max(startPos.y, endPos.y)
    if (x2 - x1 > 5 && y2 - y1 > 5) {
      onBboxAdd({
        x_center: (x1 + x2) / 2 / canvas.width,
        y_center: (y1 + y2) / 2 / canvas.height,
        width: (x2 - x1) / canvas.width,
        height: (y2 - y1) / canvas.height,
      })
    }
    setDrawing(false)
    setStartPos(null)
    setCurrentPos(null)
  }

  if (!imageSrc) {
    return (
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#666' }}>
        画像を選択してください
      </div>
    )
  }

  return (
    <canvas
      ref={canvasRef}
      style={{ cursor: 'crosshair', maxWidth: '100%', maxHeight: '100%', display: 'block' }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    />
  )
}
```

- [ ] **Step 2: コミット**

```bash
git add annotator/frontend/src/components/AnnotationCanvas.jsx
git commit -m "feat: add AnnotationCanvas component with bbox drawing"
```

---

## Task 11: BboxList コンポーネント

**Files:**
- Create: `annotator/frontend/src/components/BboxList.jsx`

- [ ] **Step 1: BboxList.jsx を作成**

`annotator/frontend/src/components/BboxList.jsx`:
```jsx
export default function BboxList({ bboxes, onCategoryChange, onInfer, onDelete, inferring }) {
  if (bboxes.length === 0) {
    return (
      <div style={{ padding: '16px', color: '#666', fontSize: '13px' }}>
        Canvas 上でドラッグして bbox を描いてください
      </div>
    )
  }

  return (
    <div style={{ overflowY: 'auto', flex: 1 }}>
      {bboxes.map((bbox, i) => (
        <div key={bbox.id} style={{
          padding: '10px',
          borderBottom: '1px solid #333',
          background: '#2a2a2a',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
            <span style={{ fontWeight: 'bold', color: '#00e676' }}>bbox-{i + 1}</span>
            <button
              onClick={() => onDelete(bbox.id)}
              style={{ background: 'none', border: 'none', color: '#f44336', cursor: 'pointer', fontSize: '16px' }}
            >×</button>
          </div>

          <div style={{ marginBottom: '6px' }}>
            <label style={{ fontSize: '12px', color: '#aaa' }}>カテゴリ</label>
            <input
              type="text"
              value={bbox.category}
              onChange={e => onCategoryChange(bbox.id, e.target.value)}
              placeholder="例: terminal"
              style={{
                width: '100%', marginTop: '2px', padding: '4px 6px',
                background: '#333', color: '#eee', border: '1px solid #555', borderRadius: '4px',
                fontSize: '13px',
              }}
            />
          </div>

          <button
            onClick={() => onInfer(bbox)}
            disabled={inferring === bbox.id}
            style={{
              width: '100%', padding: '5px',
              background: inferring === bbox.id ? '#555' : '#1565c0',
              color: '#fff', border: 'none', borderRadius: '4px',
              cursor: inferring === bbox.id ? 'not-allowed' : 'pointer',
              fontSize: '13px', marginBottom: '6px',
            }}
          >
            {inferring === bbox.id ? '推論中...' : 'VLM 推論'}
          </button>

          {(bbox.vlm_text !== null || bbox.vlm_shape !== null) && (
            <div style={{ fontSize: '12px', color: '#bbb', background: '#1a1a1a', padding: '6px', borderRadius: '4px' }}>
              <div>テキスト: <span style={{ color: '#fff' }}>{bbox.vlm_text ?? '—'}</span></div>
              <div>形状: <span style={{ color: '#fff' }}>{bbox.vlm_shape ?? '—'}</span></div>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
```

- [ ] **Step 2: コミット**

```bash
git add annotator/frontend/src/components/BboxList.jsx
git commit -m "feat: add BboxList component with inference trigger"
```

---

## Task 12: App 統合とフロントエンドビルド

**Files:**
- Create: `annotator/frontend/src/App.jsx`

- [ ] **Step 1: App.jsx を作成**

`annotator/frontend/src/App.jsx`:
```jsx
import { useState, useEffect, useCallback } from 'react'
import ImageSelector from './components/ImageSelector.jsx'
import AnnotationCanvas from './components/AnnotationCanvas.jsx'
import BboxList from './components/BboxList.jsx'
import { fetchImages, fetchImageUrl, fetchAnnotation, saveAnnotation, inferBbox } from './api.js'

let bboxCounter = 0
const newId = () => `bbox-${++bboxCounter}`

export default function App() {
  const [images, setImages] = useState([])
  const [selected, setSelected] = useState('')
  const [bboxes, setBboxes] = useState([])
  const [inferring, setInferring] = useState(null)
  const [saveStatus, setSaveStatus] = useState('')

  useEffect(() => {
    fetchImages().then(setImages).catch(console.error)
  }, [])

  const handleImageChange = useCallback(async (filename) => {
    setSelected(filename)
    setBboxes([])
    if (!filename) return
    const data = await fetchAnnotation(filename)
    setBboxes(data.connectors.map(c => ({ ...c, id: c.id || newId() })))
  }, [])

  const handleBboxAdd = useCallback((coords) => {
    setBboxes(prev => [...prev, { id: newId(), ...coords, category: '', vlm_text: null, vlm_shape: null }])
  }, [])

  const handleCategoryChange = useCallback((id, value) => {
    setBboxes(prev => prev.map(b => b.id === id ? { ...b, category: value } : b))
  }, [])

  const handleDelete = useCallback((id) => {
    setBboxes(prev => prev.filter(b => b.id !== id))
  }, [])

  const handleInfer = useCallback(async (bbox) => {
    setInferring(bbox.id)
    try {
      const result = await inferBbox(selected, {
        x_center: bbox.x_center,
        y_center: bbox.y_center,
        width: bbox.width,
        height: bbox.height,
      })
      setBboxes(prev => prev.map(b =>
        b.id === bbox.id ? { ...b, vlm_text: result.vlm_text, vlm_shape: result.vlm_shape } : b
      ))
    } catch (e) {
      console.error('Inference failed:', e)
    } finally {
      setInferring(null)
    }
  }, [selected])

  const handleSave = useCallback(async () => {
    if (!selected) return
    setSaveStatus('保存中...')
    try {
      await saveAnnotation(selected, { image: selected, connectors: bboxes })
      setSaveStatus('保存完了')
    } catch (e) {
      setSaveStatus('保存失敗')
      console.error(e)
    }
    setTimeout(() => setSaveStatus(''), 2000)
  }, [selected, bboxes])

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 8px' }}>
        <ImageSelector images={images} selected={selected} onChange={handleImageChange} />
        <button
          onClick={handleSave}
          disabled={!selected}
          style={{
            padding: '6px 16px', background: selected ? '#2e7d32' : '#555',
            color: '#fff', border: 'none', borderRadius: '4px', cursor: selected ? 'pointer' : 'not-allowed',
          }}
        >
          {saveStatus || '保存'}
        </button>
      </div>

      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        <div style={{ flex: 1, overflow: 'auto', padding: '8px', display: 'flex', alignItems: 'flex-start' }}>
          <AnnotationCanvas
            imageSrc={selected ? fetchImageUrl(selected) : ''}
            bboxes={bboxes}
            onBboxAdd={handleBboxAdd}
          />
        </div>
        <div style={{ width: '260px', borderLeft: '1px solid #333', display: 'flex', flexDirection: 'column' }}>
          <div style={{ padding: '8px', borderBottom: '1px solid #333', fontSize: '13px', color: '#aaa' }}>
            Bbox 一覧 ({bboxes.length})
          </div>
          <BboxList
            bboxes={bboxes}
            onCategoryChange={handleCategoryChange}
            onInfer={handleInfer}
            onDelete={handleDelete}
            inferring={inferring}
          />
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: フロントエンドをビルド**

```bash
cd annotator/frontend
npm run build
cd ../..
```

Expected: `annotator/frontend/dist/` が生成される

- [ ] **Step 3: Docker イメージをビルド**

```bash
docker compose build annotator
```

Expected: ビルド成功

- [ ] **Step 4: 動作確認**

```bash
docker compose up -d annotator
```

ブラウザで `http://localhost:8000` を開く

確認項目:
- [ ] アノテーション画面が表示される
- [ ] `data/samples/` の画像がドロップダウンに表示される
- [ ] 画像を選択するとキャンバスに表示される
- [ ] ドラッグで bbox が描ける（緑の枠）
- [ ] 「VLM 推論」ボタンでテキスト・形状が表示される（ollama が起動していること）
- [ ] 「保存」で `data/ground_truth/<filename>.json` が生成される
- [ ] 同じ画像を再選択すると bbox が復元される

- [ ] **Step 5: コミット**

```bash
git add annotator/frontend/src/App.jsx annotator/frontend/dist/
git commit -m "feat: complete annotation tool frontend integration"
```

---

## Task 13: バックエンド全テストとクリーンアップ

- [ ] **Step 1: 全テストを実行**

```bash
docker compose exec python pytest tests/ -v
```

Expected: 既存テスト含め全 PASSED

- [ ] **Step 2: CLAUDE.md を更新**

`CLAUDE.md` に以下を追記:
```markdown
## アノテーションツール

`annotator/` ディレクトリに FastAPI + React 製のアノテーション Web アプリがある。

- 起動: `docker compose up annotator`
- アクセス: `http://localhost:8000`
- バックエンドテスト: `docker compose exec python pytest tests/annotator/ -v`
- フロントエンドビルド: `cd annotator/frontend && npm run build`
- フロントエンドを変更したら `docker compose build annotator` が必要
```

- [ ] **Step 3: 最終コミット**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with annotator tool instructions"
```
