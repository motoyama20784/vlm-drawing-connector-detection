# 実験パイプライン実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** VLMコネクタ検出の実験基盤を構築し、プロンプト・モデル・パラメータの違いをMLflowで記録・比較できる状態にする。

**Architecture:** Ollama（推論）・MLflow（実験記録）・Python（実験スクリプト）を3コンテナ構成で動かす。Pythonコンテナから Ollama API と MLflow tracking server を叩き、バウンディングボックス検出の精度（Precision/Recall/F1）を実験ごとに記録する。

**Tech Stack:** Docker Compose 2.x, Ollama, MLflow 2.x, Python 3.11, requests, PyYAML, pytest

---

## ファイルマップ

| ファイル | 役割 |
|---|---|
| `docker-compose.yml` | 全サービス統合（新規作成） |
| `docker/python/Dockerfile` | Python実験コンテナ |
| `docker/python/requirements.txt` | Pythonパッケージ一覧 |
| `docker/ollama/.env.example` | Ollama設定テンプレート（移動） |
| `docker/ollama/Makefile` | Ollama操作コマンド（移動） |
| `docker/mlflow/.env.example` | MLflow設定テンプレート |
| `src/__init__.py` | パッケージ宣言 |
| `src/detection/__init__.py` | パッケージ宣言 |
| `src/detection/detector.py` | Ollama APIを叩いてVLM推論 |
| `src/detection/parser.py` | VLMテキスト出力→バウンディングボックス |
| `src/detection/evaluator.py` | IoU計算・Precision/Recall/F1 |
| `src/experiment/__init__.py` | パッケージ宣言 |
| `src/experiment/runner.py` | MLflow実験の記録・管理 |
| `prompts/zero_shot/v1.txt` | ゼロショットプロンプト |
| `prompts/few_shot/v1.txt` | フューショットプロンプト |
| `configs/experiment.yaml` | 実験設定ファイル |
| `tests/detection/test_parser.py` | parserのユニットテスト |
| `tests/detection/test_evaluator.py` | evaluatorのユニットテスト |
| `tests/detection/test_detector.py` | detectorのユニットテスト |
| `tests/experiment/test_runner.py` | runnerのユニットテスト |

---

## Task 1: ディレクトリ構造の整備とファイル移行

**Files:**
- Move: `src/serving/ollama/.env.example` → `docker/ollama/.env.example`
- Move: `src/serving/ollama/Makefile` → `docker/ollama/Makefile`
- Delete: `src/serving/ollama/docker-compose.yml`
- Delete: `src/serving/ollama/.env.example`（移動後）
- Delete: `src/serving/ollama/Makefile`（移動後）

- [ ] **Step 1: 新しいディレクトリを作成する**

Ubuntu側で実行：
```bash
mkdir -p docker/ollama docker/mlflow docker/python
mkdir -p src/detection src/experiment
mkdir -p tests/detection tests/experiment
mkdir -p prompts/zero_shot prompts/few_shot
mkdir -p configs
mkdir -p data/samples data/ground_truth data/outputs
mkdir -p mlflow/mlruns
```

- [ ] **Step 2: 既存ファイルを移動し、Makefileを更新する**

Ubuntu側で実行：
```bash
cp src/serving/ollama/.env.example docker/ollama/.env.example
```

`docker/ollama/Makefile` を作成（docker-compose.ymlがルートに移動したため内容を更新）：
```makefile
MODEL ?= gemma4:26b
ROOT := $(shell git rev-parse --show-toplevel)

.PHONY: up down pull logs ps shell

up:
	docker compose -f $(ROOT)/docker-compose.yml up -d

down:
	docker compose -f $(ROOT)/docker-compose.yml down

pull:
	docker compose -f $(ROOT)/docker-compose.yml exec ollama ollama pull $(MODEL)

logs:
	docker compose -f $(ROOT)/docker-compose.yml logs -f

ps:
	docker compose -f $(ROOT)/docker-compose.yml ps

shell:
	docker compose -f $(ROOT)/docker-compose.yml exec ollama bash
```

- [ ] **Step 3: 古いファイルを削除する**

Ubuntu側で実行：
```bash
rm src/serving/ollama/docker-compose.yml
rm src/serving/ollama/.env.example
rm src/serving/ollama/Makefile
rmdir src/serving/ollama
rmdir src/serving
```

- [ ] **Step 4: .gitkeep でディレクトリを記録する**

Ubuntu側で実行：
```bash
touch data/samples/.gitkeep data/ground_truth/.gitkeep data/outputs/.gitkeep
touch mlflow/mlruns/.gitkeep
touch tests/detection/.gitkeep tests/experiment/.gitkeep
```

- [ ] **Step 5: .gitignore を作成する**

`.gitignore` を作成：
```
# MLflow 実行時生成ファイル
mlflow/mlflow.db
mlflow/artifacts/

# Python
__pycache__/
*.pyc
.pytest_cache/

# 環境変数
.env

# データ（大容量ファイルはGit管理外）
data/samples/*
data/outputs/*
!data/samples/.gitkeep
!data/outputs/.gitkeep
```

- [ ] **Step 6: Commit**

```bash
git add .
git commit -m "refactor: ディレクトリ構造を整備、docker/配下に集約"
```

---

## Task 2: Docker環境の構築

**Files:**
- Create: `docker/python/requirements.txt`
- Create: `docker/python/Dockerfile`
- Create: `docker/mlflow/.env.example`
- Create: `docker-compose.yml`（ルート）

- [ ] **Step 1: requirements.txt を作成する**

`docker/python/requirements.txt` を作成：
```
requests>=2.32.0
openai>=1.50.0
mlflow>=2.14.0
pyyaml>=6.0.1
pytest>=8.2.0
```

- [ ] **Step 2: Dockerfile を作成する**

`docker/python/Dockerfile` を作成：
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY docker/python/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ENV PYTHONPATH=/app

CMD ["bash"]
```

- [ ] **Step 3: docker/mlflow/.env.example を作成する**

`docker/mlflow/.env.example` を作成：
```
MLFLOW_PORT=5000
```

- [ ] **Step 4: ルートの docker-compose.yml を作成する**

`docker-compose.yml` を作成：
```yaml
services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    restart: unless-stopped
    ports:
      - "${OLLAMA_PORT:-11434}:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - OLLAMA_HOST=0.0.0.0
      - OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE:-5m}

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.14.0
    container_name: mlflow
    restart: unless-stopped
    ports:
      - "${MLFLOW_PORT:-5000}:5000"
    volumes:
      - ./mlflow:/mlflow
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri sqlite:////mlflow/mlflow.db
      --default-artifact-root /mlflow/artifacts

  python:
    build:
      context: .
      dockerfile: docker/python/Dockerfile
    container_name: python_dev
    depends_on:
      - ollama
      - mlflow
    volumes:
      - .:/app
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - MODEL=${MODEL:-gemma4:26b}
      - PYTHONPATH=/app
    stdin_open: true
    tty: true

volumes:
  ollama_data:
    driver: local
```

- [ ] **Step 5: 全サービスを起動して確認する**

Ubuntu側で実行（プロジェクトルートから）：
```bash
docker compose up -d
```

期待される出力：
```
✔ Container ollama    Started
✔ Container mlflow    Started
✔ Container python_dev Started
```

- [ ] **Step 6: MLflow UI にアクセスできるか確認する**

Ubuntu側で実行：
```bash
curl http://localhost:5000/health
```

期待される出力：
```
OK
```

- [ ] **Step 7: Commit**

```bash
git add docker-compose.yml docker/python/ docker/mlflow/
git commit -m "feat: Docker 3サービス構成を構築 (ollama/mlflow/python)"
```

---

## Task 3: src/detection/parser.py の実装

**Files:**
- Create: `src/__init__.py`
- Create: `src/detection/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/detection/__init__.py`
- Create: `tests/detection/test_parser.py`
- Create: `src/detection/parser.py`

- [ ] **Step 1: __init__.py を作成する**

以下の空ファイルを作成：
- `src/__init__.py`（空）
- `src/detection/__init__.py`（空）
- `tests/__init__.py`（空）
- `tests/detection/__init__.py`（空）

- [ ] **Step 2: 失敗するテストを書く**

`tests/detection/test_parser.py` を作成：
```python
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
```

- [ ] **Step 3: テストが失敗することを確認する**

Ubuntu側で実行：
```bash
docker compose run --rm python pytest tests/detection/test_parser.py -v
```

期待される出力：
```
ERROR: cannot import name 'parse_connectors' from 'src.detection.parser'
```

- [ ] **Step 4: parser.py を実装する**

`src/detection/parser.py` を作成：
```python
import json


def parse_connectors(text: str) -> list[dict]:
    start = text.find('{"connectors"')
    if start == -1:
        start = text.find('{ "connectors"')
    if start == -1:
        return []

    depth = 0
    for i, char in enumerate(text[start:]):
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                json_str = text[start:start + i + 1]
                try:
                    data = json.loads(json_str)
                    return data.get("connectors", [])
                except json.JSONDecodeError:
                    return []
    return []
```

- [ ] **Step 5: テストが通ることを確認する**

Ubuntu側で実行：
```bash
docker compose run --rm python pytest tests/detection/test_parser.py -v
```

期待される出力：
```
tests/detection/test_parser.py::test_parse_valid_json PASSED
tests/detection/test_parser.py::test_parse_json_embedded_in_text PASSED
tests/detection/test_parser.py::test_parse_no_connectors PASSED
tests/detection/test_parser.py::test_parse_invalid_output_returns_empty PASSED
tests/detection/test_parser.py::test_parse_multiple_connectors PASSED
5 passed
```

- [ ] **Step 6: Commit**

```bash
git add src/__init__.py src/detection/__init__.py src/detection/parser.py tests/__init__.py tests/detection/__init__.py tests/detection/test_parser.py
git commit -m "feat: VLMテキスト出力からバウンディングボックスを抽出するparser実装"
```

---

## Task 4: src/detection/evaluator.py の実装

**Files:**
- Create: `tests/detection/test_evaluator.py`
- Create: `src/detection/evaluator.py`

- [ ] **Step 1: 失敗するテストを書く**

`tests/detection/test_evaluator.py` を作成：
```python
import pytest
from src.detection.evaluator import compute_iou, evaluate


def test_iou_perfect_match():
    box = {"x_center": 0.5, "y_center": 0.5, "width": 0.1, "height": 0.1}
    assert compute_iou(box, box) == 1.0


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
```

- [ ] **Step 2: テストが失敗することを確認する**

Ubuntu側で実行：
```bash
docker compose run --rm python pytest tests/detection/test_evaluator.py -v
```

期待される出力：
```
ERROR: cannot import name 'compute_iou' from 'src.detection.evaluator'
```

- [ ] **Step 3: evaluator.py を実装する**

`src/detection/evaluator.py` を作成：
```python
def compute_iou(box1: dict, box2: dict) -> float:
    x1_1 = box1["x_center"] - box1["width"] / 2
    y1_1 = box1["y_center"] - box1["height"] / 2
    x2_1 = box1["x_center"] + box1["width"] / 2
    y2_1 = box1["y_center"] + box1["height"] / 2

    x1_2 = box2["x_center"] - box2["width"] / 2
    y1_2 = box2["y_center"] - box2["height"] / 2
    x2_2 = box2["x_center"] + box2["width"] / 2
    y2_2 = box2["y_center"] + box2["height"] / 2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = box1["width"] * box1["height"]
    area2 = box2["width"] * box2["height"]
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def evaluate(pred_boxes: list, gt_boxes: list, iou_threshold: float = 0.5) -> dict:
    if not gt_boxes and not pred_boxes:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    if not gt_boxes:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}

    if not pred_boxes:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    matched_gt = set()
    tp = 0

    for pred in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}
```

- [ ] **Step 4: テストが通ることを確認する**

Ubuntu側で実行：
```bash
docker compose run --rm python pytest tests/detection/test_evaluator.py -v
```

期待される出力：
```
7 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/detection/evaluator.py tests/detection/test_evaluator.py
git commit -m "feat: IoUベースのバウンディングボックス評価ロジック実装"
```

---

## Task 5: src/detection/detector.py の実装

**Files:**
- Create: `tests/detection/test_detector.py`
- Create: `src/detection/detector.py`

- [ ] **Step 1: 失敗するテストを書く**

`tests/detection/test_detector.py` を作成：
```python
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
```

- [ ] **Step 2: テストが失敗することを確認する**

Ubuntu側で実行：
```bash
docker compose run --rm python pytest tests/detection/test_detector.py -v
```

期待される出力：
```
ERROR: cannot import name 'detect' from 'src.detection.detector'
```

- [ ] **Step 3: detector.py を実装する**

`src/detection/detector.py` を作成：
```python
import base64
import os

import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def detect(image_path: str, prompt: str, model: str, params: dict) -> str:
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": params.get("temperature", 0.0),
            "top_p": params.get("top_p", 0.9),
            "top_k": params.get("top_k", 40),
        },
    }

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]
```

- [ ] **Step 4: テストが通ることを確認する**

Ubuntu側で実行：
```bash
docker compose run --rm python pytest tests/detection/test_detector.py -v
```

期待される出力：
```
2 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/detection/detector.py tests/detection/test_detector.py
git commit -m "feat: Ollama APIへの画像+テキスト推論リクエスト実装"
```

---

## Task 6: src/experiment/runner.py の実装

**Files:**
- Create: `src/experiment/__init__.py`
- Create: `tests/experiment/__init__.py`
- Create: `tests/experiment/test_runner.py`
- Create: `src/experiment/runner.py`

- [ ] **Step 1: __init__.py を作成する**

以下の空ファイルを作成：
- `src/experiment/__init__.py`（空）
- `tests/experiment/__init__.py`（空）

- [ ] **Step 2: 失敗するテストを書く**

`tests/experiment/test_runner.py` を作成：
```python
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
```

- [ ] **Step 3: テストが失敗することを確認する**

Ubuntu側で実行：
```bash
docker compose run --rm python pytest tests/experiment/test_runner.py -v
```

期待される出力：
```
ERROR: cannot import name 'load_ground_truth' from 'src.experiment.runner'
```

- [ ] **Step 4: runner.py を実装する**

`src/experiment/runner.py` を作成：
```python
import argparse
import json
import os
from pathlib import Path

import mlflow
import yaml

from src.detection.detector import detect
from src.detection.evaluator import evaluate
from src.detection.parser import parse_connectors

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def load_ground_truth(gt_dir: str, image_name: str) -> list:
    gt_path = Path(gt_dir) / (Path(image_name).stem + ".json")
    if not gt_path.exists():
        return []
    with open(gt_path) as f:
        data = json.load(f)
    return data.get("connectors", [])


def run_experiment(config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    with open(config["prompt"]["file"]) as f:
        prompt_text = f.read()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("connector-detection")

    with mlflow.start_run():
        mlflow.log_param("model", config["model"])
        mlflow.log_param("prompt_file", config["prompt"]["file"])
        mlflow.log_param("prompt_type", config["prompt"]["type"])
        mlflow.log_param("temperature", config["params"]["temperature"])
        mlflow.log_param("top_p", config["params"]["top_p"])
        mlflow.log_param("top_k", config["params"]["top_k"])
        mlflow.log_param("iou_threshold", config["evaluation"]["iou_threshold"])
        mlflow.log_text(prompt_text, "prompt.txt")

        samples_dir = config["data"]["samples_dir"]
        gt_dir = config["data"]["ground_truth_dir"]
        image_paths = sorted(
            list(Path(samples_dir).glob("*.png"))
            + list(Path(samples_dir).glob("*.jpg"))
        )

        all_metrics = []
        for image_path in image_paths:
            raw_response = detect(
                str(image_path),
                prompt_text,
                config["model"],
                config["params"],
            )
            pred_boxes = parse_connectors(raw_response)
            gt_boxes = load_ground_truth(gt_dir, image_path.name)
            metrics = evaluate(
                pred_boxes, gt_boxes, config["evaluation"]["iou_threshold"]
            )

            stem = image_path.stem
            mlflow.log_metric(f"precision_{stem}", metrics["precision"])
            mlflow.log_metric(f"recall_{stem}", metrics["recall"])
            mlflow.log_metric(f"f1_{stem}", metrics["f1"])
            mlflow.log_text(raw_response, f"responses/{stem}.txt")
            all_metrics.append(metrics)
            print(f"{image_path.name}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}")

        if all_metrics:
            avg_p = sum(m["precision"] for m in all_metrics) / len(all_metrics)
            avg_r = sum(m["recall"] for m in all_metrics) / len(all_metrics)
            avg_f1 = sum(m["f1"] for m in all_metrics) / len(all_metrics)
            mlflow.log_metric("precision", avg_p)
            mlflow.log_metric("recall", avg_r)
            mlflow.log_metric("f1", avg_f1)
            print(f"\n平均: P={avg_p:.3f} R={avg_r:.3f} F1={avg_f1:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="実験設定ファイルのパス")
    args = parser.parse_args()
    run_experiment(args.config)
```

- [ ] **Step 5: テストが通ることを確認する**

Ubuntu側で実行：
```bash
docker compose run --rm python pytest tests/experiment/test_runner.py -v
```

期待される出力：
```
3 passed
```

- [ ] **Step 6: 全テストが通ることを確認する**

Ubuntu側で実行：
```bash
docker compose run --rm python pytest tests/ -v
```

期待される出力：
```
12 passed
```

- [ ] **Step 7: Commit**

```bash
git add src/experiment/ tests/experiment/
git commit -m "feat: MLflow実験記録runnerを実装"
```

---

## Task 7: プロンプトと設定ファイルの作成

**Files:**
- Create: `prompts/zero_shot/v1.txt`
- Create: `prompts/few_shot/v1.txt`
- Create: `configs/experiment.yaml`

- [ ] **Step 1: ゼロショットプロンプトを作成する**

`prompts/zero_shot/v1.txt` を作成：
```
この技術図面を解析し、コネクタ（端子・接続部品・プラグ・ソケット等）を検出してください。

コネクタを検出した場合、以下のJSON形式のみで出力してください：
{"connectors": [{"x_center": 0.25, "y_center": 0.40, "width": 0.10, "height": 0.06}]}

座標は画像全体を1.0とした正規化座標（0.0〜1.0）で記述してください。
コネクタが存在しない場合は {"connectors": []} と出力してください。
JSON以外のテキストは出力しないでください。
```

- [ ] **Step 2: フューショットプロンプトを作成する**

`prompts/few_shot/v1.txt` を作成：
```
この技術図面を解析し、コネクタ（端子・接続部品・プラグ・ソケット等）を検出してください。

【検出例】
- 矩形の端子ブロック → コネクタ
- ピン形状の接続部 → コネクタ
- ケーブル接続口 → コネクタ
- 通常の配線や導体 → コネクタではない

検出した場合、以下のJSON形式のみで出力してください：
{"connectors": [{"x_center": 0.25, "y_center": 0.40, "width": 0.10, "height": 0.06}]}

座標は画像全体を1.0とした正規化座標（0.0〜1.0）で記述してください。
コネクタが存在しない場合は {"connectors": []} と出力してください。
JSON以外のテキストは出力しないでください。
```

- [ ] **Step 3: 実験設定ファイルを作成する**

`configs/experiment.yaml` を作成：
```yaml
model: gemma4:26b

prompt:
  type: zero_shot
  file: prompts/zero_shot/v1.txt

params:
  temperature: 0.0
  top_p: 0.9
  top_k: 40

evaluation:
  iou_threshold: 0.5

data:
  samples_dir: data/samples
  ground_truth_dir: data/ground_truth
```

- [ ] **Step 4: Commit**

```bash
git add prompts/ configs/
git commit -m "feat: ゼロショット・フューショットプロンプトと実験設定ファイルを追加"
```

---

## Task 8: 統合動作確認

- [ ] **Step 1: 全サービスが起動していることを確認する**

Ubuntu側で実行：
```bash
docker compose ps
```

期待される出力：
```
NAME           STATUS
ollama         running
mlflow         running
python_dev     running
```

- [ ] **Step 2: MLflow UI を確認する**

ブラウザで `http://localhost:5000` にアクセスし、MLflow UIが表示されることを確認する。

- [ ] **Step 3: サンプル画像と正解データを配置する**

`data/samples/` に図面画像（PNG/JPG）を配置する。
`data/ground_truth/` に同名のJSONファイルを配置する（例: `drawing_001.png` → `drawing_001.json`）。

JSONフォーマット：
```json
{
  "connectors": [
    {"x_center": 0.25, "y_center": 0.40, "width": 0.10, "height": 0.06}
  ]
}
```

- [ ] **Step 4: 実験を実行する**

Ubuntu側で実行：
```bash
docker compose exec python python src/experiment/runner.py --config configs/experiment.yaml
```

期待される出力例：
```
drawing_001.png: P=0.800 R=0.750 F1=0.774

平均: P=0.800 R=0.750 F1=0.774
```

- [ ] **Step 5: MLflow UI で結果を確認する**

ブラウザで `http://localhost:5000` にアクセスし、`connector-detection` 実験に run が記録されていることを確認する。

パラメータ・メトリクス・アーティファクト（prompt.txt, responses/）が表示されれば完了。
