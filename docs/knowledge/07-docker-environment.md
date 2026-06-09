# Docker環境構築（推論・学習）

## 概要

Docker を使うことで、CUDA / ドライバ / ライブラリのバージョン依存関係を環境ごとに分離・固定できます。オンプレ運用において最も安定した方法の一つです。

---

## 1. NVIDIA Container Toolkit のインストール

GPU を Docker コンテナから利用するために必須のツールです。

```bash
# 1. NVIDIA Container Toolkit リポジトリの追加
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 2. インストール
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 3. Docker デーモンの設定
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 4. 動作確認
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

期待される出力: `nvidia-smi` の結果が表示され、RTX 5090 が認識されていれば OK です。

---

## 2. 推奨ベースイメージ

NVIDIA が提供する公式イメージを使うと、CUDA / cuDNN が事前インストール済みで安定しています。

```
nvcr.io/nvidia/pytorch:25.02-py3
  ├─ Ubuntu 22.04
  ├─ CUDA 12.8
  ├─ cuDNN 9.x
  ├─ PyTorch 2.6
  └─ Python 3.11
```

利用可能なタグの確認:
```bash
# NGC カタログで確認
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
# タグ命名規則: YY.MM-py3 (例: 25.02-py3 = 2025年2月リリース)
```

---

## 3. 推論用 Dockerfile

Gemma 4 を vLLM または Transformers で動かす推論サービス用の Dockerfile です。

```dockerfile
# inference/Dockerfile

# NVIDIA 公式 PyTorch イメージをベースに使用
# CUDA 12.8 + PyTorch 2.6 対応（Blackwell RTX 5090 対応）
FROM nvcr.io/nvidia/pytorch:25.02-py3

# 環境変数
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/cache/huggingface
ENV TRANSFORMERS_CACHE=/cache/huggingface

# システムパッケージのインストール
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python パッケージのインストール
WORKDIR /app

COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# アプリケーションコードのコピー
COPY src/ ./src/
COPY configs/ ./configs/

# ポートの公開
EXPOSE 8000

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# 起動コマンド（vLLM による OpenAI 互換 API サービング）
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "/models/gemma4-27b", \
     "--quantization", "awq", \
     "--max-model-len", "8192", \
     "--gpu-memory-utilization", "0.90", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
```

```text
# requirements-inference.txt
vllm>=0.6.0
transformers>=4.49.0
accelerate>=0.34.0
bitsandbytes>=0.43.0
sentencepiece>=0.1.99
pillow>=10.0.0
fastapi>=0.115.0
uvicorn>=0.30.0
httpx>=0.27.0
```

---

## 4. 学習（ファインチューニング）用 Dockerfile

QLoRA ファインチューニング用の Dockerfile です。

```dockerfile
# training/Dockerfile

FROM nvcr.io/nvidia/pytorch:25.02-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/cache/huggingface

# システムパッケージ
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Flash Attention 2 のビルド（高速化、オプション）
# RTX 5090 (Blackwell sm_120) に対応したバージョンを指定
RUN pip install --no-cache-dir \
    flash-attn>=2.7.0 \
    --no-build-isolation

# 学習用パッケージのインストール
COPY requirements-training.txt .
RUN pip install --no-cache-dir -r requirements-training.txt

# 学習スクリプトのコピー
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/train.py ./train.py

# データとモデルのマウントポイント
VOLUME ["/models", "/data", "/output"]

# デフォルト起動（override 可能）
CMD ["python", "train.py", "--config", "configs/qlora_gemma4.yaml"]
```

```text
# requirements-training.txt
transformers>=4.49.0
accelerate>=0.34.0
bitsandbytes>=0.43.0
peft>=0.12.0
trl>=0.11.0
datasets>=3.0.0
sentencepiece>=0.1.99
pillow>=10.0.0
mlflow>=2.13.0
deepspeed>=0.15.0
evaluate>=0.4.0
scikit-learn>=1.5.0
```

---

## 5. docker-compose.yml

推論サービスと関連ツールをまとめて管理します。

```yaml
# docker-compose.yml

version: "3.9"

services:
  # ===== 推論サービス =====
  inference:
    build:
      context: .
      dockerfile: inference/Dockerfile
    image: vlm-connector-inference:latest
    container_name: vlm-inference
    
    # GPU の割り当て（全 GPU を使用）
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all        # または count: 1 で GPU を1枚指定
              capabilities: [gpu]
    
    # ボリュームマウント（モデルウェイトを共有）
    volumes:
      - /mnt/models:/models:ro          # モデルディレクトリ（読み取り専用）
      - /mnt/cache/huggingface:/cache/huggingface  # HF キャッシュ
      - ./logs:/app/logs                # ログ出力
    
    # 環境変数
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_NAME=gemma4-27b
      - MAX_MODEL_LEN=8192
      - GPU_MEMORY_UTILIZATION=0.90
    
    ports:
      - "8000:8000"
    
    restart: unless-stopped
    
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s  # モデルのロードに時間がかかるため長めに設定

  # ===== 推論 API プロキシ (オプション) =====
  nginx:
    image: nginx:alpine
    container_name: vlm-nginx
    ports:
      - "80:80"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - inference
    restart: unless-stopped

  # ===== 実験管理 =====
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    container_name: vlm-mlflow
    ports:
      - "5000:5000"
    volumes:
      - ./mlflow-data:/mlflow     # 実験ログ・アーティファクトをホストに永続化
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri /mlflow/experiments
      --default-artifact-root /mlflow/artifacts
    restart: unless-stopped

  # ===== モニタリング (オプション) =====
  prometheus:
    image: prom/prometheus:latest
    container_name: vlm-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    restart: unless-stopped

volumes:
  hf-cache:
    driver: local
```

### よく使うコマンド

```bash
# サービスの起動
docker compose up -d

# ログの確認
docker compose logs -f inference

# サービスの停止
docker compose down

# コンテナに入る
docker compose exec inference bash

# GPU 使用状況の確認（コンテナ内から）
docker compose exec inference nvidia-smi
```

---

## 6. GPU 割り当て設定

### --gpus フラグを使う場合（docker run）

```bash
# 全 GPU を使用
docker run --gpus all vlm-connector-inference:latest

# 特定の GPU を指定（GPU 0 のみ）
docker run --gpus '"device=0"' vlm-connector-inference:latest

# 複数 GPU を指定（GPU 0 と 1）
docker run --gpus '"device=0,1"' vlm-connector-inference:latest

# GPU 数で指定（最初の2枚）
docker run --gpus 2 vlm-connector-inference:latest
```

### docker-compose での指定

```yaml
# 方法1: count: all（全 GPU）
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]

# 方法2: 特定の GPU を指定
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ["0"]  # GPU 0 のみ
          capabilities: [gpu]

# 方法3: 環境変数で指定（コンテナ内の CUDA_VISIBLE_DEVICES）
environment:
  - CUDA_VISIBLE_DEVICES=0
```

---

## 7. ボリュームマウントによるモデルウェイト共有

複数のコンテナ（推論・学習）でモデルファイルを共有し、ディスクの節約と管理の一元化を実現します。

### ディレクトリ構成

```
/mnt/models/                       # ホスト側のモデルディレクトリ
  ├── gemma4-27b/                  # Hugging Face 形式
  │   ├── config.json
  │   ├── tokenizer.json
  │   └── model.safetensors.index.json
  ├── gemma4-27b-awq/              # AWQ 量子化済み
  │   └── ...
  └── gemma4-27b-q4km.gguf        # GGUF 形式（Ollama/llama.cpp 用）
```

### マウント設定の例

```yaml
# docker-compose.yml
services:
  inference:
    volumes:
      # モデルディレクトリ（読み取り専用でマウント）
      - /mnt/models:/models:ro
      
      # Hugging Face キャッシュ（ダウンロード済みモデルの共有）
      - /mnt/cache/huggingface:/root/.cache/huggingface
      
  training:
    volumes:
      # 学習用はモデルの読み書き両方が必要
      - /mnt/models:/models:rw
      
      # 学習データ（読み取り専用）
      - /mnt/data:/data:ro
      
      # 学習結果の出力先
      - /mnt/output:/output:rw
```

### モデルの事前ダウンロード

```bash
# ホスト側でモデルをダウンロード（コンテナ外）
pip install huggingface_hub

# Hugging Face からダウンロード
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='google/gemma-4-27b-it',
    local_dir='/mnt/models/gemma4-27b',
    ignore_patterns=['*.bin'],  # .safetensors のみダウンロード
)
"

# GGUF ファイルの場合（Ollama）
ollama pull gemma4:27b
# モデルは /root/.ollama/models/ に保存される
```

---

## 8. よくある問題とトラブルシューティング

### 問題 1: CUDA バージョン不一致

**症状**: コンテナ内で `torch.cuda.is_available()` が `False` になる、または CUDA エラーが出る

```bash
# 確認コマンド
# ホスト側
nvidia-smi  # ドライババージョン確認

# コンテナ内
docker run --gpus all --rm nvcr.io/nvidia/pytorch:25.02-py3 \
  python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"
```

**解決策**:
```bash
# ベースイメージのCUDAバージョンとホストドライバの互換性確認
# CUDA 12.8 → ドライバ 570.xx 以上が必要

# イメージのCUDAバージョン確認
docker run --rm nvcr.io/nvidia/pytorch:25.02-py3 nvcc --version

# ドライバが古い場合はホスト側のドライバを更新
sudo apt install nvidia-driver-570
sudo reboot
```

### 問題 2: OOM (Out of Memory) エラー

**症状**: `CUDA out of memory` エラーでコンテナがクラッシュ

```bash
# 解決策1: gpu-memory-utilization を下げる
CMD ["python", "-m", "vllm.entrypoints.openai.api_server",
     "--gpu-memory-utilization", "0.80",  # 0.90 → 0.80 に下げる
     ...]

# 解決策2: max-model-len を制限する
CMD ["python", "-m", "vllm.entrypoints.openai.api_server",
     "--max-model-len", "4096",  # 8192 → 4096 に制限
     ...]

# 解決策3: コンテナ再起動でVRAMをクリア
docker compose restart inference
```

### 問題 3: NVIDIA Container Toolkit が認識されない

**症状**: `docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`

```bash
# 確認
docker info | grep -i runtime

# 再設定
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 確認
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

### 問題 4: コンテナ内でモデルが見つからない

**症状**: `FileNotFoundError: /models/gemma4-27b not found`

```bash
# ボリュームマウントの確認
docker compose exec inference ls -la /models/

# ホスト側のパスが正しいか確認
ls -la /mnt/models/

# docker-compose.yml のパスを修正
volumes:
  - /mnt/models:/models:ro  # ← ホスト側のパスが正しいか確認
```

### 問題 5: Flash Attention のビルドエラー

**症状**: `flash-attn` のインストール中にビルドエラー

```bash
# 解決策1: プリビルド済みのホイールを使う
pip install flash-attn --no-build-isolation

# 解決策2: Flash Attention を無効化して Transformers を使う
# transformers の from_pretrained で attn_implementation を省略するか
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="eager",  # flash_attention_2 ではなく eager を使用
)
```

---

## 9. イメージのビルドとプッシュ

```bash
# イメージのビルド
docker compose build inference

# タグ付け（バージョン管理）
docker tag vlm-connector-inference:latest vlm-connector-inference:v1.0.0

# ローカルレジストリへのプッシュ（オンプレ環境用）
docker tag vlm-connector-inference:latest registry.local:5000/vlm-connector-inference:v1.0.0
docker push registry.local:5000/vlm-connector-inference:v1.0.0

# イメージのエクスポート（オフライン環境への転送）
docker save vlm-connector-inference:v1.0.0 | gzip > vlm-connector-inference-v1.0.0.tar.gz

# 別マシンでのインポート
docker load < vlm-connector-inference-v1.0.0.tar.gz
```
