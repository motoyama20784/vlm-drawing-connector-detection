# 環境構築ガイド（Ubuntu + RTX 5090）

## 方針：Docker ファースト

本プロジェクトは**Docker上でOllamaを動かす**構成を採用する。

| レイヤー | 何が必要か | 誰が管理するか |
|---------|-----------|--------------|
| ホスト | NVIDIAドライバ + Docker + NVIDIA Container Toolkit | ホストOS |
| コンテナ | CUDA / Python / PyTorch / Ollama(オーラマ) | Dockerイメージ |

> CUDA Toolkit・Python・PyTorchはホストに直接インストールしない。コンテナ内で完結する。

---

## RTX 5090 (Blackwell) の注意点

RTX 5090 は Blackwell アーキテクチャ (sm_120) を採用しており、**CUDA 12.8以上が必須**。

| 項目 | 要件 |
|------|------|
| CUDAバージョン | 12.8以上（12.6以前はBlackwell非対応） |
| ドライババージョン | 570.xx以上 |
| Dockerイメージ | `ollama/ollama:latest`（CUDA 12.8対応済み） |

Docker を使う場合でも、**ホストのドライバは570以上が必要**（コンテナはホストドライバを経由してGPUにアクセスするため）。

---

## 1. NVIDIAドライバのインストール（ホスト）

```bash
sudo apt update
sudo apt install ubuntu-drivers-common

# 推奨ドライバの確認
ubuntu-drivers devices

# 自動インストール（570系が選ばれるはず）
sudo ubuntu-drivers autoinstall
sudo reboot
```

バージョン指定する場合：

```bash
sudo apt install nvidia-driver-570
sudo reboot
```

確認：

```bash
nvidia-smi
# Driver Version: 570.xx.xx, CUDA Version: 12.8 と表示されればOK
```

---

## 2. Docker Engineのインストール（ホスト）

```bash
# 古いDockerの削除
sudo apt remove docker docker-engine docker.io containerd runc

# 公式リポジトリ経由でインストール
sudo apt update
sudo apt install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# sudo なしで実行できるようにする
sudo usermod -aG docker $USER
newgrp docker

# 確認
docker --version
```

---

## 3. NVIDIA Container Toolkitのインストール（ホスト）

コンテナからGPUにアクセスするために必要。

```bash
# リポジトリの追加
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit

# Dockerと統合
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 確認（コンテナからGPUが見えるか）
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

RTX 5090が表示されればOK。

---

## 4. Ollamaコンテナの起動

`src/serving/ollama/docker-compose.yml` を使って起動する。

```bash
cd src/serving/ollama
docker compose up -d

# ログ確認
docker compose logs -f
```

起動後にGemma 4をダウンロード：

```bash
# コンテナ内でモデルをpull（約14GB）
docker exec -it ollama ollama pull gemma4:27b

# ダウンロード済みモデルの確認
docker exec -it ollama ollama list
```

---

## 5. 動作確認

### テキスト推論（curl）

```bash
curl http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4:27b",
    "prompt": "こんにちは。1文で自己紹介してください。",
    "stream": false
  }'
```

### 画像入力（base64エンコード）

```bash
# 図面画像をbase64に変換してAPIに投げる
IMAGE_B64=$(base64 -w 0 data/samples/sample_drawing.png)

curl http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"gemma4:27b\",
    \"prompt\": \"この図面に含まれるコネクタをすべて列挙してください。\",
    \"images\": [\"${IMAGE_B64}\"],
    \"stream\": false
  }"
```

### OpenAI SDK経由（Python）

```python
from openai import OpenAI

# base_urlをローカルに向けるだけ。トークン料なし。
client = OpenAI(base_url="http://localhost:11434/v1", api_key="dummy")

response = client.chat.completions.create(
    model="gemma4:27b",
    messages=[{"role": "user", "content": "この図面のコネクタを検出してください"}]
)
print(response.choices[0].message.content)
```

---

## Gemma 4 の量子化とVRAM要件

| 精度 | VRAM (27B) | RTX 5090 (32GB) | 備考 |
|------|-----------|-----------------|------|
| fp32 | ~108GB | 不可 | - |
| fp16 | ~54GB | 不可 | 96GBマシン推奨 |
| 8-bit | ~27GB | ギリギリ・不安定 | OOMリスクあり |
| **4-bit (GGUF)** | **~14GB** | **推奨** | Ollamaが自動選択 |

Ollamaは `gemma4:27b` をpullすると自動的に4-bit量子化（Q4_K_M相当）が選ばれる。

---

## トラブルシューティング

### コンテナからGPUが見えない

```bash
# Container Toolkitの再設定
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### モデルのダウンロードが途中で止まる

```bash
# 再実行すれば続きからダウンロードされる
docker exec -it ollama ollama pull gemma4:27b
```

### OOM（Out of Memory）エラー

```bash
# 他のプロセスのVRAM使用量を確認
nvidia-smi

# 軽量版に切り替え（8GB程度）
docker exec -it ollama ollama pull gemma4:12b
```
