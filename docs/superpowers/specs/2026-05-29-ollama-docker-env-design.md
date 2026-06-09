# Ollama + Docker 環境構築 設計ドキュメント

## 目的

Ubuntu + RTX 5090 上で Docker を使って Ollama を立ち上げ、Gemma 4 27B（4-bit量子化）による推論が動作する状態を作る。

---

## スコープ

**含む:**
- NVIDIA Container Toolkit のセットアップ
- Ollama の Docker コンテナ構成
- Gemma 4 モデルのダウンロード・動作確認
- OpenAI互換APIによる推論確認

**含まない:**
- ファインチューニング環境（後続フェーズ）
- vLLM / HF Transformers への移行（後続フェーズ）
- 図面コネクタ検出パイプライン（後続フェーズ）

---

## 前提環境

| 項目 | 内容 |
|------|------|
| OS | Ubuntu 22.04 / 24.04 |
| GPU | NVIDIA RTX 5090（32GB VRAM、Blackwell） |
| 必要 CUDA | 12.8 以上（Blackwell 対応） |
| Docker | Docker Engine 24.0 以上 |

---

## アーキテクチャ

```
[Ubuntu ホスト]
  └─ Docker Engine
      └─ ollama コンテナ（ollama/ollama:latest）
          ├─ Ollama サーバー（ポート 11434）
          ├─ Gemma 4 27B (4-bit GGUF)
          └─ GPU アクセス（NVIDIA Container Toolkit 経由）

[ホスト側ボリューム]
  ~/.ollama  ←→  /root/.ollama（モデルキャッシュ永続化）
```

---

## コンテナ構成

### docker-compose.yml の方針

- ベースイメージ: `ollama/ollama:latest`
- GPU: `deploy.resources.reservations.devices` で全GPU割り当て
- ポート: `11434:11434`
- ボリューム: `~/.ollama:/root/.ollama`（モデルを再ダウンロードしなくて済む）
- 再起動ポリシー: `unless-stopped`

---

## モデル選定

| モデル | VRAM（4-bit） | 備考 |
|--------|-------------|------|
| `gemma4:27b` | ~16GB | 推奨。ビジョン対応 |
| `gemma4:12b` | ~8GB | 軽量版（精度トレードオフあり） |

RTX 5090（32GB）では `gemma4:27b` が余裕を持って動作する。

---

## 動作確認の方針

1. **テキスト推論**: curl でシンプルな質問を投げて応答確認
2. **画像入力**: サンプル図面を base64 エンコードして送信し、コネクタの有無を問う
3. **OpenAI SDK 経由**: Python スクリプトで `base_url` をローカルに向けて動作確認

---

## ファイル構成

```
src/serving/ollama/
  docker-compose.yml     # Ollama コンテナ定義
  .env.example           # 設定値テンプレート

research/notebooks/
  01-ollama-setup-verify.ipynb   # 動作確認ノートブック
```

---

## 成功条件

- [ ] `docker compose up` でコンテナが起動する
- [ ] `ollama list` で `gemma4:27b` が表示される
- [ ] curl でテキスト応答が返る
- [ ] サンプル図面画像を入力して応答が返る
