# アノテーションツール設計ドキュメント

## 目的

zero-shot / few-shot の性能比較に使うアノテーション正解データを効率よく作成するための Web アノテーションツールを構築する。図面画像上に bbox を描き、VLM（ollama）でクロップ領域のテキスト・形状情報を抽出し、JSON として保存する。

---

## スコープ

**含む:**
- bbox 描画 UI（React + Canvas）
- FastAPI バックエンド（画像配信・アノテーション読み書き・VLM 推論中継）
- Docker Compose への annotator サービス追加
- 既存 `data/ground_truth/` フォーマットへの出力

**含まない:**
- 画像のアップロード（フェーズ1は `data/samples/` の既存画像のみ）
- ユーザー認証
- アノテーションのバージョン管理
- ファインチューニング用データ変換

---

## アーキテクチャ

```
docker-compose.yml
├── ollama      port 11434  (既存・変更なし)
├── mlflow      port 5000   (既存・変更なし)
├── python                  (既存・変更なし)
└── annotator   port 8000   (新規)
    ├── FastAPI  /api/*
    └── React    /  (ビルド済み静的ファイルを FastAPI から配信)
```

**データフロー:**
```
Windows ブラウザ
  ↕ HTTP :8000
annotator (FastAPI)
  ├── data/samples/      ← 画像読み込み (volume mount)
  ├── data/ground_truth/ ← アノテーション JSON 読み書き (volume mount)
  └── http://ollama:11434 ← bbox クロップ画像を VLM に投げる
```

---

## ディレクトリ構成

```
annotator/                    (新規・トップレベル)
├── backend/
│   ├── main.py               FastAPI アプリ本体
│   └── routes/
│       ├── images.py         画像一覧・配信
│       ├── annotations.py    アノテーション読み書き
│       └── infer.py          VLM 推論中継
└── frontend/
    ├── src/                  React ソース
    └── dist/                 ビルド後静的ファイル（FastAPI から配信）

docker/annotator/
├── Dockerfile
└── requirements.txt          (fastapi, uvicorn, httpx, pillow, python-multipart)

prompts/annotator/
└── v1.txt                    (新規) VLM 推論用プロンプト
```

---

## フロントエンド

**画面構成（シングルページ）:**

```
┌─────────────────────────────────────────────────────┐
│ [画像選択ドロップダウン]            [保存ボタン]     │
├──────────────────────────┬──────────────────────────┤
│                          │ Bbox リスト              │
│   キャンバス             │ ┌──────────────────────┐ │
│   （画像表示＋bbox描画） │ │ bbox-1               │ │
│                          │ │  カテゴリ: [入力欄]  │ │
│   マウスドラッグで       │ │  [VLM推論] ボタン    │ │
│   bbox を描く            │ │  テキスト: CN1       │ │
│                          │ │  形状: rectangular   │ │
│                          │ ├──────────────────────┤ │
│                          │ │ bbox-2               │ │
│                          │ │  ...                 │ │
└──────────────────────────┴──────────────────────────┘
```

**操作フロー:**
1. ドロップダウンで `data/samples/` の画像を選択
2. Canvas 上でマウスドラッグ → bbox が描かれる（複数可）
3. 右パネルで各 bbox のカテゴリを手入力
4. 各 bbox の「VLM推論」ボタン → ollama にクロップ画像を送り、テキスト・形状分類を右パネルに表示
5. 「保存」ボタン → `data/ground_truth/<filename>.json` に書き出す
6. 既存アノテーションがある画像を選ぶと bbox が自動復元される

**使用ライブラリ:**
- React + Vite
- HTML5 Canvas API（bbox 描画）
- axios（API 通信）

---

## API エンドポイント

| メソッド | パス | 説明 |
|---------|------|------|
| GET | `/api/images` | `data/samples/` の画像ファイル一覧 |
| GET | `/api/images/{filename}` | 画像ファイル本体を返す |
| GET | `/api/annotations/{filename}` | 既存アノテーション JSON を返す（なければ空） |
| POST | `/api/annotations/{filename}` | アノテーション JSON を保存 |
| POST | `/api/infer` | bbox クロップを ollama に投げ結果を返す |

---

## アノテーション JSON 形式

既存の `data/ground_truth/` フォーマットを拡張する。`x_center` / `y_center` / `width` / `height` は既存の evaluator.py と互換性を保つ。

```json
{
  "image": "drawing_001.png",
  "connectors": [
    {
      "id": "bbox-1",
      "x_center": 0.25,
      "y_center": 0.40,
      "width": 0.10,
      "height": 0.06,
      "category": "terminal",
      "vlm_text": "CN1",
      "vlm_shape": "rectangular"
    }
  ]
}
```

- `id` : フロント側で bbox を識別するキー
- `category` : 人間が手入力するラベル
- `vlm_text` : VLM が抽出したテキスト（推論前は `null`）
- `vlm_shape` : VLM による形状分類（推論前は `null`）

---

## VLM 推論詳細

**`POST /api/infer` リクエスト:**
```json
{
  "image": "drawing_001.png",
  "bbox": { "x_center": 0.25, "y_center": 0.40, "width": 0.10, "height": 0.06 }
}
```

**処理フロー:**
1. `data/samples/{image}` を Pillow で開く
2. bbox 座標（正規化 → ピクセル変換）でクロップ
3. クロップ画像を base64 エンコード
4. `prompts/annotator/v1.txt` のプロンプトを読み込む
5. ollama API（multimodal）に投げる
6. レスポンスをパースして返す

**レスポンス:**
```json
{ "vlm_text": "CN1", "vlm_shape": "rectangular" }
```

**設定:**
- モデル: 環境変数 `MODEL`（デフォルト: `gemma4:26b`）
- プロンプトファイル: 環境変数 `ANNOTATOR_PROMPT_FILE`（デフォルト: `prompts/annotator/v1.txt`）
- プロンプトファイルは volume mount 経由で読むため、コンテナ再ビルド不要で変更可能

**デフォルトプロンプト (`prompts/annotator/v1.txt`):**
```
この図面の一部を見てください。
以下のJSON形式のみで回答してください：
{"vlm_text": "画像内のテキストや記号", "vlm_shape": "形状の種類（例: rectangular, circular, line など）"}
テキストがなければ vlm_text は空文字にしてください。
```

---

## Docker 設定

**`docker-compose.yml` への追加:**
```yaml
annotator:
  build:
    context: .
    dockerfile: docker/annotator/Dockerfile
  ports:
    - "8000:8000"
  volumes:
    - ./data:/app/data
    - ./prompts:/app/prompts
  environment:
    - OLLAMA_BASE_URL=http://ollama:11434
    - MODEL=gemma4:26b
    - ANNOTATOR_PROMPT_FILE=prompts/annotator/v1.txt
  depends_on:
    - ollama
```

---

## 成功条件

- [ ] `docker compose up` で annotator サービスが起動する
- [ ] `http://localhost:8000` でアノテーション画面が表示される
- [ ] `data/samples/` の画像を選択して Canvas に表示できる
- [ ] bbox を描いて「VLM推論」を実行すると vlm_text / vlm_shape が表示される
- [ ] 「保存」で `data/ground_truth/<filename>.json` が生成される
- [ ] 既存アノテーション済み画像を選ぶと bbox が復元される
- [ ] 既存の evaluator.py がアノテーション JSON を読み込める（互換性維持）
