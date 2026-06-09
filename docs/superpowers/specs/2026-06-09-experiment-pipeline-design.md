# 実験パイプライン設計ドキュメント

## 目的

VLMによるコネクタ検出の実験管理基盤を構築する。ゼロショット・フューショットのプロンプト実験、モデル切り替え、パラメータチューニングの結果をMLflowで記録・比較できる状態を作る。

---

## スコープ

**含む:**
- Docker Composeによる全サービス統合（Ollama / MLflow / Python）
- コネクタ検出パイプライン（推論・解析・評価）
- MLflowによる実験記録
- バウンディングボックスのIoUベース評価

**含まない:**
- ファインチューニング
- WebUI / APIサービス化
- コネクタの種類分類（現フェーズはあり・なしの検出のみ）

---

## アーキテクチャ

```
[docker-compose.yml]
├── ollama    推論サーバー (port 11434)
├── mlflow    実験記録サーバー (port 5000)
└── python    実験スクリプト実行コンテナ

[通信]
python → ollama   : http://ollama:11434
python → mlflow   : http://mlflow:5000
[外部アクセス]
ホスト → ollama   : http://localhost:11434
ホスト → mlflow   : http://localhost:5000
```

---

## ディレクトリ構成

```
project root/
├── docker-compose.yml
│
├── docker/
│   ├── ollama/
│   │   └── .env.example
│   ├── mlflow/
│   │   └── .env.example
│   └── python/
│       └── Dockerfile
│
├── src/
│   ├── detection/
│   │   ├── detector.py       Ollama APIを叩いてVLM推論
│   │   ├── parser.py         VLMテキスト出力→バウンディングボックスに変換
│   │   └── evaluator.py      IoU計算・Precision/Recall/F1
│   └── experiment/
│       └── runner.py         MLflow実験の記録・管理
│
├── prompts/
│   ├── zero_shot/
│   │   └── v1.txt
│   └── few_shot/
│       └── v1.txt
│
├── configs/
│   └── experiment.yaml       モデル名・パラメータ・プロンプト設定
│
├── data/
│   ├── samples/              テスト用図面画像
│   ├── ground_truth/         アノテーション（JSON）
│   └── outputs/              検出結果の出力
│
├── mlflow/
│   └── mlruns/               MLflow成果物の永続化
│
└── research/
    └── notebooks/
```

---

## 各コンポーネントの役割

### detector.py
- Ollama APIへリクエスト送信
- モデル名・プロンプト・画像・パラメータを受け取り、VLMの生テキストを返す
- `OLLAMA_BASE_URL` は環境変数で設定

### parser.py
- VLMのテキスト出力からJSONブロックを抽出
- バウンディングボックスを正規化座標（0〜1）で取得
- パース失敗時は空リストを返す（実験を止めない）

### evaluator.py
- 予測ボックスと正解ボックスのIoUを計算
- IoU閾値（デフォルト0.5）でTP/FP/FNを判定
- Precision / Recall / F1を返す

### runner.py
- MLflow runを開始し、パラメータ・メトリクス・アーティファクトを記録
- 1実験 = 1 MLflow run
- 記録内容は後から追加しやすい構造にする

---

## バウンディングボックスのフォーマット

### アノテーション（正解データ）

```json
{
  "image": "drawing_001.png",
  "connectors": [
    {"x_center": 0.25, "y_center": 0.40, "width": 0.10, "height": 0.06},
    {"x_center": 0.60, "y_center": 0.30, "width": 0.08, "height": 0.05}
  ]
}
```

### VLMへの出力指示（プロンプト末尾に付与）

```
コネクタを検出した場合、以下のJSON形式のみで出力してください：
{"connectors": [{"x_center": 0.25, "y_center": 0.40, "width": 0.10, "height": 0.06}]}
座標は画像全体を1.0とした正規化座標で記述してください。
コネクタがなければ {"connectors": []} と出力してください。
```

---

## MLflowに記録する内容

### パラメータ（実験条件）
- `model` : モデル名（例: gemma4:26b）
- `prompt_file` : 使用したプロンプトファイル名
- `prompt_type` : zero_shot / few_shot
- `temperature` : サンプリング温度
- `top_p` : top-p サンプリング
- `top_k` : top-k サンプリング
- `iou_threshold` : 評価時のIoU閾値

### メトリクス（評価結果）
- `precision` / `recall` / `f1` : 全画像の平均
- `precision_<image>` / `recall_<image>` / `f1_<image>` : 画像ごとの値

### アーティファクト
- 使用したプロンプトテキスト
- VLMの生テキスト出力（画像ごと）

※ 記録項目は実験を進めながら追加する

---

## experiment.yaml の構成

```yaml
model: gemma4:26b
prompt:
  type: zero_shot       # zero_shot / few_shot
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

---

## 実験実行フロー

```
1. experiment.yaml を読み込む
2. MLflow run を開始
3. パラメータを記録
4. data/samples の画像を1枚ずつ処理:
   a. detector.py でVLM推論
   b. parser.py でバウンディングボックスを抽出
   c. evaluator.py で正解データと照合
   d. 画像ごとのメトリクスを記録
5. 全画像の平均メトリクスを記録
6. MLflow run を終了
```

---

## 実験の実行方法

```bash
# Pythonコンテナに入って実行
docker compose exec python python src/experiment/runner.py --config configs/experiment.yaml

# またはコンテナ起動時に直接実行
docker compose run --rm python python src/experiment/runner.py --config configs/experiment.yaml
```

---

## 既存ファイルの移行

現在 `src/serving/ollama/` にある以下のファイルを `docker/ollama/` に移動する：
- `.env.example`
- `Makefile`

`docker-compose.yml` はルートに新規作成し、既存の `src/serving/ollama/docker-compose.yml` は削除する。

---

## 成功条件

- [ ] `docker compose up` で3サービスが起動する
- [ ] MLflow UI が `http://localhost:5000` で表示される
- [ ] `python runner.py` で実験が1件記録される
- [ ] MLflow UI上で実験結果を比較できる
