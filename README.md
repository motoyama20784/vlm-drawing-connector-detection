# VLM Drawing Connector Detection

技術図面からコネクタを検出するVLM（Vision Language Model）オンプレ実装プロジェクト。

---

## このプロジェクトでやること

CAD図面・回路図・配線図などの技術図面を画像として入力し、図中に含まれる**コネクタ（端子・接続部品）の位置・種類・接続関係を自動で抽出する**。

クラウドAPIに依存せず、社内のGPUサーバー上でモデルを動かす（オンプレ）ことで、図面データを外部に送出せずに処理できる。

---

## 環境

| 項目 | 内容 |
|------|------|
| OS | Ubuntu |
| GPU（開発機） | NVIDIA RTX 5090（32GB VRAM） |
| GPU（大規模処理機） | 96GB VRAM 構成 |
| 推論基盤 | Docker + Ollama |
| メインモデル | Gemma 4 27B（4-bit量子化） |

---

## 進め方

### フェーズ1：推論環境の構築（現在）

Docker + Ollama で Gemma 4 を動かし、図面画像を入力して応答が返るところまで確認する。

### フェーズ2：PoC（概念実証）

実際の図面サンプルを使い、プロンプトエンジニアリングでコネクタ検出が動作するデモを作る。

### フェーズ3：精度向上

結果を見て、以下のいずれかまたは組み合わせで精度を高める：

- ファインチューニング（QLoRA + アノテーションデータ）
- YOLOとのハイブリッド（YOLO検出 → VLMで詳細分類）
- 他モデルへの切り替え（InternVL2、Qwen2-VL 等）

---

## 技術的な方針

**なぜVLMか**

- アノテーションデータなしでゼロショット検出を試せる
- 「断面Aのコネクタだけ」のような自然言語での条件指定ができる
- 検出結果に加えて種類・接続関係の説明も生成できる

**なぜオンプレか**

- 図面データを外部に送らずに処理できる（プライバシー）
- トークン使用料が発生しない（電気代のみ）
- モデルのカスタマイズ（ファインチューニング）が自由にできる

**なぜDockerか**

- CUDA・Python・フレームワークのバージョン管理をコンテナに閉じ込められる
- ホスト環境を汚さずに複数フレームワーク（Ollama / vLLM / HF Transformers）を切り替えられる

---

## ディレクトリ構成

```
src/                    # 実装コード
  serving/ollama/       # Ollama Docker構成
  serving/vllm/         # vLLM Docker構成（後続フェーズ）
  detection/            # コネクタ検出パイプライン
  visualization/        # 検出結果の可視化
  utils/                # 共通ユーティリティ

research/               # 実験・探索（手を動かすもの）
  notebooks/            # Jupyter実験ノート
  prompts/              # プロンプトのバリエーションと評価

docs/                   # ドキュメント（読んで参照するもの）
  specs/                # 設計ドキュメント・意思決定記録
  knowledge/            # 技術調査・フレームワーク比較

data/
  samples/              # テスト用図面サンプル
  outputs/              # 検出結果の出力
```

---

## 技術調査ドキュメント

`docs/knowledge/` に以下の調査資料がある。

| ファイル | 内容 |
|---------|------|
| 01-vlm-fundamentals | VLMの仕組み・CNNとの違い |
| 02-gemma4-overview | Gemma 4の特性・成功事例 |
| 03-environment-setup | Ubuntu + RTX 5090 + Docker 環境構築手順 |
| 04-hardware-requirements | VRAM要件・32GB vs 96GB の業務効率比較 |
| 05-object-detection-with-vlm | VLMで物体検知を行う3つのアプローチ |
| 06-fine-tuning | LoRA / QLoRAの手法とRTX 5090向け設定 |
| 07-docker-environment | Docker推論・学習環境の詳細 |
| 08-yolo-vs-vlm | YOLOとの比較・使い分け判断基準 |
| 09-on-premises-operations | オンプレ運用のメリット・コスト比較 |
| 10-alternative-models | DeepSeek・InternVL2・Qwen2-VLなど代替モデル |
| 11-serving-frameworks | Ollama / vLLM / HF Transformers の比較・ライセンス |
