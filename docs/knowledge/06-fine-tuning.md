# ファインチューニング手法（LoRA・QLoRA等）

## VLMに対するファインチューニングの種類

### 1. Full Fine-tuning（フルファインチューニング）

モデルの全パラメータを更新する最もシンプルな手法。

- **特性**: 全重みが更新されるため、理論上最高精度が期待できる
- **VRAM要件**: Gemma 4 27B の場合、学習には**100GB+ VRAM** が必要（A100 80GB × 2枚以上）
- **RTX 5090 (32GB) での実施**: **不可能**（モデルの推論だけで40GB+ 必要）
- **用途**: 大規模クラスタ環境、根本的なドメイン適応が必要な場合

```
モデルサイズ別のFull Fine-tuning VRAM目安（混合精度 bf16）:
  Gemma 4 4B  → ~30GB（ギリギリ可能）
  Gemma 4 12B → ~90GB（複数GPU必要）
  Gemma 4 27B → ~200GB（非現実的）
```

---

### 2. LoRA（Low-Rank Adaptation）

**元の重みを固定し、低ランク行列（追加パラメータ）のみを学習する**効率的なファインチューニング手法。

#### 仕組み

Transformer の線形層（Attention の Q, K, V, O 射影など）に対して、以下の低ランク行列を挿入する。

```
元の重み W ∈ R^(d×k)  ←  凍結（更新しない）
追加行列: ΔW = B × A
  A ∈ R^(r×k)  ←  ランダム初期化、学習対象
  B ∈ R^(d×r)  ←  ゼロ初期化、学習対象
  r << min(d, k)  (r は rank, 通常 4〜64)

実際の推論: W + ΔW = W + BA
```

#### Gemma 4 への適用

対象となる主なモジュール:
- `q_proj` / `k_proj` / `v_proj` / `o_proj`（Attention 層）
- `gate_proj` / `up_proj` / `down_proj`（FFN 層）
- VLM の場合: `linear_proj`（Vision→Language の投影層）

#### VRAM要件（Gemma 4 27B、bf16）

```
rank=16, target_modules=["q_proj","v_proj","k_proj","o_proj"] の場合:
  追加パラメータ: 数百MB程度
  学習時 VRAM: ~40〜50GB（勾配チェックポイントなし）
                 ~30GB（勾配チェックポイントあり）
→ RTX 5090 単体では厳しい（QLoRAを推奨）
```

---

### 3. QLoRA（Quantized LoRA）

**モデルを4-bit量子化した上でLoRAを適用する**手法。RTX 5090 (32GB VRAM) での Gemma 4 27B ファインチューニングを可能にする。

#### 仕組み

```
1. モデルの重みを NF4（4-bit Normal Float）形式に量子化
   → VRAM 使用量が約 1/4 に削減
2. 計算時のみ bf16 に復元（bfloat16 での計算精度を維持）
3. 凍結した量子化重みの上に LoRA アダプタを追加・学習
```

#### VRAM要件（Gemma 4 27B、4-bit量子化 + LoRA）

```
量子化後モデルサイズ: ~14GB
LoRA アダプタ: ~0.5GB
学習オーバーヘッド: ~10〜15GB（gradient checkpointing 有効時）
合計: ~25〜30GB → RTX 5090 (32GB) で動作可能
```

#### Full LoRA vs QLoRA の比較

| 項目 | LoRA (bf16) | QLoRA (4-bit) |
|---|---|---|
| VRAM使用量 | 多い（40GB+） | 少ない（25〜30GB） |
| 学習速度 | 速い | やや遅い（量子化/逆量子化オーバーヘッド） |
| 精度 | Full Fine-tuningに近い | わずかに低下するが実用範囲内 |
| RTX 5090 対応 | 困難 | 可能 |

---

### 4. Prefix Tuning / Prompt Tuning

入力に学習可能なトークン（プレフィックス）を追加する手法。

- **Prefix Tuning**: 各 Transformer 層の key/value に学習可能ベクトルを追加
- **Prompt Tuning**: 入力の先頭に学習可能な「ソフトプロンプト」を追加
- **特性**: パラメータ効率が高いが、複雑なタスクへの適応力は LoRA より低い
- **用途**: 非常に限られた VRAM 環境、タスクが単純な場合

---

## RTX 5090 (32GB) での LoRA/QLoRA 実用設定

### QLoRA 推奨設定（Gemma 4 27B）

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 量子化設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NF4 量子化（QLoRA推奨）
    bnb_4bit_compute_dtype=torch.bfloat16,  # 計算は bf16 で実施
    bnb_4bit_use_double_quant=True,     # 二重量子化でさらにVRAM削減
)

# モデルロード
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-27b-it",
    quantization_config=bnb_config,
    device_map="auto",
)

# 勾配チェックポイント有効化
model = prepare_model_for_kbit_training(model)
model.enable_input_require_grads()

# LoRA 設定
lora_config = LoraConfig(
    r=16,                    # rank（低いほどパラメータ少、高いほど表現力高）
    lora_alpha=32,           # スケーリング係数（通常 r×2 が目安）
    target_modules=[         # LoRA を適用する層
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        # 必要に応じて追加:
        # "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,       # 過学習防止のドロップアウト
    bias="none",             # バイアス項は学習しない
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 例: trainable params: 83,886,080 || all params: 27,053,432,832 || trainable%: 0.31%
```

### 学習設定（SFTTrainer）

```python
from transformers import TrainingArguments
from trl import SFTTrainer

training_args = TrainingArguments(
    output_dir="./gemma4-connector-detection",
    per_device_train_batch_size=1,          # RTX 5090 では1が安全
    gradient_accumulation_steps=8,          # 実効バッチサイズ = 1×8 = 8
    gradient_checkpointing=True,            # VRAM削減（速度とのトレードオフ）
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    fp16=False,
    bf16=True,                              # RTX 5090 は bf16 対応
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    report_to="mlflow",                     # 学習監視（MLflow ローカルサーバー）
    dataloader_num_workers=4,
    optim="paged_adamw_8bit",               # 8-bit optimizer でVRAM削減
)
```

---

## 主要ライブラリ

### peft（HuggingFace）

LoRA / QLoRA の実装ライブラリ。

```bash
pip install peft
```

- `LoraConfig`: LoRAの設定（rank, alpha, target_modules）
- `get_peft_model`: 既存モデルに LoRA を適用
- `PeftModel.from_pretrained`: 学習済み LoRA アダプタのロード

### bitsandbytes

量子化ライブラリ（4-bit / 8-bit）。

```bash
pip install bitsandbytes
```

- `BitsAndBytesConfig`: 量子化設定（nf4, fp4）
- RTX 5090 では CUDA 12.x 対応バージョンを使用すること

### trl（Transformer Reinforcement Learning）

SFTTrainer（Supervised Fine-tuning Trainer）を提供。

```bash
pip install trl
```

- `SFTTrainer`: 対話形式データセットのファインチューニングに最適化
- データフォーマットの自動変換機能あり

### unsloth

高速LoRA実装。標準実装より**2倍高速**で VRAM 効率も高い。

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

- Triton カーネルによる最適化
- Gemma シリーズへの対応あり
- `FastLanguageModel.from_pretrained` で簡単に適用可能

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-4-27b-it",
    max_seq_length=4096,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=32,
    lora_dropout=0,    # unsloth では 0 推奨
    bias="none",
    use_gradient_checkpointing="unsloth",  # unsloth 最適化版
)
```

### axolotl

設定ファイル（YAML）ベースのファインチューニングフレームワーク。コーディングなしで実行可能。

```bash
pip install axolotl
```

```yaml
# axolotl 設定例 (config.yaml)
base_model: google/gemma-4-27b-it
model_type: AutoModelForCausalLM

load_in_4bit: true
adapter: lora
lora_r: 16
lora_alpha: 32
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj

datasets:
  - path: ./data/connector_detection.jsonl
    type: sharegpt

sequence_len: 4096
gradient_accumulation_steps: 8
micro_batch_size: 1
num_epochs: 3
learning_rate: 0.0002
```

---

## 学習データ形式（VLM用）

### 標準フォーマット（ShareGPT形式）

```json
[
  {
    "image": "path/to/drawing_001.png",
    "conversations": [
      {
        "role": "user",
        "content": "この図面のコネクタを検出してください。JSON形式で出力してください。"
      },
      {
        "role": "assistant",
        "content": "{\"connectors\": [{\"id\": 1, \"type\": \"terminal\", \"bbox\": [100, 200, 150, 250], \"part_number\": \"CN1\"}, {\"id\": 2, \"type\": \"plug\", \"bbox\": [300, 150, 360, 210], \"part_number\": \"J3\"}], \"total_count\": 2}"
      }
    ]
  },
  {
    "image": "path/to/drawing_002.png",
    "conversations": [
      {
        "role": "user",
        "content": "添付の電気図面を解析し、すべての電気コネクタの位置と種類を特定してください。"
      },
      {
        "role": "assistant",
        "content": "{\"connectors\": [{\"id\": 1, \"type\": \"socket\", \"bbox\": [80, 320, 140, 380], \"part_number\": \"CN2\"}], \"total_count\": 1}"
      }
    ]
  }
]
```

### データセット作成のガイドライン

- **画像形式**: PNG（可逆圧縮）または JPEG（品質90以上）
- **解像度**: 最低1024×768px 推奨、600DPI換算相当
- **座標系**: 画像左上を (0, 0)、右下を (width, height) とするピクセル座標
- **アノテーション精度**: bounding box の余白は5〜10px 程度
- **データ量**: 最低100サンプル、500サンプル以上で安定した精度

---

## ファインチューニング時の注意点

### Vision Encoder の扱い

一般的には **Vision Encoder（ViT部分）は凍結**し、Language Model 側の層にのみ LoRA を適用する。

```python
# Vision Encoder の凍結例
for name, param in model.named_parameters():
    if "vision_tower" in name or "vision_encoder" in name:
        param.requires_grad = False  # 凍結

# Language Model 側のみ LoRA 適用（target_modules で指定）
```

**理由**:
- Vision Encoder は一般的な視覚特徴を学習済みであり、図面タスクでも有効な特徴を持つ
- Language 部分のファインチューニングで「図面の視覚特徴をコネクタの座標/種類に変換する能力」を習得させる
- Vision Encoder まで学習すると、汎化性能が低下（破滅的忘却）するリスクがある

### 過学習の防止

```python
# Early Stopping の設定
from transformers import EarlyStoppingCallback

trainer = SFTTrainer(
    ...
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,    # 3回評価で改善なければ停止
            early_stopping_threshold=0.001
        )
    ]
)
```

過学習の兆候:
- 学習ロスは下がるが検証ロスが上昇する
- 学習データにない図面での精度が低下する
- 出力形式が崩れる（JSONが壊れるなど）

### 学習データの品質チェック

```python
# データ検証スクリプト例
import json
from PIL import Image

def validate_dataset(data_path):
    with open(data_path) as f:
        dataset = json.load(f)

    errors = []
    for i, sample in enumerate(dataset):
        # 画像ファイルの存在確認
        try:
            img = Image.open(sample["image"])
            w, h = img.size
        except Exception as e:
            errors.append(f"Sample {i}: 画像読み込みエラー - {e}")
            continue

        # JSON応答の検証
        try:
            response = json.loads(sample["conversations"][1]["content"])
            for conn in response["connectors"]:
                x1, y1, x2, y2 = conn["bbox"]
                assert 0 <= x1 < x2 <= w, f"x座標が不正: {conn['bbox']}"
                assert 0 <= y1 < y2 <= h, f"y座標が不正: {conn['bbox']}"
        except Exception as e:
            errors.append(f"Sample {i}: アノテーションエラー - {e}")

    return errors
```

### 推奨される学習フロー

```
1. データ収集・アノテーション（100〜500枚）
2. データ検証（座標範囲、JSON形式チェック）
3. Train/Val 分割（80:20 または 90:10）
4. MLflow サーバー起動（docker compose up mlflow）
5. QLoRA ファインチューニング開始
6. 検証ロスのモニタリング（MLflow UI: http://localhost:5000）
7. Early Stopping または 3 epoch で停止
8. テストデータでの評価（IoU, mAP）
9. 実図面での動作確認
10. 必要に応じて追加データ収集・再学習
```

### MLflow との連携

学習コンテナから MLflow コンテナに接続するには環境変数を設定する。

```python
import os
os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"  # docker-compose 内はコンテナ名で解決

import mlflow
mlflow.set_experiment("connector-detection-qlora")

with mlflow.start_run():
    mlflow.log_params({"model": "gemma4-27b", "lora_r": 16, "epochs": 3})
    # TrainingArguments の report_to="mlflow" により自動でメトリクスが記録される
```

### VRAM 使用量の削減テクニック

```python
# テクニック1: 勾配チェックポイント（速度 vs VRAM トレードオフ）
model.gradient_checkpointing_enable()

# テクニック2: Flash Attention 2 の利用（高速化 + VRAM削減）
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-27b-it",
    attn_implementation="flash_attention_2",
    ...
)

# テクニック3: バッチサイズ1 + gradient accumulation
# 実効バッチサイズ8 を実現する場合:
per_device_train_batch_size = 1
gradient_accumulation_steps = 8

# テクニック4: 短いシーケンス長の設定
max_seq_length = 2048  # 4096 より VRAM を削減
```
