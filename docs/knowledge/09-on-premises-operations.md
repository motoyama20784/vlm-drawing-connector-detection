# オンプレ運用の考え方

## 概要

本プロジェクトは、クラウド API を使わずオンプレミス（自社サーバー）で Gemma 4 を動かすことを前提としています。本ドキュメントでは、オンプレ運用のメリット・デメリット・コスト感・ベストプラクティスを整理します。

---

## 1. オンプレのメリット

### 1-1. トークン使用料ゼロ

最も直接的なコスト優位性です。

- API を使う場合: 推論のたびにトークン課金が発生（入力・出力それぞれ課金）
- オンプレの場合: **電気代のみ**（GPU サーバーの消費電力）

```
例: GPT-4o API でのコスト試算
  入力 2,000 トークン × 1,000 回/日 = 2,000,000 トークン/日
  $5.00 / 1M トークン × 2 = $10/日 = 約300,000円/月

オンプレ (RTX 5090 x1, 消費電力 ~500W):
  500W × 24h × 30日 = 360kWh/月
  30円/kWh × 360 = 約10,800円/月
```

> **試算は概算です。** API の種類や実際の使用量、電力単価によって大きく変わります。ただし、大量推論になるほどオンプレが経済的になる傾向は明確です。

### 1-2. データプライバシー

産業用途（機械設計図面・回路図面・設備図面など）では特に重要です。

- 図面データが **外部サーバーに送信されない**
- NDA・秘密保持契約の対象データも安全に処理可能
- GDPR・個人情報保護法などのコンプライアンス上のリスクを低減
- セキュリティ要件が厳しい顧客への導入が可能

### 1-3. 低レイテンシ

- ネットワーク経由の RTT（往復遅延）が発生しない
- LAN 内での通信のみ → 安定した応答時間
- ネットワーク障害の影響を受けない（オフライン運用が可能）

典型的なレイテンシ比較:
| 方式 | 初回トークンまでの時間 (TTFT) |
|------|------------------------------|
| クラウド API（海外） | 500ms〜2000ms |
| クラウド API（国内リージョン） | 200ms〜500ms |
| **オンプレ（LAN内）** | **50ms〜200ms** |

### 1-4. カスタマイズ自由度

- モデルのファインチューニングが可能（図面特化モデルへの改変）
- プロンプトテンプレートの最適化を自由に実施
- 出力形式（JSON スキーマなど）のカスタマイズ
- 特定のコネクタ種別をシステムに追加しやすい
- 推論パラメータ（温度、top-p など）を用途に合わせて固定可能

---

## 2. オンプレの難しさ・注意点

### 2-1. 初期セットアップコスト

CUDA / ドライバ / フレームワークの互換性問題が最大のハードルです。

```
よくある問題:
  - CUDA バージョンが PyTorch の要件と合わない
  - ドライバが古く Blackwell (sm_120) に対応していない
  - bitsandbytes が CUDA バージョンに未対応
  - Flash Attention のビルドに失敗する
```

対策:
- Docker で環境を固定する（後述の `07-docker-environment.md` 参照）
- バージョン組み合わせを事前に検証してから本番投入
- `nvidia-smi` / `nvcc --version` / `python -c "import torch; print(torch.version.cuda)"` で常に確認

### 2-2. モデルの更新・管理

- Gemma 4 27B の 4-bit 量子化モデルでも **14〜17GB** のファイルサイズ
- 複数バージョンを保持するとストレージを圧迫
- モデルの更新には長時間のダウンロードが必要

```bash
# モデルのバージョン管理例（シンボリックリンクで切り替え）
/models/
  gemma4-27b-q4km-v1.0.gguf  (14GB)
  gemma4-27b-q4km-v1.1.gguf  (14GB)
  current -> gemma4-27b-q4km-v1.1.gguf  # シンボリックリンク

# ダウンロードにかかる時間の目安
# 14GB @ 100Mbps = 約 18 分
# 14GB @ 1Gbps  = 約  2 分
```

### 2-3. スケーラビリティの制限

- リクエスト増加に対してハードウェアの追加が必要（クラウドのような即時スケールアウト不可）
- GPU 追加・サーバー増設には調達リードタイムが発生
- 複数 GPU/サーバーでの負荷分散は構成が複雑になる

### 2-4. ソフトウェアバージョン依存関係

典型的な依存関係スタック:

```
Linux カーネル
  └─ NVIDIA ドライバ (570.xx+)
      └─ CUDA Toolkit (12.8+)
          └─ cuDNN (9.x+)
              └─ PyTorch (2.6+)
                  ├─ Transformers (4.49+)
                  ├─ bitsandbytes (0.43+)
                  ├─ vLLM (0.6+)
                  └─ Flash Attention 2
```

一つのバージョンを上げると連鎖的に他のパッケージも更新が必要になることがあります。

対策:
- Docker イメージでバージョンを固定
- 更新はステージング環境で先に検証
- `requirements.txt` でバージョンを固定して管理

### 2-5. モデルの監視・ヘルスチェック

- GPU 温度・VRAM 使用量の監視が必要
- モデルサービスのクラッシュ検知と自動再起動
- 推論レイテンシの劣化検知

```bash
# GPU 監視の基本コマンド
nvidia-smi dmon -s u  # VRAM 使用量を連続表示
nvidia-smi dmon -s t  # 温度を連続表示

# dcgm (Data Center GPU Manager) による詳細監視
sudo apt install datacenter-gpu-manager
dcgmi diag -r 1  # 診断テスト
```

---

## 3. コスト比較: API vs オンプレ

### 試算の前提条件

| 項目 | 値 |
|------|----|
| 処理件数 | 1,000 図面/日 |
| 1 図面あたりの入力トークン | 2,000 tokens |
| 1 図面あたりの出力トークン | 500 tokens |
| 稼働日数 | 250 日/年 |

### API コスト（GPT-4o 相当）

```
入力: 2,000 tokens × 1,000 件/日 × 250 日 = 500,000,000 tokens/年
出力: 500 tokens  × 1,000 件/日 × 250 日 = 125,000,000 tokens/年

コスト（目安）:
  入力: $5.00 / 1M tokens × 500M = $2,500/年
  出力: $15.00 / 1M tokens × 125M = $1,875/年
  合計: $4,375/年 ≈ 約 65 万円/年

※ 価格は変動します。最新の API 料金を必ず確認してください。
```

### オンプレコスト（RTX 5090 × 1）

```
ハードウェア:
  RTX 5090:       約 32 万円（5年償却 → 6.4 万円/年）
  サーバー本体:    約 30 万円（5年償却 → 6.0 万円/年）
  合計ハードウェア: 約 12.4 万円/年

電力:
  消費電力 (GPU + システム全体): 約 600W
  稼働時間: 8時間/日 × 250日 = 2,000時間/年
  電力量: 600W × 2,000h = 1,200kWh/年
  電気代: 1,200 × 30円/kWh = 約 3.6 万円/年

保守・管理工数:
  月1回のメンテナンス × 2時間 × 12ヶ月 = 24時間/年
  5,000円/時 × 24h = 約 12 万円/年

合計: 約 28 万円/年
```

### 比較まとめ

| 項目 | API | オンプレ |
|------|-----|---------|
| 固定費 | 低（初期投資なし） | 高（ハードウェア投資） |
| 変動費 | 高（使えば使うほど増加） | 低（主に電気代） |
| 年間コスト目安 | ~65 万円/年 | ~28 万円/年 |
| 損益分岐点 | - | ハードウェア費用の回収期間 ~1〜2年 |

> **免責**: 上記はあくまで概算です。実際の導入コストは使用量・API 価格・電力単価・人件費などによって大きく異なります。

---

## 4. 運用安定化のためのベストプラクティス

### 4-1. systemd によるサービス管理

```ini
# /etc/systemd/system/vlm-inference.service
[Unit]
Description=VLM Inference Service (Gemma 4)
After=network.target nvidia-persistenced.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/vlm-connector-detection
ExecStart=/opt/vlm-connector-detection/venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model /models/gemma4-27b \
    --quantization awq \
    --port 8000
Restart=always
RestartSec=10
Environment=CUDA_VISIBLE_DEVICES=0

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable vlm-inference
sudo systemctl start vlm-inference
```

### 4-2. NVIDIA Persistence Mode の有効化

GPU の初期化時間を短縮し、安定した応答時間を確保します。

```bash
# Persistence Mode の有効化
sudo nvidia-smi -pm 1

# 起動時に自動適用（/etc/rc.local または systemd service で）
sudo nvidia-persistenced --user root
```

### 4-3. ヘルスチェックエンドポイント

```python
# FastAPI を使ったヘルスチェック例
from fastapi import FastAPI
import torch

app = FastAPI()

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "gpu": torch.cuda.get_device_name(0),
        "vram_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
    }
```

### 4-4. ログ管理

```bash
# ログローテーションの設定
# /etc/logrotate.d/vlm-inference
/var/log/vlm-inference/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
}
```

### 4-5. モデルファイルのバックアップと整合性確認

```bash
# モデルファイルの SHA256 ハッシュを保存
sha256sum /models/gemma4-27b-q4km.gguf > /models/gemma4-27b-q4km.gguf.sha256

# 起動前の整合性チェック（破損・改ざん検知）
sha256sum -c /models/gemma4-27b-q4km.gguf.sha256
```

### 4-6. GPU 温度アラート

```bash
# GPU 温度が 85℃ を超えたらアラートを送るスクリプト例
#!/bin/bash
TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
THRESHOLD=85

if [ "$TEMP" -gt "$THRESHOLD" ]; then
    echo "WARNING: GPU temperature is ${TEMP}°C" | mail -s "GPU Temperature Alert" admin@example.com
fi
```

### 4-7. 定期的なモデル性能評価

- 推論結果のサンプリングと品質確認を定期的に実施
- コネクタ検出の精度指標（精度・再現率）を記録
- モデルのドリフト（精度低下）を早期に検知

```bash
# cron で毎日 2:00 AM に評価スクリプトを実行
0 2 * * * /opt/vlm-connector-detection/scripts/eval_model.sh >> /var/log/vlm-eval.log 2>&1
```

---

## 5. 障害発生時の対応フロー

```
推論エラー発生
  │
  ├─ VRAM OOM？
  │   └─ torch.cuda.empty_cache() → サービス再起動
  │
  ├─ GPU が応答しない？
  │   └─ nvidia-smi で確認 → ドライバ再起動 → 最悪 OS 再起動
  │
  ├─ レイテンシが急増？
  │   └─ GPU 温度確認 → サーマルスロットリング対応（冷却確認）
  │
  └─ モデルの出力品質が低下？
      └─ プロンプト・量子化設定の見直し → モデル再ロード
```
