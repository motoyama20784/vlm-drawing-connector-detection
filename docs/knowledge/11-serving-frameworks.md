# 推論サービングフレームワーク 比較・ライセンス

## フレームワーク概要比較

| | Ollama | vLLM | HF Transformers |
|--|--------|------|-----------------|
| ライセンス | MIT | Apache 2.0 | Apache 2.0 |
| 商用利用 | ✅ 可 | ✅ 可 | ✅ 可 |
| 起動難易度 | ★☆☆ | ★★☆ | ★★★ |
| モデル管理 | 自動（内蔵） | 手動 | 手動 |
| 量子化形式 | GGUF（4-bit標準） | AWQ / GPTQ / FP8 | bitsandbytes |
| 同時リクエスト | 弱い | 強い（Continuous Batching） | なし（コード次第） |
| OpenAI互換API | ✅ | ✅ | ❌（自前実装が必要） |
| ファインチューニング | ❌ | ❌ | ✅ |

---

## OpenAI互換APIとは何か

「OpenAI互換API」とはOpenAIのクラウドサービスを使うことではなく、**APIのリクエスト・レスポンス形式がOpenAIと同じ**であることを指す。

```
OpenAIの規格（形式）
  POST /v1/chat/completions
  { "model": "...", "messages": [...] }

↑ この形式に対応しているだけ。通信先は自分のマシン。
```

### なぜ便利か

`base_url` を変えるだけで、OpenAI向けに書いたコードがそのままローカルモデルで動く：

```python
from openai import OpenAI

# ── OpenAI 本家（クラウド・有料）──────────────────
client = OpenAI(api_key="sk-...")

# ── ローカルの Ollama（オンプレ・無料）────────────
client = OpenAI(base_url="http://localhost:11434/v1", api_key="dummy")

# ── ローカルの vLLM（オンプレ・無料）─────────────
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# 呼び出し方はすべて同じ
response = client.chat.completions.create(
    model="gemma4:27b",
    messages=[{"role": "user", "content": "この図面のコネクタを検出して"}]
)
```

**トークン使用料は一切発生しない。** 通信は自分のマシン内で完結する。

---

## モデル本体のライセンス（重要）

フレームワークとは別に、**モデルの重み自体にもライセンスがある**。商用利用の際はフレームワークだけでなくモデルライセンスの確認が必要。

| モデル | ライセンス | 商用利用 | 主な制約 |
|--------|-----------|---------|---------|
| Gemma 4 | Gemma Terms of Use（Google独自） | ✅ 基本的に可 | 競合モデルの学習に使用不可、再配布不可 |
| DeepSeek-VL2 | MIT | ✅ 可 | ほぼ制約なし |
| LLaVA-NeXT | Apache 2.0 | ✅ 可 | ほぼ制約なし |
| InternVL2 | MIT | ✅ 可 | ほぼ制約なし |
| Qwen2-VL | Apache 2.0 | ✅ 可 | ほぼ制約なし |

### Gemma 4 の商用利用における注意点

- 製品・サービスへの組み込みは可能
- Gemma の重みを使って別のモデルを訓練・公開することは不可
- Google の [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/terms) に違反する用途は不可
- 社内利用・オンプレ業務利用は問題なし

---

## 選定指針

```
まず動かしたい（PoC）          → Ollama
大量の図面を処理したい          → vLLM
ファインチューニングもやる       → HF Transformers
商用プロダクトに組み込む         → ライセンス確認の上どれでも可
```
