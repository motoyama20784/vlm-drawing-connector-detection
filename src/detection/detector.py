import base64
import json
import os
import sys
import threading
import time

import requests
from tqdm import tqdm

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_model_info(model: str) -> dict:
    """/api/show でモデルの詳細情報（量子化レベル等）を返す。"""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/show",
            json={"name": model},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        details = data.get("details", {})
        return {
            "model": model,
            "parameter_size": details.get("parameter_size", "unknown"),
            "quantization_level": details.get("quantization_level", "unknown"),
            "family": details.get("family", "unknown"),
        }
    except Exception as e:
        return {"model": model, "error": str(e)}


def get_gpu_usage() -> dict:
    """/api/ps で現在ロード中モデルのGPU使用量を返す。"""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/ps", timeout=10)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        if not models:
            return {"loaded_models": []}
        result = []
        for m in models:
            size_vram = m.get("size_vram", 0)
            size_total = m.get("size", 0)
            result.append({
                "name": m.get("name", "unknown"),
                "processor": m.get("details", {}).get("processor", m.get("processor", "unknown")),
                "vram_gb": round(size_vram / 1024 ** 3, 2),
                "total_size_gb": round(size_total / 1024 ** 3, 2),
            })
        return {"loaded_models": result}
    except Exception as e:
        return {"error": str(e)}


def print_model_info(model: str) -> None:
    info = get_model_info(model)
    if "error" in info:
        print(f"[model info] 取得失敗: {info['error']}")
        return
    print(
        f"[model] {info['model']} | "
        f"params={info['parameter_size']} | "
        f"quant={info['quantization_level']} | "
        f"family={info['family']}"
    )


def print_gpu_usage() -> None:
    usage = get_gpu_usage()
    if "error" in usage:
        print(f"[gpu] 取得失敗: {usage['error']}")
        return
    if not usage["loaded_models"]:
        print("[gpu] ロード中のモデルなし")
        return
    for m in usage["loaded_models"]:
        print(
            f"[gpu] {m['name']} | "
            f"VRAM={m['vram_gb']}GB / total={m['total_size_gb']}GB | "
            f"processor={m['processor']}"
        )


def detect(image_path: str, prompt: str, model: str, params: dict) -> tuple[str, dict]:
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    options = {
        "temperature": params.get("temperature", 0.0),
        "top_p": params.get("top_p", 0.9),
        "top_k": params.get("top_k", 40),
        "seed": params.get("seed", 42),
    }
    if "num_ctx" in params:
        options["num_ctx"] = params["num_ctx"]
    if "num_predict" in params:
        options["num_predict"] = params["num_predict"]
    thinking_timeout = params.get("thinking_timeout_seconds")
    request_timeout = thinking_timeout or 120

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": True,
        "options": options,
    }
    if "think" in params:
        payload["think"] = params["think"]

    # 推論前のベースライン VRAM（≒モデル重み）を取得
    weights_gb = 0.0
    baseline = get_gpu_usage()
    for m in baseline.get("loaded_models", []):
        weights_gb = max(weights_gb, m["vram_gb"])

    peak_vram = [0.0]
    stop_event = threading.Event()

    def _monitor_vram():
        while not stop_event.is_set():
            usage = get_gpu_usage()
            for m in usage.get("loaded_models", []):
                if peak_vram[0] < m["vram_gb"]:
                    peak_vram[0] = m["vram_gb"]
            stop_event.wait(1.0)

    monitor_thread = threading.Thread(target=_monitor_vram, daemon=True)
    monitor_thread.start()

    print(f"[inference] リクエスト送信 → モデルロード/thinking待ち...", flush=True)
    result = _stream_generate(payload, thinking_timeout=thinking_timeout, request_timeout=request_timeout)

    thinking_fallback = False
    if result is None:
        thinking_fallback = True
        print(f"[thinking fallback] think: false でリトライ", flush=True)
        payload["think"] = False
        result = _stream_generate(payload, thinking_timeout=None, request_timeout=request_timeout)

    stop_event.set()
    monitor_thread.join()

    full_response, token_stats = result
    total_vram_gb = peak_vram[0]
    kv_cache_gb = round(max(total_vram_gb - weights_gb, 0.0), 2)

    token_stats["thinking_fallback"] = thinking_fallback
    token_stats["peak_vram_gb"] = total_vram_gb
    token_stats["weights_vram_gb"] = weights_gb
    token_stats["kv_cache_vram_gb"] = kv_cache_gb

    print_gpu_usage()
    return full_response, token_stats


def _stream_generate(
    payload: dict,
    thinking_timeout: int | None,
    request_timeout: int = 120,
) -> tuple[str, dict] | None:
    """ストリーミングで生成し (レスポンス文字列, トークン統計) を返す。
    thinking_timeout 秒を超えた場合は None を返す（fallback 用）。
    """
    read_timeout = thinking_timeout if thinking_timeout else request_timeout
    full_response = ""
    think_chunks = 0
    resp_chunks = 0
    token_stats = {"eval_count": 0, "prompt_eval_count": 0}
    start_time = time.time()

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            stream=True,
            timeout=(10, read_timeout),
        )
        resp.raise_for_status()
        print(f"[inference] 接続確立 → ストリーミング開始 (elapsed={time.time()-start_time:.0f}s)", flush=True)

        with tqdm(unit=" chunks", desc="generating", file=sys.stdout, dynamic_ncols=True) as pbar:
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    thinking_text = data.get("thinking", "")
                    response_text = data.get("response", "")
                    if thinking_text:
                        think_chunks += 1
                    if response_text:
                        resp_chunks += 1
                        full_response += response_text
                    pbar.update(1)
                    elapsed = time.time() - start_time
                    pbar.set_postfix(think=think_chunks, resp=resp_chunks, elapsed=f"{elapsed:.0f}s", refresh=True)
                    if data.get("done", False):
                        token_stats["eval_count"] = data.get("eval_count", 0)
                        token_stats["prompt_eval_count"] = data.get("prompt_eval_count", 0)
                        break
    except requests.exceptions.ReadTimeout:
        elapsed = time.time() - start_time
        print(f"\n[thinking timeout] {thinking_timeout}s 超過 (elapsed={elapsed:.0f}s) → キャンセル", flush=True)
        return None

    return full_response, token_stats
