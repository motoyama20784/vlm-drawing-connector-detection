import base64
import os

import requests

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


def detect(image_path: str, prompt: str, model: str, params: dict) -> str:
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": params.get("temperature", 0.0),
            "top_p": params.get("top_p", 0.9),
            "top_k": params.get("top_k", 40),
        },
    }

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()

    print_gpu_usage()

    return resp.json()["response"]
