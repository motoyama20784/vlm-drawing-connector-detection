import base64
import os

import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


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
    return resp.json()["response"]
