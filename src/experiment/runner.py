import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import mlflow
import yaml

from src.detection.detector import detect, get_model_info, get_gpu_usage, print_model_info, print_gpu_usage
from src.detection.evaluator import evaluate
from src.detection.parser import parse_connectors
from src.visualization.draw_bboxes import draw_bboxes

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def load_ground_truth(gt_dir: str, image_name: str) -> list:
    gt_path = Path(gt_dir) / (Path(image_name).stem + ".json")
    if not gt_path.exists():
        return []
    with open(gt_path) as f:
        data = json.load(f)
    return data.get("connectors", [])


def run_experiment(config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_env = os.getenv("MODEL")
    if model_env:
        config["model"] = model_env

    with open(config["prompt"]["file"]) as f:
        prompt_text = f.read()

    outputs_dir = Path(config["data"].get("outputs_dir", "data/outputs"))
    outputs_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("connector-detection")

    print_model_info(config["model"])
    print_gpu_usage()

    with mlflow.start_run():
        mlflow.log_param("model", config["model"])
        mlflow.log_param("prompt_file", config["prompt"]["file"])
        mlflow.log_param("prompt_type", config["prompt"]["type"])
        mlflow.log_param("temperature", config["params"]["temperature"])
        mlflow.log_param("top_p", config["params"]["top_p"])
        mlflow.log_param("top_k", config["params"]["top_k"])
        mlflow.log_param("seed", config["params"].get("seed", 42))   
        mlflow.log_param("iou_threshold", config["evaluation"]["iou_threshold"])

        model_info = get_model_info(config["model"])
        if "error" not in model_info:
            mlflow.log_param("model_family", model_info.get("family", "unknown"))
            mlflow.log_param("model_parameter_size", model_info.get("parameter_size", "unknown"))
            mlflow.log_param("model_quantization", model_info.get("quantization_level", "unknown"))

        gpu_usage = get_gpu_usage()
        if "error" not in gpu_usage and gpu_usage.get("loaded_models"):
            first = gpu_usage["loaded_models"][0]
            mlflow.log_param("gpu_vram_gb", first.get("vram_gb", "unknown"))
            mlflow.log_param("gpu_processor", first.get("processor", "unknown"))

        mlflow.log_text(prompt_text, "prompt.txt")

        samples_dir = config["data"]["samples_dir"]
        gt_dir = config["data"]["ground_truth_dir"]
        image_paths = sorted(
            list(Path(samples_dir).glob("*.png"))
            + list(Path(samples_dir).glob("*.jpg"))
        )

        all_metrics = []
        for image_path in image_paths:
            raw_response = detect(
                str(image_path),
                prompt_text,
                config["model"],
                config["params"],
            )
            pred_boxes = parse_connectors(raw_response)
            gt_boxes = load_ground_truth(gt_dir, image_path.name)
            metrics = evaluate(
                pred_boxes, gt_boxes, config["evaluation"]["iou_threshold"]
            )

            stem = image_path.stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 生レスポンス保存（パース失敗調査用）
            raw_path = outputs_dir / f"{timestamp}_{stem}_raw.txt"
            raw_path.write_text(raw_response, encoding="utf-8")

            # JSON出力
            json_path = outputs_dir / f"{timestamp}_{stem}.json"
            json_path.write_text(
                json.dumps({"connectors": pred_boxes}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # bbox描画済み画像の出力
            vis_path = outputs_dir / f"{timestamp}_{stem}_vis.jpg"
            draw_bboxes(str(image_path), pred_boxes, str(vis_path))

            mlflow.log_metric(f"precision_{stem}", metrics["precision"])
            mlflow.log_metric(f"recall_{stem}", metrics["recall"])
            mlflow.log_metric(f"f1_{stem}", metrics["f1"])
            mlflow.log_text(raw_response, f"responses/{stem}.txt")
            mlflow.log_artifact(str(json_path), f"detections")
            mlflow.log_artifact(str(vis_path), f"visualizations")

            all_metrics.append(metrics)
            print(
                f"{image_path.name}: "
                f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f} | "
                f"検出数={len(pred_boxes)} | "
                f"JSON={json_path.name} | vis={vis_path.name}"
            )

        if all_metrics:
            avg_p = sum(m["precision"] for m in all_metrics) / len(all_metrics)
            avg_r = sum(m["recall"] for m in all_metrics) / len(all_metrics)
            avg_f1 = sum(m["f1"] for m in all_metrics) / len(all_metrics)
            mlflow.log_metric("precision", avg_p)
            mlflow.log_metric("recall", avg_r)
            mlflow.log_metric("f1", avg_f1)
            print(f"\n平均: P={avg_p:.3f} R={avg_r:.3f} F1={avg_f1:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="実験設定ファイルのパス")
    args = parser.parse_args()
    run_experiment(args.config)
