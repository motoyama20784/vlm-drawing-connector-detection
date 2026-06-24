import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mlflow
import yaml

from src.detection.detector import detect, get_model_info, get_gpu_usage, print_model_info, print_gpu_usage
from src.detection.evaluator import evaluate
from src.detection.parser import parse_connectors
from src.visualization.draw_bboxes import draw_bboxes

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def _log_summary_figure(glb: dict, all_metrics: list, image_stems: list):
    """重要指標のサマリー図を生成して MLflow に artifact として保存する。"""
    DARK_BG  = "#0d1b2a"
    PANEL_BG = "#112236"
    GREEN    = "#4ade80"
    ORANGE   = "#ffa726"
    RED      = "#f87171"
    PURPLE   = "#c084fc"
    BLUE     = "#60a5fa"
    GRAY     = "#7a9cc0"

    def _color(v, good_high=True):
        if good_high:
            return GREEN if v >= 0.7 else ORANGE if v >= 0.4 else RED
        else:
            return GREEN if v <= 0.3 else ORANGE if v <= 0.6 else RED

    fig = plt.figure(figsize=(14, 8), facecolor=DARK_BG)
    fig.suptitle("Experiment Summary", color="white", fontsize=14, fontweight="bold", y=0.97)

    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.35,
                          left=0.07, right=0.97, top=0.88, bottom=0.08)

    # --- 1. グローバル P/R/F1 ゲージ ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PANEL_BG)
    labels = ["Precision", "Recall", "F1"]
    vals   = [glb["precision"], glb["recall"], glb["f1"]]
    colors = [_color(v) for v in vals]
    bars = ax1.barh(labels, vals, color=colors, height=0.5)
    for bar, v in zip(bars, vals):
        ax1.text(min(v + 0.02, 0.97), bar.get_y() + bar.get_height() / 2,
                 f"{v:.3f}", va="center", color="white", fontsize=10, fontweight="bold")
    ax1.set_xlim(0, 1)
    ax1.set_title("P / R / F1  (global)", color=GRAY, fontsize=10)
    ax1.tick_params(colors="white"); ax1.spines[:].set_color("#1e3a5f")
    ax1.set_facecolor(PANEL_BG)

    # --- 2. エラー分類（near_fp / ghost_fp / miss） ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PANEL_BG)
    err_labels = ["Miss rate", "Near FP\n(coord shift)", "Ghost FP\n(hallucination)"]
    err_vals   = [glb["miss_rate"], glb["near_fp_rate"], glb["ghost_fp_rate"]]
    err_colors = [_color(v, good_high=False) for v in err_vals]
    bars2 = ax2.barh(err_labels, err_vals, color=err_colors, height=0.5)
    for bar, v in zip(bars2, err_vals):
        ax2.text(min(v + 0.02, 0.97), bar.get_y() + bar.get_height() / 2,
                 f"{v:.3f}", va="center", color="white", fontsize=10, fontweight="bold")
    ax2.set_xlim(0, 1)
    ax2.set_title("Error breakdown  (global)", color=GRAY, fontsize=10)
    ax2.tick_params(colors="white"); ax2.spines[:].set_color("#1e3a5f")

    # --- 3. avg matched IoU ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(PANEL_BG)
    iou_val = glb["avg_matched_iou"]
    iou_color = _color(iou_val)
    theta = [0, iou_val * 3.14159]
    ax3.barh(["Avg IoU\n(TP)"], [iou_val], color=iou_color, height=0.4)
    ax3.barh(["Avg IoU\n(TP)"], [1 - iou_val], left=[iou_val], color="#1e3a5f", height=0.4)
    ax3.text(0.5, 0, f"{iou_val:.3f}", ha="center", va="center",
             color="white", fontsize=16, fontweight="bold", transform=ax3.transAxes)
    ax3.set_xlim(0, 1); ax3.set_title("Avg Matched IoU", color=GRAY, fontsize=10)
    ax3.tick_params(colors="white"); ax3.spines[:].set_color("#1e3a5f")

    # --- 4. 画像ごと F1 棒グラフ ---
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.set_facecolor(PANEL_BG)
    f1s = [m["f1"] for m in all_metrics]
    bar_colors = [_color(v) for v in f1s]
    ax4.bar(image_stems, f1s, color=bar_colors, width=0.5)
    for i, v in enumerate(f1s):
        ax4.text(i, v + 0.01, f"{v:.2f}", ha="center", color="white", fontsize=9)
    ax4.set_ylim(0, 1.1)
    ax4.set_title("F1 per image", color=GRAY, fontsize=10)
    ax4.tick_params(colors="white", axis="x", rotation=15)
    ax4.tick_params(colors="white", axis="y")
    ax4.spines[:].set_color("#1e3a5f")

    # --- 5. 凡例 ---
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(PANEL_BG)
    ax5.axis("off")
    patches = [
        mpatches.Patch(color=GREEN,  label="Good  (≥0.7 / ≤0.3)"),
        mpatches.Patch(color=ORANGE, label="Mid   (≥0.4 / ≤0.6)"),
        mpatches.Patch(color=RED,    label="Poor  (<0.4  / >0.6)"),
    ]
    ax5.legend(handles=patches, loc="center", fontsize=9,
               facecolor=PANEL_BG, labelcolor="white", edgecolor="#1e3a5f")
    ax5.set_title("Color scale", color=GRAY, fontsize=10)

    mlflow.log_figure(fig, "summary.png")
    plt.close(fig)


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

    with open(config["prompt"]["file"]) as f:
        prompt_text = f.read()

    outputs_base = Path(config["data"].get("outputs_dir", "data/outputs"))

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("connector-detection")

    print_model_info(config["model"])

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        run_name = run.info.run_name or run_id[:8]
        samples_path = Path(config["data"]["samples_dir"])
        image_mode = "masked" if "masking" in samples_path.parts else "original"
        outputs_dir = outputs_base / f"{run_name}_{run_id[:8]}_{image_mode}"
        json_dir = outputs_dir / "json"
        raw_dir  = outputs_dir / "raw"
        vis_dir  = outputs_dir / "vis"
        for d in (json_dir, raw_dir, vis_dir):
            d.mkdir(parents=True, exist_ok=True)

        mlflow.set_tag("output_dir", str(outputs_dir))
        mlflow.log_artifact(config_path, "config")

        mlflow.log_param("image_mode", image_mode)
        mlflow.log_param("samples_dir", config["data"]["samples_dir"])
        mlflow.log_param("model", config["model"])
        mlflow.log_param("prompt_file", config["prompt"]["file"])
        mlflow.log_param("prompt_type", config["prompt"]["type"])
        mlflow.log_param("temperature", config["params"]["temperature"])
        mlflow.log_param("top_p", config["params"]["top_p"])
        mlflow.log_param("top_k", config["params"]["top_k"])
        mlflow.log_param("seed", config["params"].get("seed", 42))
        if "num_ctx" in config["params"]:
            mlflow.log_param("num_ctx", config["params"]["num_ctx"])
        if "think" in config["params"]:
            mlflow.log_param("think", config["params"]["think"])
        if "thinking_timeout_seconds" in config["params"]:
            mlflow.log_param("thinking_timeout_seconds", config["params"]["thinking_timeout_seconds"])
        mlflow.log_param("iou_threshold", config["evaluation"]["iou_threshold"])

        model_info = get_model_info(config["model"])
        if "error" not in model_info:
            mlflow.log_param("model_family", model_info.get("family", "unknown"))
            mlflow.log_param("model_parameter_size", model_info.get("parameter_size", "unknown"))
            mlflow.log_param("model_quantization", model_info.get("quantization_level", "unknown"))
            if model_info.get("quantization_level") in ("F16", "BF16"):
                config["params"].setdefault("num_ctx", 32768)
                print(f"[num_ctx] F16/BF16モデルのため num_ctx={config['params']['num_ctx']} を適用")

        gpu_usage = get_gpu_usage()
        if "error" not in gpu_usage and gpu_usage.get("loaded_models"):
            first = gpu_usage["loaded_models"][0]
            mlflow.log_param("gpu_vram_gb", first.get("vram_gb", "unknown"))
            mlflow.log_param("gpu_processor", first.get("processor", "unknown"))

        mlflow.log_text(prompt_text, "prompt.txt")

        run_info = {
            "run_id": run_id,
            "run_name": run_name,
            "image_mode": image_mode,
            "mlflow_url": f"{MLFLOW_TRACKING_URI}/#/experiments/1/runs/{run_id}",
            "output_dir": str(outputs_dir),
        }
        (outputs_dir / "run_info.json").write_text(
            json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        samples_dir = config["data"]["samples_dir"]
        gt_dir = config["data"]["ground_truth_dir"]
        image_paths = sorted(
            list(Path(samples_dir).glob("*.png"))
            + list(Path(samples_dir).glob("*.jpg"))
        )

        all_metrics = []
        # グローバル集計用カウンタ
        totals = {
            "tp": 0, "fp": 0, "fn": 0,
            "gt_count": 0, "pred_count": 0,
            "near_fp_count": 0,
            "sum_matched_iou": 0.0,  # avg_matched_iou をTP数で重み付け合算するため
            "ghost_fp_count": 0,
            "duplicate_gt_count": 0,
            "merged_pred_count": 0,
            "parse_failed_count": 0,
        }

        for image_path in image_paths:
            t0 = time.time()
            raw_response, token_stats = detect(
                str(image_path),
                prompt_text,
                config["model"],
                config["params"],
            )
            inference_sec = time.time() - t0
            pred_boxes, parse_failed = parse_connectors(raw_response)
            gt_boxes = load_ground_truth(gt_dir, image_path.name)
            metrics = evaluate(
                pred_boxes, gt_boxes, config["evaluation"]["iou_threshold"]
            )

            stem = image_path.stem

            raw_path  = raw_dir  / f"{stem}.txt"
            json_path = json_dir / f"{stem}.json"
            vis_path  = vis_dir  / f"{stem}.jpg"

            raw_path.write_text(raw_response, encoding="utf-8")
            json_path.write_text(
                json.dumps({"connectors": pred_boxes}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            draw_bboxes(str(image_path), pred_boxes, str(vis_path))

            mlflow.log_metric(f"inference_sec_{stem}", round(inference_sec, 1))
            mlflow.log_metric(f"eval_count_{stem}", token_stats["eval_count"])
            mlflow.log_metric(f"prompt_eval_count_{stem}", token_stats["prompt_eval_count"])
            mlflow.log_metric(f"peak_vram_gb_{stem}", token_stats["peak_vram_gb"])
            mlflow.log_metric(f"weights_vram_gb_{stem}", token_stats["weights_vram_gb"])
            mlflow.log_metric(f"kv_cache_vram_gb_{stem}", token_stats["kv_cache_vram_gb"])
            mlflow.log_param(f"thinking_fallback_{stem}", token_stats["thinking_fallback"])
            mlflow.log_metric(f"parse_failed_{stem}", int(parse_failed))
            mlflow.log_metric(f"precision_{stem}", metrics["precision"])
            mlflow.log_metric(f"recall_{stem}", metrics["recall"])
            mlflow.log_metric(f"f1_{stem}", metrics["f1"])
            mlflow.log_metric(f"miss_rate_{stem}", metrics["miss_rate"])
            mlflow.log_metric(f"near_fp_rate_{stem}", metrics["near_fp_rate"])
            mlflow.log_metric(f"avg_matched_iou_{stem}", metrics["avg_matched_iou"])
            mlflow.log_metric(f"ghost_fp_rate_{stem}", metrics["ghost_fp_rate"])
            mlflow.log_metric(f"duplicate_gt_rate_{stem}", metrics["duplicate_gt_rate"])
            mlflow.log_metric(f"merged_pred_rate_{stem}", metrics["merged_pred_rate"])
            mlflow.log_text(raw_response, f"raw/{stem}.txt")
            mlflow.log_artifact(str(json_path), "json")
            mlflow.log_artifact(str(vis_path), "vis")

            all_metrics.append(metrics)

            # グローバル集計用カウンタを更新
            totals["tp"]               += metrics["tp"]
            totals["fp"]               += metrics["fp"]
            totals["fn"]               += metrics["fn"]
            totals["gt_count"]         += len(gt_boxes)
            totals["pred_count"]       += len(pred_boxes)
            totals["near_fp_count"]    += metrics["near_fp_count"]
            totals["sum_matched_iou"]  += metrics["avg_matched_iou"] * metrics["tp"]
            totals["ghost_fp_count"]   += metrics["ghost_fp_count"]
            totals["duplicate_gt_count"] += metrics["duplicate_gt_count"]
            totals["merged_pred_count"]  += metrics["merged_pred_count"]
            totals["parse_failed_count"] += int(parse_failed)

            print(
                f"\n{image_path.name}:\n"
                f"  パース   : {'失敗' if parse_failed else '成功'}\n"
                f"  基本     : P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}\n"
                f"  TP/FP/FN : {metrics['tp']} / {metrics['fp']} / {metrics['fn']}  (GT={len(gt_boxes)} pred={len(pred_boxes)})\n"
                f"  [A] miss_rate       = {metrics['miss_rate']:.3f}  (見逃し {metrics['fn']}件)\n"
                f"  [B] near_fp_rate    = {metrics['near_fp_rate']:.3f}  ({metrics['near_fp_count']}件)\n"
                f"  [C] avg_matched_iou = {metrics['avg_matched_iou']:.3f}\n"
                f"  [D] ghost_fp_rate   = {metrics['ghost_fp_rate']:.3f}  ({metrics['ghost_fp_count']}件)\n"
                f"  [E] duplicate_gt_rate = {metrics['duplicate_gt_rate']:.3f}  ({metrics['duplicate_gt_count']}件)\n"
                f"  [F] merged_pred_rate  = {metrics['merged_pred_rate']:.3f}  ({metrics['merged_pred_count']}件)"
            )

        if all_metrics:
            n = len(all_metrics)

            # --- マクロ平均（画像ごとの指標を単純平均） ---
            macro = {
                "precision":        sum(m["precision"]        for m in all_metrics) / n,
                "recall":           sum(m["recall"]           for m in all_metrics) / n,
                "f1":               sum(m["f1"]               for m in all_metrics) / n,
                "miss_rate":        sum(m["miss_rate"]        for m in all_metrics) / n,
                "near_fp_rate":     sum(m["near_fp_rate"]     for m in all_metrics) / n,
                "avg_matched_iou":  sum(m["avg_matched_iou"]  for m in all_metrics) / n,
                "ghost_fp_rate":    sum(m["ghost_fp_rate"]    for m in all_metrics) / n,
                "duplicate_gt_rate": sum(m["duplicate_gt_rate"] for m in all_metrics) / n,
                "merged_pred_rate": sum(m["merged_pred_rate"] for m in all_metrics) / n,
            }

            # --- グローバル集計（全TP/FP/FNを合算して再計算） ---
            g_tp   = totals["tp"]
            g_fp   = totals["fp"]
            g_fn   = totals["fn"]
            g_gt   = totals["gt_count"]
            g_pred = totals["pred_count"]
            g_p  = g_tp / (g_tp + g_fp) if (g_tp + g_fp) > 0 else 0.0
            g_r  = g_tp / (g_tp + g_fn) if (g_tp + g_fn) > 0 else 0.0
            g_f1 = 2 * g_p * g_r / (g_p + g_r) if (g_p + g_r) > 0 else 0.0
            glb = {
                "precision":        g_p,
                "recall":           g_r,
                "f1":               g_f1,
                "miss_rate":        g_fn / g_gt   if g_gt   > 0 else 0.0,
                "near_fp_rate":     totals["near_fp_count"]     / g_pred if g_pred > 0 else 0.0,
                "avg_matched_iou":  totals["sum_matched_iou"]   / g_tp   if g_tp   > 0 else 0.0,
                "ghost_fp_rate":    totals["ghost_fp_count"]    / g_pred if g_pred > 0 else 0.0,
                "duplicate_gt_rate": totals["duplicate_gt_count"] / g_gt if g_gt   > 0 else 0.0,
                "merged_pred_rate": totals["merged_pred_count"] / g_pred if g_pred > 0 else 0.0,
            }

            parse_failure_rate = totals["parse_failed_count"] / n

            # MLflow へ記録
            for key, val in macro.items():
                mlflow.log_metric(f"macro_{key}", val)
            for key, val in glb.items():
                mlflow.log_metric(f"global_{key}", val)
            mlflow.log_metric("global_parse_failure_rate", parse_failure_rate)

            image_stems = [p.stem for p in image_paths]
            _log_summary_figure(glb, all_metrics, image_stems)

            print(
                f"\n=== マクロ平均 ({n}画像・各画像均等重み) ===\n"
                f"  基本     : P={macro['precision']:.3f}  R={macro['recall']:.3f}  F1={macro['f1']:.3f}\n"
                f"  [A] miss_rate       = {macro['miss_rate']:.3f}\n"
                f"  [B] near_fp_rate    = {macro['near_fp_rate']:.3f}\n"
                f"  [C] avg_matched_iou = {macro['avg_matched_iou']:.3f}\n"
                f"  [D] ghost_fp_rate   = {macro['ghost_fp_rate']:.3f}\n"
                f"  [E] duplicate_gt_rate = {macro['duplicate_gt_rate']:.3f}\n"
                f"  [F] merged_pred_rate  = {macro['merged_pred_rate']:.3f}\n"
                f"\n=== グローバル集計 (GT={g_gt} pred={g_pred} TP={g_tp} FP={g_fp} FN={g_fn}) ===\n"
                f"  基本     : P={glb['precision']:.3f}  R={glb['recall']:.3f}  F1={glb['f1']:.3f}\n"
                f"  [A] miss_rate       = {glb['miss_rate']:.3f}\n"
                f"  [B] near_fp_rate    = {glb['near_fp_rate']:.3f}\n"
                f"  [C] avg_matched_iou = {glb['avg_matched_iou']:.3f}\n"
                f"  [D] ghost_fp_rate   = {glb['ghost_fp_rate']:.3f}\n"
                f"  [E] duplicate_gt_rate = {glb['duplicate_gt_rate']:.3f}\n"
                f"  [F] merged_pred_rate  = {glb['merged_pred_rate']:.3f}\n"
                f"\n=== パース失敗 ===\n"
                f"  失敗数 / 画像数 : {totals['parse_failed_count']} / {n}  (rate={parse_failure_rate:.3f})"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="実験設定ファイルのパス")
    args = parser.parse_args()
    run_experiment(args.config)
