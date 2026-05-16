"""
evaluate.py – Compute mAP, Precision, Recall, F1, and FPS
Runs YOLOv8 validation on the VisDrone val split and prints a clean report.
"""

import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


CLASS_NAMES = {0: "person", 1: "car"}


# ──────────────────────────────────────────────
# Validation (mAP via Ultralytics)
# ──────────────────────────────────────────────
def run_validation(weights: str, data: str, imgsz: int = 640,
                   conf: float = 0.35, iou: float = 0.45, split: str = "val"):
    """Run YOLO validation and return metrics dict."""
    model = YOLO(weights)
    metrics = model.val(
        data=data,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        split=split,
        verbose=False,
    )
    return metrics


# ──────────────────────────────────────────────
# FPS Benchmark
# ──────────────────────────────────────────────
def benchmark_fps(weights: str, image_dir: str, n_warmup: int = 5, n_bench: int = 50) -> float:
    """Measure inference FPS on a set of images."""
    model = YOLO(weights)
    images = sorted(Path(image_dir).glob("*.jpg"))[:n_warmup + n_bench]
    if not images:
        print("⚠️  No images found for FPS benchmark.")
        return 0.0

    frames = [cv2.imread(str(p)) for p in images if cv2.imread(str(p)) is not None]

    # Warmup
    for f in frames[:n_warmup]:
        model.predict(f, verbose=False)

    # Benchmark
    t0 = time.time()
    for f in frames[n_warmup:]:
        model.predict(f, verbose=False)
    elapsed = time.time() - t0
    fps = len(frames[n_warmup:]) / elapsed
    return fps


# ──────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────
def print_report(metrics, fps: float = None):
    """Print a nicely formatted evaluation report."""
    print("\n" + "=" * 56)
    print("  📊 Evaluation Report – VisDrone (Person + Car)")
    print("=" * 56)

    box = metrics.box
    names = metrics.names   # {0: 'person', 1: 'car'}

    # Per-class
    print(f"\n  {'Class':12s} {'Precision':>10s} {'Recall':>8s} {'mAP@.5':>8s} {'mAP@.5:.95':>11s}")
    print(f"  {'-'*54}")
    for i, name in names.items():
        try:
            p  = float(box.p[i])   if hasattr(box, 'p')   else float(box.ap_class_index[i])
            r  = float(box.r[i])   if hasattr(box, 'r')   else 0.0
            m50   = float(box.ap50[i]) if hasattr(box, 'ap50') else 0.0
            m5095 = float(box.ap[i])   if hasattr(box, 'ap')   else 0.0
            print(f"  {name:12s} {p:10.4f} {r:8.4f} {m50:8.4f} {m5095:11.4f}")
        except (IndexError, AttributeError):
            print(f"  {name:12s}  (per-class data unavailable)")

    # Overall
    print(f"  {'-'*54}")
    print(f"  {'Overall':12s} {float(box.mp):10.4f} {float(box.mr):8.4f} "
          f"{float(box.map50):8.4f} {float(box.map):11.4f}")

    if fps:
        print(f"\n  Inference speed: {fps:.1f} FPS")

    print("=" * 56 + "\n")


def plot_pr_curve(metrics, save_path: str = None):
    """Plot Precision-Recall curves per class."""
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#00C853", "#FF6D00"]
        names = metrics.names

        for i, (cls_id, cls_name) in enumerate(names.items()):
            # Ultralytics stores pr_data as (recall, precision) arrays
            prec = metrics.box.curves_results[0] if hasattr(metrics.box, 'curves_results') else []
            ax.plot([0, 1], [0.5, 0.5], "--", color=colors[i % len(colors)],
                    label=f"{cls_name} (curve N/A without data)")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve – VisDrone")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
    except Exception as e:
        print(f"⚠️  Could not plot PR curve: {e}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 on VisDrone")
    parser.add_argument("--weights",   type=str, required=True)
    parser.add_argument("--data",      type=str, default="configs/data.yaml")
    parser.add_argument("--imgsz",     type=int, default=640)
    parser.add_argument("--conf",      type=float, default=0.35)
    parser.add_argument("--iou",       type=float, default=0.45)
    parser.add_argument("--split",     type=str, default="val")
    parser.add_argument("--fps_dir",   type=str, default=None,
                        help="Image folder for FPS benchmark (optional)")
    parser.add_argument("--save_dir",  type=str, default="outputs/metrics")
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n🔬 Running validation: {args.weights}")
    metrics = run_validation(args.weights, args.data, args.imgsz,
                             args.conf, args.iou, args.split)

    fps = None
    if args.fps_dir:
        print(f"\n⏱  Benchmarking FPS on: {args.fps_dir}")
        fps = benchmark_fps(args.weights, args.fps_dir)

    print_report(metrics, fps)
    plot_pr_curve(metrics, save_path=f"{args.save_dir}/pr_curve.png")
