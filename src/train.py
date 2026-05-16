"""
train.py – YOLOv8 Fine-Tuning on VisDrone
Trains a YOLOv8 model for person + car detection in aerial/drone imagery.
"""

import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch


def train(config_path: str = "configs/train_config.yaml"):
    """Load config and launch YOLOv8 training."""

    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("\n" + "=" * 60)
    print("  🚁 Drone Detection – YOLOv8 Training")
    print("=" * 60)
    print(f"  Model   : {cfg['model']}")
    print(f"  Dataset : {cfg['data']}")
    print(f"  Epochs  : {cfg['epochs']}")
    print(f"  Image   : {cfg['imgsz']}px")
    print(f"  Batch   : {cfg['batch']}")
    print(f"  Device  : {'GPU ' + str(cfg['device']) if cfg['device'] != 'cpu' else 'CPU'}")
    print("=" * 60 + "\n")

    # Detect available device
    if cfg.get("device", 0) != "cpu" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU. Training will be slow.")
        cfg["device"] = "cpu"

    # Load pretrained YOLOv8
    model = YOLO(cfg["model"])

    # Fine-tune
    results = model.train(
        data=cfg["data"],
        epochs=cfg["epochs"],
        batch=cfg["batch"],
        imgsz=cfg["imgsz"],
        workers=cfg.get("workers", 4),
        device=cfg["device"],

        # Optimizer
        optimizer=cfg.get("optimizer", "AdamW"),
        lr0=cfg.get("lr0", 0.001),
        lrf=cfg.get("lrf", 0.01),
        momentum=cfg.get("momentum", 0.937),
        weight_decay=cfg.get("weight_decay", 0.0005),
        warmup_epochs=cfg.get("warmup_epochs", 3),

        # Loss weights
        box=cfg.get("box", 7.5),
        cls=cfg.get("cls", 0.5),
        dfl=cfg.get("dfl", 1.5),

        # Augmentation
        hsv_h=cfg.get("hsv_h", 0.015),
        hsv_s=cfg.get("hsv_s", 0.7),
        hsv_v=cfg.get("hsv_v", 0.4),
        degrees=cfg.get("degrees", 10.0),
        translate=cfg.get("translate", 0.1),
        scale=cfg.get("scale", 0.5),
        fliplr=cfg.get("fliplr", 0.5),
        flipud=cfg.get("flipud", 0.3),
        mosaic=cfg.get("mosaic", 1.0),
        mixup=cfg.get("mixup", 0.15),
        copy_paste=cfg.get("copy_paste", 0.1),

        # Thresholds
        conf=cfg.get("conf", 0.35),
        iou=cfg.get("iou", 0.45),

        # Output
        project=cfg.get("project", "runs/detect"),
        name=cfg.get("name", "visdrone_yolov8n"),
        save=cfg.get("save", True),
        save_period=cfg.get("save_period", 10),
        plots=cfg.get("plots", True),
    )

    print("\n✅ Training complete!")
    print(f"   Best weights: {results.save_dir}/weights/best.pt")
    print(f"   Results dir : {results.save_dir}")

    # Print best metrics
    if hasattr(results, "results_dict"):
        d = results.results_dict
        print(f"\n📊 Final Metrics:")
        print(f"   mAP@0.5      : {d.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"   mAP@0.5:0.95 : {d.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        print(f"   Precision    : {d.get('metrics/precision(B)', 'N/A'):.4f}")
        print(f"   Recall       : {d.get('metrics/recall(B)', 'N/A'):.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on VisDrone")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--data",   type=str, default=None, help="Override data.yaml path")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch",  type=int, default=None, help="Override batch size")
    parser.add_argument("--imgsz",  type=int, default=None, help="Override image size")
    parser.add_argument("--weights",type=str, default=None, help="Starting weights (override config)")
    args = parser.parse_args()

    # Apply CLI overrides to config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.data:   cfg["data"] = args.data
    if args.epochs: cfg["epochs"] = args.epochs
    if args.batch:  cfg["batch"] = args.batch
    if args.imgsz:  cfg["imgsz"] = args.imgsz
    if args.weights:cfg["model"] = args.weights

    # Write temp config and train
    tmp = "configs/_train_tmp.yaml"
    with open(tmp, "w") as f:
        yaml.dump(cfg, f)
    train(tmp)
