"""
dataset.py – VisDrone Dataset Utilities
Handles annotation conversion, class filtering, and dataset analysis.

VisDrone annotation format (per line):
  <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category>,<truncation>,<occlusion>

YOLO format (normalized, per line):
  <class_id> <cx> <cy> <w> <h>
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import argparse
import yaml


# ──────────────────────────────────────────────
# VisDrone class mapping → our 2-class schema
# ──────────────────────────────────────────────
VISDRONE_TO_OURS = {
    1: 0,   # pedestrian → person
    2: 0,   # people     → person
    4: 1,   # car        → car
    5: 1,   # van        → car (optional: comment out to exclude)
}
CLASS_NAMES = {0: "person", 1: "car"}
CLASS_COLORS = {0: (0, 255, 80), 1: (0, 140, 255)}   # BGR: green, orange


# ──────────────────────────────────────────────
# Annotation Conversion
# ──────────────────────────────────────────────
def convert_visdrone_to_yolo(annotation_path: str, image_path: str, output_path: str) -> dict:
    """
    Convert a single VisDrone annotation file to YOLO format.
    Returns stats dict: {total, kept, per_class}.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"total": 0, "kept": 0, "per_class": {}}
    h, w = img.shape[:2]

    stats = {"total": 0, "kept": 0, "per_class": defaultdict(int)}
    yolo_lines = []

    with open(annotation_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue

            x, y, bw, bh = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            score = int(parts[4])
            category = int(parts[5])

            stats["total"] += 1

            # Skip ignored regions (score=0) and unmapped classes
            if score == 0 or category not in VISDRONE_TO_OURS:
                continue

            # Skip degenerate boxes
            if bw <= 0 or bh <= 0:
                continue

            class_id = VISDRONE_TO_OURS[category]

            # Convert to YOLO normalized format
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h

            # Clamp to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            stats["kept"] += 1
            stats["per_class"][CLASS_NAMES[class_id]] += 1

    with open(output_path, "w") as f:
        f.write("\n".join(yolo_lines))

    return stats


def convert_dataset(data_root: str, splits: list = None):
    """Convert all VisDrone annotation files to YOLO format."""
    if splits is None:
        splits = ["VisDrone2019-DET-train", "VisDrone2019-DET-val"]

    total_stats = {"total": 0, "kept": 0, "per_class": defaultdict(int)}

    for split in splits:
        ann_dir = Path(data_root) / split / "annotations"
        img_dir = Path(data_root) / split / "images"
        label_dir = Path(data_root) / split / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)

        ann_files = sorted(ann_dir.glob("*.txt"))
        print(f"\n[{split}] Converting {len(ann_files)} annotations...")

        for ann_file in tqdm(ann_files):
            img_file = img_dir / ann_file.with_suffix(".jpg").name
            if not img_file.exists():
                img_file = img_dir / ann_file.with_suffix(".png").name
            label_file = label_dir / ann_file.name

            stats = convert_visdrone_to_yolo(str(ann_file), str(img_file), str(label_file))
            total_stats["total"] += stats["total"]
            total_stats["kept"] += stats["kept"]
            for cls, cnt in stats["per_class"].items():
                total_stats["per_class"][cls] += cnt

    print(f"\n✅ Conversion complete!")
    print(f"   Total annotations : {total_stats['total']:,}")
    print(f"   Kept (person+car) : {total_stats['kept']:,}")
    print(f"   Per class         : {dict(total_stats['per_class'])}")
    return total_stats


# ──────────────────────────────────────────────
# Dataset Analysis & Visualization
# ──────────────────────────────────────────────
def analyze_dataset(data_root: str, split: str = "VisDrone2019-DET-train", max_images: int = 500):
    """Analyze the dataset and return statistics."""
    label_dir = Path(data_root) / split / "labels"
    img_dir   = Path(data_root) / split / "images"

    label_files = sorted(label_dir.glob("*.txt"))[:max_images]

    stats = {
        "n_images": len(label_files),
        "n_objects": 0,
        "per_class": defaultdict(int),
        "widths": [],
        "heights": [],
        "objects_per_image": [],
        "box_areas": [],
    }

    for lf in tqdm(label_files, desc="Analyzing"):
        with open(lf) as f:
            lines = [l.strip() for l in f if l.strip()]

        stats["n_objects"] += len(lines)
        stats["objects_per_image"].append(len(lines))

        for line in lines:
            parts = line.split()
            cls_id = int(parts[0])
            w, h = float(parts[3]), float(parts[4])
            stats["per_class"][CLASS_NAMES.get(cls_id, str(cls_id))] += 1
            stats["widths"].append(w)
            stats["heights"].append(h)
            stats["box_areas"].append(w * h)

    return stats


def visualize_samples(data_root: str, split: str = "VisDrone2019-DET-train",
                       n_samples: int = 6, save_path: str = None):
    """Draw GT bounding boxes on sample images."""
    img_dir   = Path(data_root) / split / "images"
    label_dir = Path(data_root) / split / "labels"

    img_files = sorted(img_dir.glob("*.jpg"))[:n_samples]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("VisDrone – Sample Ground Truth Annotations", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for idx, img_file in enumerate(img_files):
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        label_file = label_dir / img_file.with_suffix(".txt").name
        ax = axes[idx]
        ax.imshow(img)

        person_count, car_count = 0, 0
        if label_file.exists():
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    px = (cx - bw / 2) * w
                    py = (cy - bh / 2) * h
                    pw = bw * w
                    ph = bh * h
                    color = "#00FF50" if cls_id == 0 else "#FF8C00"
                    rect = patches.Rectangle((px, py), pw, ph,
                                             linewidth=1, edgecolor=color, facecolor="none")
                    ax.add_patch(rect)
                    if cls_id == 0: person_count += 1
                    else: car_count += 1

        ax.set_title(f"👤 {person_count}  🚗 {car_count}", fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


def plot_dataset_stats(stats: dict, save_path: str = None):
    """Plot dataset statistics: class distribution, box sizes, density."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("VisDrone Dataset Statistics (Person + Car)", fontsize=13, fontweight="bold")

    # 1. Class distribution
    ax = axes[0]
    classes = list(stats["per_class"].keys())
    counts  = list(stats["per_class"].values())
    bars = ax.bar(classes, counts, color=["#00C853", "#FF6D00"], width=0.5, edgecolor="white")
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{cnt:,}", ha="center", va="bottom", fontsize=10)

    # 2. Bounding box size distribution
    ax = axes[1]
    areas = np.array(stats["box_areas"]) * 100  # percentage of image area
    ax.hist(areas, bins=60, color="#2979FF", edgecolor="none", alpha=0.85)
    ax.set_title("Bounding Box Area Distribution")
    ax.set_xlabel("% of Image Area")
    ax.set_ylabel("Frequency")
    ax.axvline(np.median(areas), color="red", linestyle="--", label=f"Median: {np.median(areas):.2f}%")
    ax.legend()

    # 3. Objects per image
    ax = axes[2]
    opp = stats["objects_per_image"]
    ax.hist(opp, bins=40, color="#AA00FF", edgecolor="none", alpha=0.85)
    ax.set_title("Objects per Image")
    ax.set_xlabel("Count")
    ax.set_ylabel("Frequency")
    ax.axvline(np.mean(opp), color="orange", linestyle="--", label=f"Mean: {np.mean(opp):.1f}")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisDrone dataset utilities")
    parser.add_argument("--convert",    action="store_true", help="Convert annotations to YOLO format")
    parser.add_argument("--analyze",    action="store_true", help="Analyze dataset statistics")
    parser.add_argument("--visualize",  action="store_true", help="Visualize sample annotations")
    parser.add_argument("--data_root",  type=str, default="./data", help="Dataset root directory")
    parser.add_argument("--split",      type=str, default="VisDrone2019-DET-train")
    parser.add_argument("--save_dir",   type=str, default="./outputs/images")
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    if args.convert:
        convert_dataset(args.data_root)

    if args.analyze:
        stats = analyze_dataset(args.data_root, args.split)
        print(f"\n📊 Dataset Analysis [{args.split}]")
        print(f"   Images       : {stats['n_images']:,}")
        print(f"   Total objects: {stats['n_objects']:,}")
        print(f"   Per class    : {dict(stats['per_class'])}")
        print(f"   Avg objects/image: {np.mean(stats['objects_per_image']):.1f}")
        print(f"   Median box area  : {np.median(stats['box_areas'])*100:.3f}% of image")
        plot_dataset_stats(stats, save_path=f"{args.save_dir}/dataset_stats.png")

    if args.visualize:
        visualize_samples(args.data_root, args.split,
                          save_path=f"{args.save_dir}/sample_annotations.png")
