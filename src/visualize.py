"""
visualize.py – Result Visualization Utilities
Grid plots, count histograms, tracking heatmaps, and comparison figures.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import List, Dict


CLASS_COLORS_MPL = {0: "#00C853", 1: "#FF6D00"}   # matplotlib hex


def create_detection_grid(image_paths: List[str], n_cols: int = 3,
                           title: str = "Detection Results",
                           save_path: str = None) -> None:
    """Create a grid of annotated detection images."""
    n = len(image_paths)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)

    if n_rows == 1:
        axes = [axes] if n_cols == 1 else list(axes)
    axes = np.array(axes).flatten()

    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(Path(path).name, fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_count_histogram(results: List[Dict], save_path: str = None) -> None:
    """
    Plot histogram of person and car counts across processed images.
    results: list of dicts with 'person_count' and 'car_count' keys.
    """
    persons = [r["person_count"] for r in results]
    cars    = [r["car_count"]    for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Object Count Distribution Across Images", fontsize=13, fontweight="bold")

    axes[0].hist(persons, bins=20, color="#00C853", edgecolor="white", alpha=0.85)
    axes[0].set_title("Person Count per Image")
    axes[0].set_xlabel("Count")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(np.mean(persons), color="red", linestyle="--",
                    label=f"Mean: {np.mean(persons):.1f}")
    axes[0].legend()

    axes[1].hist(cars, bins=20, color="#FF6D00", edgecolor="white", alpha=0.85)
    axes[1].set_title("Car Count per Image")
    axes[1].set_xlabel("Count")
    axes[1].set_ylabel("Frequency")
    axes[1].axvline(np.mean(cars), color="red", linestyle="--",
                    label=f"Mean: {np.mean(cars):.1f}")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def create_tracking_heatmap(trails: Dict[int, List], frame_size: tuple,
                             class_assignments: Dict[int, int] = None,
                             save_path: str = None) -> None:
    """
    Create a heatmap of where objects have been detected across all frames.
    trails: {track_id: [(cx, cy), ...]}
    """
    h, w = frame_size
    heatmap_person = np.zeros((h, w), dtype=np.float32)
    heatmap_car    = np.zeros((h, w), dtype=np.float32)

    for tid, trail in trails.items():
        cls_id = class_assignments.get(tid, 0) if class_assignments else 0
        for cx, cy in trail:
            if 0 <= cx < w and 0 <= cy < h:
                if cls_id == 0:
                    heatmap_person[cy, cx] += 1
                else:
                    heatmap_car[cy, cx] += 1

    # Gaussian blur for smoothing
    heatmap_person = cv2.GaussianBlur(heatmap_person, (51, 51), 0)
    heatmap_car    = cv2.GaussianBlur(heatmap_car,    (51, 51), 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Object Movement Heatmaps", fontsize=14, fontweight="bold")

    for ax, hmap, title, cmap in [
        (axes[0], heatmap_person, "Person Density", "Greens"),
        (axes[1], heatmap_car,    "Car Density",    "Oranges"),
    ]:
        im = ax.imshow(hmap, cmap=cmap, origin="upper")
        ax.set_title(title)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_curves(results_csv: str, save_path: str = None) -> None:
    """Plot loss and mAP curves from YOLOv8 results.csv."""
    import pandas as pd
    df = pd.read_csv(results_csv, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Curves – YOLOv8 on VisDrone", fontsize=14, fontweight="bold")

    plots = [
        ("train/box_loss", "val/box_loss", "Box Loss",  axes[0, 0]),
        ("train/cls_loss", "val/cls_loss", "Class Loss", axes[0, 1]),
        ("metrics/precision(B)", None, "Precision",     axes[1, 0]),
        ("metrics/mAP50(B)",     "metrics/mAP50-95(B)", "mAP", axes[1, 1]),
    ]

    for train_col, val_col, title, ax in plots:
        epochs = df["epoch"] if "epoch" in df.columns else range(len(df))
        if train_col in df.columns:
            ax.plot(epochs, df[train_col], label="train", color="#2979FF")
        if val_col and val_col in df.columns:
            ax.plot(epochs, df[val_col], label="val", color="#FF6D00")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def create_summary_card(image_path: str, person_count: int, car_count: int,
                         model_name: str = "YOLOv8n", save_path: str = None) -> None:
    """Create a nice summary card for a single processed image."""
    img = cv2.imread(image_path)
    if img is None:
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    fig = plt.figure(figsize=(10, 6))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1], figure=fig)

    ax_img  = fig.add_subplot(gs[0])
    ax_info = fig.add_subplot(gs[1])

    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title(Path(image_path).name, fontsize=9, color="gray")

    ax_info.axis("off")
    ax_info.set_facecolor("#111111")

    info_text = (
        f"🚁  DRONE DETECTION\n\n"
        f"Model:  {model_name}\n\n"
        f"👤  Persons:  {person_count}\n\n"
        f"🚗  Cars:     {car_count}\n\n"
        f"Resolution:\n{w} × {h} px"
    )
    ax_info.text(0.1, 0.5, info_text, transform=ax_info.transAxes,
                 fontsize=12, verticalalignment="center",
                 fontfamily="monospace", color="white",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#1E1E1E", edgecolor="#444"))

    fig.patch.set_facecolor("#0D0D0D")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
