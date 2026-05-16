#!/usr/bin/env python3
"""
quickstart.py – End-to-End Demo Script
Run this to see the full pipeline in action on sample images.
Automatically downloads a pretrained YOLOv8 model (no training needed for demo).

Usage:
    python quickstart.py --demo_images path/to/images/
    python quickstart.py --demo_video  path/to/video.mp4
"""

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import time


CLASS_NAMES  = {0: "person", 1: "car"}
CLASS_COLORS = {0: (0, 255, 80), 1: (0, 160, 255)}


def demo_on_images(image_dir: str, weights: str = "yolov8n.pt",
                   output_dir: str = "outputs/quickstart"):
    """Run detection + counting on a folder of images."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    images = sorted(Path(image_dir).glob("*.jpg"))[:9]

    if not images:
        print(f"⚠️  No .jpg images found in: {image_dir}")
        return

    print(f"\n🚁 Drone Detection Demo")
    print(f"   Model  : {weights}")
    print(f"   Images : {len(images)}")
    print(f"   Output : {output_dir}\n")

    results_data = []
    annotated_frames = []

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        t0 = time.time()
        preds = model.predict(frame, conf=0.35, iou=0.45, verbose=False)[0]
        ms = (time.time() - t0) * 1000

        persons = cars = 0
        annotated = frame.copy()

        for box in preds.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in CLASS_NAMES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            color = CLASS_COLORS[cls_id]
            label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            ly = max(y1 - 4, th + 4)
            cv2.rectangle(annotated, (x1, ly - th - 3), (x1 + tw + 2, ly + 2), color, -1)
            cv2.putText(annotated, label, (x1 + 1, ly - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            if cls_id == 0: persons += 1
            else:           cars    += 1

        # Count overlay
        cv2.rectangle(annotated, (0, 0), (200, 70), (20, 20, 20), -1)
        cv2.putText(annotated, f"Humans : {persons}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2)
        cv2.putText(annotated, f"Cars   : {cars}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 160, 255), 2)

        annotated_frames.append(annotated)
        results_data.append({"name": img_path.name, "persons": persons, "cars": cars, "ms": ms})
        print(f"  ✓ {img_path.name:30s}  persons={persons:3d}  cars={cars:3d}  ({ms:.0f}ms)")

        # Save individual output
        cv2.imwrite(f"{output_dir}/det_{img_path.name}", annotated)

    # Grid visualization
    if annotated_frames:
        n = len(annotated_frames)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
        fig.suptitle("🚁 Drone Detection Results", fontsize=15, fontweight="bold")
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]

        for i, (frame, info) in enumerate(zip(annotated_frames, results_data)):
            axes_flat[i].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            axes_flat[i].set_title(f"👤{info['persons']}  🚗{info['cars']}  |  {info['name']}", fontsize=9)
            axes_flat[i].axis("off")

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis("off")

        plt.tight_layout()
        grid_path = f"{output_dir}/detection_grid.png"
        plt.savefig(grid_path, dpi=150, bbox_inches="tight")
        print(f"\n✅ Grid saved: {grid_path}")
        plt.show()

    # Summary
    total_p = sum(r["persons"] for r in results_data)
    total_c = sum(r["cars"]    for r in results_data)
    avg_ms  = np.mean([r["ms"] for r in results_data])
    print(f"\n📊 Summary:")
    print(f"   Total persons : {total_p}")
    print(f"   Total cars    : {total_c}")
    print(f"   Avg FPS       : {1000/avg_ms:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone Detection Quickstart Demo")
    parser.add_argument("--demo_images", type=str, default=None,  help="Folder of images")
    parser.add_argument("--demo_video",  type=str, default=None,  help="Video file")
    parser.add_argument("--weights",     type=str, default="yolov8n.pt",
                        help="Weights path. Use 'yolov8n.pt' for pretrained COCO (person+car classes 0,2).")
    parser.add_argument("--output",      type=str, default="outputs/quickstart")
    args = parser.parse_args()

    if args.demo_images:
        demo_on_images(args.demo_images, args.weights, args.output)
    elif args.demo_video:
        from src.track import track_video
        track_video(args.weights, args.demo_video, args.output)
    else:
        print("Usage: python quickstart.py --demo_images path/to/images/")
        print("       python quickstart.py --demo_video  path/to/video.mp4")
        print("\nNo source provided. Using sample synthetic demo...")

        # Synthetic test: generate a noise image if no data provided
        dummy = np.zeros((640, 1280, 3), dtype=np.uint8)
        dummy[:] = (40, 40, 40)
        cv2.putText(dummy, "No image source provided.", (300, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 2)
        cv2.putText(dummy, "Run: python quickstart.py --demo_images <path>", (200, 370),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (160, 160, 160), 1)
        cv2.imwrite("outputs/quickstart/demo_placeholder.png", dummy)
        print("Created placeholder at outputs/quickstart/demo_placeholder.png")
