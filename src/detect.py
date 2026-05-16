"""
detect.py – Inference Pipeline: Detection + Human Counting
Runs YOLOv8 inference on images or video, draws bounding boxes,
and overlays human + car counts.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
CLASS_NAMES  = {0: "person", 1: "car"}
CLASS_COLORS = {0: (0, 255, 80), 1: (0, 160, 255)}   # BGR
CONF_THRESH  = 0.35
IOU_THRESH   = 0.45


# ──────────────────────────────────────────────
# Core Detection Logic
# ──────────────────────────────────────────────
def run_inference(model: YOLO, frame: np.ndarray, conf: float = CONF_THRESH, iou: float = IOU_THRESH):
    """
    Run YOLOv8 on a single frame.
    Returns: list of dicts with keys: cls_id, cls_name, conf, box (x1,y1,x2,y2)
    """
    results = model.predict(frame, conf=conf, iou=iou, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        if cls_id not in CLASS_NAMES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        detections.append({
            "cls_id":   cls_id,
            "cls_name": CLASS_NAMES[cls_id],
            "conf":     float(box.conf[0].item()),
            "box":      (x1, y1, x2, y2),
        })
    return detections


def count_objects(detections: list) -> dict:
    """Count detections by class."""
    counts = {"person": 0, "car": 0}
    for d in detections:
        if d["cls_name"] in counts:
            counts[d["cls_name"]] += 1
    return counts


# ──────────────────────────────────────────────
# Drawing
# ──────────────────────────────────────────────
def draw_detections(frame: np.ndarray, detections: list, counts: dict) -> np.ndarray:
    """Draw bounding boxes, labels, and count overlay on frame."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Draw bounding boxes
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = CLASS_COLORS[det["cls_id"]]
        label = f"{det['cls_name']} {det['conf']:.2f}"

        # Box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        label_y = max(y1 - 5, th + 5)
        cv2.rectangle(annotated, (x1, label_y - th - 4), (x1 + tw + 2, label_y + 2), color, -1)
        cv2.putText(annotated, label, (x1 + 1, label_y - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Count overlay panel (top-left)
    overlay_h, overlay_w = 80, 220
    overlay = annotated[:overlay_h, :overlay_w].copy()
    cv2.rectangle(annotated, (0, 0), (overlay_w, overlay_h), (20, 20, 20), -1)
    cv2.addWeighted(annotated[:overlay_h, :overlay_w], 0.5,
                    overlay, 0.5, 0, annotated[:overlay_h, :overlay_w])

    cv2.putText(annotated, f"Humans : {counts['person']}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 80), 2, cv2.LINE_AA)
    cv2.putText(annotated, f"Cars   : {counts['car']}", (12, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 160, 255), 2, cv2.LINE_AA)

    return annotated


# ──────────────────────────────────────────────
# Image Pipeline
# ──────────────────────────────────────────────
def process_image(model: YOLO, image_path: str, output_dir: str,
                  conf: float = CONF_THRESH, iou: float = IOU_THRESH) -> dict:
    """Process a single image: detect, count, draw, save."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"⚠️  Could not read: {image_path}")
        return {}

    t0 = time.time()
    detections = run_inference(model, frame, conf, iou)
    elapsed = time.time() - t0

    counts = count_objects(detections)
    annotated = draw_detections(frame, detections, counts)

    # Save
    out_path = Path(output_dir) / ("det_" + Path(image_path).name)
    cv2.imwrite(str(out_path), annotated)

    result = {
        "image": image_path,
        "detections": len(detections),
        "person_count": counts["person"],
        "car_count": counts["car"],
        "inference_ms": elapsed * 1000,
        "output": str(out_path),
    }
    print(f"  ✓ {Path(image_path).name:30s}  "
          f"persons={counts['person']:3d}  cars={counts['car']:3d}  "
          f"({elapsed*1000:.1f}ms)")
    return result


def process_folder(model: YOLO, image_dir: str, output_dir: str,
                   conf: float = CONF_THRESH, iou: float = IOU_THRESH) -> list:
    """Process all images in a folder."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts]

    print(f"\n🔍 Processing {len(images)} images from: {image_dir}")
    results = []
    for img in sorted(images):
        r = process_image(model, str(img), output_dir, conf, iou)
        if r:
            results.append(r)

    if results:
        total_persons = sum(r["person_count"] for r in results)
        total_cars    = sum(r["car_count"]    for r in results)
        avg_ms = sum(r["inference_ms"] for r in results) / len(results)
        print(f"\n📊 Summary:")
        print(f"   Images processed : {len(results)}")
        print(f"   Total persons    : {total_persons}")
        print(f"   Total cars       : {total_cars}")
        print(f"   Avg inference    : {avg_ms:.1f}ms  (~{1000/avg_ms:.0f} FPS)")

    return results


# ──────────────────────────────────────────────
# Video Pipeline
# ──────────────────────────────────────────────
def process_video(model: YOLO, video_path: str, output_dir: str,
                  conf: float = CONF_THRESH, iou: float = IOU_THRESH) -> str:
    """Process a video file frame by frame."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️  Could not open video: {video_path}")
        return ""

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = str(Path(output_dir) / ("det_" + Path(video_path).stem + ".mp4"))
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    print(f"\n🎬 Processing video: {video_path}")
    print(f"   Resolution: {width}x{height}  FPS: {fps:.1f}  Frames: {total}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = run_inference(model, frame, conf, iou)
        counts     = count_objects(detections)
        annotated  = draw_detections(frame, detections, counts)

        # Add frame counter
        cv2.putText(annotated, f"Frame {frame_idx}/{total}", (width - 180, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{total}  persons={counts['person']}  cars={counts['car']}")

    cap.release()
    writer.release()
    print(f"\n✅ Video saved: {out_path}")
    return out_path


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone Detection + Human Counting")
    parser.add_argument("--source",  type=str, required=True,  help="Image file, folder, or video")
    parser.add_argument("--weights", type=str, required=True,  help="YOLOv8 weights (.pt)")
    parser.add_argument("--output",  type=str, default="outputs/images", help="Output directory")
    parser.add_argument("--conf",    type=float, default=CONF_THRESH)
    parser.add_argument("--iou",     type=float, default=IOU_THRESH)
    args = parser.parse_args()

    print(f"\n🚁 Loading model: {args.weights}")
    model = YOLO(args.weights)

    source = Path(args.source)
    if source.is_dir():
        process_folder(model, str(source), args.output, args.conf, args.iou)
    elif source.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        process_video(model, str(source), "outputs/videos", args.conf, args.iou)
    elif source.is_file():
        Path(args.output).mkdir(parents=True, exist_ok=True)
        process_image(model, str(source), args.output, args.conf, args.iou)
    else:
        print(f"❌ Source not found: {source}")
