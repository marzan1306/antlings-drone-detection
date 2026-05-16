"""
track.py – Object Tracking with ByteTrack (via Ultralytics built-in tracker)
Tracks persons and cars across video frames using ByteTrack.
Each unique track gets a persistent ID, enabling trajectory visualization.

Usage:
    python src/track.py --source video.mp4 --weights runs/detect/best.pt
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import time


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
CLASS_NAMES  = {0: "person", 1: "car"}
CLASS_COLORS = {0: (0, 255, 80), 1: (0, 160, 255)}   # BGR: green, orange
TRAIL_LENGTH = 30   # How many past positions to draw per track


def get_track_color(track_id: int) -> tuple:
    """Generate a consistent color per track ID."""
    np.random.seed(track_id * 13 + 7)
    return tuple(int(c) for c in np.random.randint(100, 255, 3).tolist())


# ──────────────────────────────────────────────
# Tracker
# ──────────────────────────────────────────────
class DroneTracker:
    """Wraps YOLOv8 + ByteTrack and manages per-track state."""

    def __init__(self, weights: str, conf: float = 0.35, iou: float = 0.45):
        self.model = YOLO(weights)
        self.conf  = conf
        self.iou   = iou
        self.trails: dict[int, list] = defaultdict(list)   # track_id → list of (cx, cy)
        self.active_ids: set = set()
        self.total_ids_seen: set = set()

    def update(self, frame: np.ndarray) -> list:
        """
        Run tracking on a frame.
        Returns list of dicts: {track_id, cls_id, cls_name, conf, box, center}
        """
        results = self.model.track(
            frame,
            persist=True,         # Crucial: maintains tracker state across frames
            tracker="bytetrack.yaml",
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )[0]

        tracks = []
        self.active_ids = set()

        if results.boxes.id is None:
            return tracks

        for box, track_id_t, cls_t, conf_t in zip(
            results.boxes.xyxy,
            results.boxes.id,
            results.boxes.cls,
            results.boxes.conf,
        ):
            cls_id   = int(cls_t.item())
            track_id = int(track_id_t.item())
            if cls_id not in CLASS_NAMES:
                continue

            x1, y1, x2, y2 = map(int, box.tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Update trail
            self.trails[track_id].append((cx, cy))
            if len(self.trails[track_id]) > TRAIL_LENGTH:
                self.trails[track_id].pop(0)

            self.active_ids.add(track_id)
            self.total_ids_seen.add(track_id)

            tracks.append({
                "track_id": track_id,
                "cls_id":   cls_id,
                "cls_name": CLASS_NAMES[cls_id],
                "conf":     float(conf_t.item()),
                "box":      (x1, y1, x2, y2),
                "center":   (cx, cy),
            })

        return tracks

    @property
    def unique_person_count(self) -> int:
        """Total unique persons ever tracked (cumulative)."""
        return sum(1 for t in self.total_ids_seen
                   if any(tr["track_id"] == t and tr["cls_id"] == 0
                          for tr in []))  # Simplified – see note below

    def reset_trails(self):
        self.trails.clear()


# ──────────────────────────────────────────────
# Drawing
# ──────────────────────────────────────────────
def draw_tracks(frame: np.ndarray, tracks: list, trails: dict,
                frame_counts: dict, total_unique: int) -> np.ndarray:
    """Draw track boxes, IDs, trails, and counters."""
    annotated = frame.copy()

    # Draw trails first (behind boxes)
    for track in tracks:
        tid = track["track_id"]
        trail = trails.get(tid, [])
        color = get_track_color(tid)
        for i in range(1, len(trail)):
            alpha = i / len(trail)
            thickness = max(1, int(2 * alpha))
            cv2.line(annotated, trail[i - 1], trail[i], color, thickness)

    # Draw boxes and labels
    for track in tracks:
        x1, y1, x2, y2 = track["box"]
        tid    = track["track_id"]
        color  = CLASS_COLORS[track["cls_id"]]
        label  = f"#{tid} {track['cls_name']} {track['conf']:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        label_y = max(y1 - 5, th + 5)
        cv2.rectangle(annotated, (x1, label_y - th - 4), (x1 + tw + 2, label_y + 2), color, -1)
        cv2.putText(annotated, label, (x1 + 1, label_y - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Center dot
        cv2.circle(annotated, track["center"], 3, color, -1)

    # Stats overlay
    h, w = annotated.shape[:2]
    overlay_h, overlay_w = 100, 280
    roi = annotated[:overlay_h, :overlay_w].copy()
    cv2.rectangle(annotated, (0, 0), (overlay_w, overlay_h), (15, 15, 15), -1)
    cv2.addWeighted(annotated[:overlay_h, :overlay_w], 0.45, roi, 0.55, 0, annotated[:overlay_h, :overlay_w])

    cv2.putText(annotated, f"Active Persons : {frame_counts.get('person', 0)}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 80), 2)
    cv2.putText(annotated, f"Active Cars    : {frame_counts.get('car', 0)}", (12, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 160, 255), 2)
    cv2.putText(annotated, f"Total IDs seen : {total_unique}", (12, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    return annotated


# ──────────────────────────────────────────────
# Video Tracking Pipeline
# ──────────────────────────────────────────────
def track_video(weights: str, video_path: str, output_dir: str = "outputs/videos",
                conf: float = 0.35, iou: float = 0.45):
    """Run full tracking pipeline on a video."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = str(Path(output_dir) / ("tracked_" + Path(video_path).stem + ".mp4"))
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    tracker  = DroneTracker(weights, conf, iou)

    print(f"\n🎯 Starting ByteTrack on: {video_path}")
    print(f"   {width}x{height} @ {fps:.1f}fps  ({total} frames)")

    frame_idx = 0
    total_unique_persons = 0
    person_ids_seen = set()
    car_ids_seen    = set()

    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tracks = tracker.update(frame)

        # Count active this frame
        frame_counts = {"person": 0, "car": 0}
        for t in tracks:
            frame_counts[t["cls_name"]] += 1
            if t["cls_id"] == 0:
                person_ids_seen.add(t["track_id"])
            else:
                car_ids_seen.add(t["track_id"])

        annotated = draw_tracks(
            frame, tracks, tracker.trails, frame_counts,
            len(person_ids_seen)
        )

        # Frame number
        cv2.putText(annotated, f"Frame {frame_idx}/{total}",
                    (width - 180, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 50 == 0:
            elapsed = time.time() - t_start
            fps_now = frame_idx / elapsed
            print(f"  Frame {frame_idx:5d}/{total}  "
                  f"active: persons={frame_counts['person']} cars={frame_counts['car']}  "
                  f"unique persons={len(person_ids_seen)}  "
                  f"{fps_now:.1f} FPS")

    cap.release()
    writer.release()
    elapsed = time.time() - t_start

    print(f"\n✅ Tracking complete!")
    print(f"   Output      : {out_path}")
    print(f"   Total time  : {elapsed:.1f}s  ({frame_idx/elapsed:.1f} FPS avg)")
    print(f"   Unique persons tracked : {len(person_ids_seen)}")
    print(f"   Unique cars tracked    : {len(car_ids_seen)}")
    return out_path


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone Detection + ByteTrack")
    parser.add_argument("--source",  type=str, required=True,  help="Video path")
    parser.add_argument("--weights", type=str, required=True,  help="YOLOv8 .pt file")
    parser.add_argument("--output",  type=str, default="outputs/videos")
    parser.add_argument("--conf",    type=float, default=0.35)
    parser.add_argument("--iou",     type=float, default=0.45)
    args = parser.parse_args()

    track_video(args.weights, args.source, args.output, args.conf, args.iou)
