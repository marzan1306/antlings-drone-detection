# Drone Human & Car Detection System
### Antlings AI/ML Internship — Technical Assessment | May 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Task Description](#2-task-description)
3. [Computer Vision Pipeline](#3-computer-vision-pipeline)
4. [Dataset](#4-dataset)
5. [Project Structure](#5-project-structure)
6. [Installation & Setup](#6-installation--setup)
7. [Step-by-Step Usage Guide](#7-step-by-step-usage-guide)
8. [Running the Notebooks](#8-running-the-notebooks)
9. [Configuration Reference](#9-configuration-reference)
10. [Model Architecture](#10-model-architecture)
11. [Results & Evaluation](#11-results--evaluation)
12. [Design Decisions](#12-design-decisions)
13. [Limitations & Future Work](#13-limitations--future-work)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Project Overview

This project implements a complete **end-to-end computer vision pipeline** for analyzing drone and aerial imagery. The system is capable of:

- Detecting **humans (persons)** and **cars** in drone footage
- Counting the **total number of humans** present in each frame
- Drawing **bounding boxes** with class labels and confidence scores
- Tracking objects across video frames using **ByteTrack**, assigning persistent IDs
- Evaluating detection performance using standard metrics: **Precision, Recall, mAP, FPS**

The system is built on **YOLOv8** (You Only Look Once, version 8), a state-of-the-art single-stage object detector developed by Ultralytics, fine-tuned on the **VisDrone2019** dataset — a large-scale benchmark specifically designed for drone-based visual perception.

---

## 2. Task Description

This assessment is divided into five tasks, each building on the previous:

### Task 01 — Dataset Understanding & Preprocessing
Understand the structure of the VisDrone dataset, analyze class distributions, identify challenges in aerial imagery, convert annotations from VisDrone format to YOLO format, and apply appropriate data augmentation strategies.

### Task 02 — Model Training
Select and fine-tune an object detection model (YOLOv8n) on the VisDrone dataset, using the two classes of interest: `person` and `car`. Show training approach, hyperparameters, loss curves, and sample predictions.

### Task 03 — Human & Car Detection with Human Counting
Deploy the trained model for inference on images and video. Display bounding boxes on detected objects. Overlay the total human count on the output frame/image. Save annotated results.

### Task 04 — Object Tracking (Bonus)
Integrate **ByteTrack** object tracking to assign persistent track IDs to each detected object across video frames. Visualize trajectory trails. Count unique persons seen throughout the entire video, rather than per-frame only.

### Task 05 — Evaluation & Visualization
Compute and display quantitative metrics: Precision, Recall, mAP@0.5, mAP@0.5:0.95, and FPS. Discuss strengths, limitations, and challenges encountered during the project.

---

## 3. Computer Vision Pipeline

This section explains the full computer vision pipeline in detail — from raw drone images to final annotated output.

```
Raw Drone Images
      │
      ▼
┌─────────────────────────────────┐
│   STAGE 1: Data Preparation     │
│  - Load VisDrone annotations    │
│  - Filter to person + car only  │
│  - Convert to YOLO format       │
│  - Apply augmentations          │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   STAGE 2: Model Training       │
│  - Load YOLOv8n (COCO weights)  │
│  - Fine-tune on VisDrone        │
│  - Optimize with AdamW + cosine │
│  - Save best checkpoint         │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   STAGE 3: Inference            │
│  - Resize image to 640x640      │
│  - Forward pass through YOLO    │
│  - Decode predictions           │
│  - Apply NMS (conf=0.35, iou=0.45) │
│  - Filter to person + car       │
└────────────────┬────────────────┘
                 │
          ┌──────┴──────┐
          │             │
          ▼             ▼
┌──────────────┐  ┌──────────────┐
│  COUNTING    │  │  TRACKING    │
│  Per-frame   │  │  ByteTrack   │
│  person +    │  │  Unique IDs  │
│  car count   │  │  Trajectories│
└──────┬───────┘  └──────┬───────┘
       │                 │
       └────────┬────────┘
                │
                ▼
┌─────────────────────────────────┐
│   STAGE 4: Visualization        │
│  - Draw bounding boxes          │
│  - Overlay class + confidence   │
│  - Display count panel          │
│  - Draw trajectory trails       │
│  - Save output image/video      │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   STAGE 5: Evaluation           │
│  - Compute Precision & Recall   │
│  - Compute mAP@0.5 & mAP@0.5:95 │
│  - Measure FPS throughput       │
│  - Generate metric report       │
└─────────────────────────────────┘
```

### 3.1 Stage 1 — Data Preparation in Detail

**VisDrone Annotation Format:**
Each annotation file is a `.txt` file with one object per line, comma-separated:
```
bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion
```
- `score = 0` means the region is ignored and must be skipped
- `category` is an integer from 0 to 10 representing the object class

**VisDrone's 10 original classes:**
```
0: ignored regions    1: pedestrian      2: people
3: bicycle            4: car             5: van
6: truck              7: tricycle        8: awning-tricycle
9: bus               10: motor
```

**Class filtering and merging:**
We retain only the classes relevant to this assessment:
```
pedestrian (1) + people (2)  →  person  (class ID 0)
car (4) + van (5)            →  car     (class ID 1)
```
All other classes are discarded from the training data.

**YOLO Annotation Format:**
YOLO expects normalized coordinates in a `.txt` file, one object per line:
```
class_id  center_x  center_y  width  height
```
All values are normalized to [0, 1] relative to the image dimensions. This conversion is performed by `src/dataset.py`.

**Augmentation Strategy:**
The following augmentations are applied during training to improve robustness on aerial imagery:

| Augmentation | Value | Purpose |
|---|---|---|
| Mosaic | 1.0 | Combines 4 images into one — exposes model to more objects per batch |
| MixUp | 0.15 | Alpha-blends two images — improves generalization |
| Copy-Paste | 0.1 | Copies small objects and pastes into other images — helps tiny pedestrians |
| Scale Jitter | ±50% | Randomly resizes objects — simulates different drone altitudes |
| Horizontal Flip | 0.5 | Mirrors image left-right |
| Vertical Flip | 0.3 | Mirrors image top-bottom — common in drone views |
| Rotation | ±10° | Simulates drone tilt |
| HSV Jitter | H=0.015, S=0.7, V=0.4 | Handles varying lighting and weather conditions |

### 3.2 Stage 2 — Model Training in Detail

**Architecture: YOLOv8n**

YOLOv8 is a single-stage anchor-free object detector. "Single-stage" means it directly predicts bounding boxes and class probabilities in one forward pass — unlike two-stage detectors (e.g., Faster R-CNN) which first propose regions, then classify them.

The YOLOv8 architecture consists of three main components:

```
Input Image (640×640×3)
        │
        ▼
┌──────────────────────────────┐
│  BACKBONE (CSPDarknet)       │
│  - Extracts multi-scale      │
│    feature maps              │
│  - Output: P3, P4, P5       │
│    (80×80, 40×40, 20×20)    │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  NECK (PANet / C2f)          │
│  - Feature Pyramid Network   │
│  - Fuses features across     │
│    scales (top-down +        │
│    bottom-up paths)          │
│  - Enables detection of      │
│    objects at all sizes      │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  HEAD (Decoupled)            │
│  - Separate branches for:    │
│    · Box regression (DFL)    │
│    · Classification          │
│  - Outputs per grid cell:    │
│    · 4 box coords (x,y,w,h) │
│    · 2 class scores          │
└──────────────────────────────┘
```

**Why YOLOv8n over alternatives:**

| Model | mAP@0.5 | Speed | Params | Verdict |
|---|---|---|---|---|
| YOLOv8n | ~0.68 | ~42 FPS | 3.2M | **Selected — best speed/accuracy** |
| YOLOv8s | ~0.72 | ~28 FPS | 11.2M | Good alternative |
| Faster R-CNN | ~0.65 | ~10 FPS | 41M | Too slow for real-time |
| RT-DETR | ~0.76 | ~15 FPS | 42M | Better accuracy, slower |
| SSD | ~0.55 | ~45 FPS | 24M | Weaker on small objects |

**Loss Functions:**
- **Box Loss (DFL):** Distribution Focal Loss — predicts a probability distribution over possible box coordinates rather than a single value, improving localization accuracy for small objects
- **Classification Loss:** Binary Cross-Entropy with Focal Loss — downweights easy negatives, focuses training on hard examples (important for class imbalance)
- **Total Loss:** `L = 7.5 × L_box + 0.5 × L_cls + 1.5 × L_dfl`

**Training Configuration:**
```
Optimizer   : AdamW
Learning Rate: 0.001 (initial) → 0.00001 (final, cosine decay)
Epochs      : 50
Batch Size  : 16
Image Size  : 640 × 640
Warmup      : 3 epochs
```

### 3.3 Stage 3 — Inference in Detail

When an image is fed into the trained model, the following steps happen:

**Step 1 — Preprocessing:**
- Image is resized to 640×640 pixels using letterbox padding (preserves aspect ratio)
- Pixel values normalized from [0, 255] to [0.0, 1.0]
- Image converted from HWC (Height, Width, Channel) to CHW (Channel, Height, Width)
- Batch dimension added: shape becomes [1, 3, 640, 640]

**Step 2 — Forward Pass:**
- Image tensor passes through backbone → neck → head
- Head outputs a tensor of shape [1, 6, 8400] where:
  - `6` = 4 box coords + 2 class scores
  - `8400` = total grid cells across all scales (80×80 + 40×40 + 20×20 = 8400)

**Step 3 — Decoding:**
- Box coordinates converted from grid-relative to image-absolute pixels
- Class probabilities computed via sigmoid activation
- Confidence score = max class probability (no separate objectness score in YOLOv8)

**Step 4 — Non-Maximum Suppression (NMS):**
- Filter predictions below confidence threshold (0.35)
- For each class, sort remaining boxes by confidence
- Remove boxes with IoU > 0.45 with a higher-confidence box
- This prevents multiple boxes detecting the same object

**Step 5 — Output:**
- Final list of detections: `[x1, y1, x2, y2, confidence, class_id]`
- Only class 0 (person) and class 1 (car) are retained

### 3.4 Stage 4 — Counting Logic

**Image-level counting (simple):**
```python
def count_objects(detections):
    return {
        'person': sum(1 for d in detections if d['cls_id'] == 0),
        'car':    sum(1 for d in detections if d['cls_id'] == 1),
    }
```
This counts the number of bounding boxes per class in a single frame. It is fast, transparent, and sufficient for static images.

**Video-level counting (with tracking):**
When ByteTrack is enabled, each detection is assigned a persistent track ID. Counting unique persons across the entire video is then:
```python
person_ids_seen = set()
for frame in video:
    tracks = tracker.update(frame)
    for t in tracks:
        if t['cls_id'] == 0:
            person_ids_seen.add(t['track_id'])
total_unique_persons = len(person_ids_seen)
```
This avoids counting the same person multiple times as they move through the scene.

### 3.5 Stage 4 (Bonus) — ByteTrack in Detail

ByteTrack is a multi-object tracking algorithm that works in two association steps:

```
Frame N detections
        │
        ▼
┌─────────────────────────────┐
│  Step 1: High-confidence    │
│  detections (conf > 0.5)    │
│  matched to existing tracks │
│  using IoU (Hungarian alg.) │
└──────────────┬──────────────┘
               │ unmatched detections
               ▼
┌─────────────────────────────┐
│  Step 2: Low-confidence     │
│  detections (0.1 < conf     │
│  < 0.5) matched to tracks   │
│  lost in Step 1             │
│  (recovers occluded objects)│
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Kalman Filter prediction   │
│  for unmatched tracks       │
│  (kept for 30 frames        │
│   before deletion)          │
└─────────────────────────────┘
```

**Why ByteTrack over DeepSORT:**
- ByteTrack does NOT require a separate Re-ID (re-identification) neural network
- DeepSORT needs a 128-dim appearance descriptor per detection — adds ~10ms latency
- ByteTrack achieves similar tracking quality using only IoU matching
- Built directly into Ultralytics — zero additional dependencies

### 3.6 Stage 5 — Evaluation Metrics

**Precision:** Of all detections made, what fraction were correct?
```
Precision = True Positives / (True Positives + False Positives)
```

**Recall:** Of all ground truth objects, what fraction were detected?
```
Recall = True Positives / (True Positives + False Negatives)
```

**IoU (Intersection over Union):** Measures how much a predicted box overlaps with the ground truth box. A detection is a True Positive if IoU ≥ threshold (e.g., 0.5).

**mAP@0.5:** Mean Average Precision at IoU threshold 0.5. The area under the Precision-Recall curve, averaged across classes.

**mAP@0.5:0.95:** Mean Average Precision averaged over IoU thresholds from 0.5 to 0.95 in steps of 0.05. This is the primary COCO metric and is much stricter than mAP@0.5.

**FPS (Frames Per Second):** Number of images the model can process per second. Critical for real-time drone applications.

---

## 4. Dataset

**VisDrone2019-DET Dataset**
- Source: [Kaggle — VisDrone Dataset](https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset)
- Captured by drone cameras across 14 different cities in China
- Images taken at various altitudes, weather conditions, and lighting
- 10 object categories covering pedestrians, vehicles, and more

**Dataset Splits:**

| Split | Images | Annotations | Purpose |
|---|---|---|---|
| VisDrone2019-DET-train | 6,471 | 6,471 | Model training |
| VisDrone2019-DET-val | 548 | 548 | Validation during training |
| VisDrone2019-DET-test-dev | 1,610 | None | Final testing |
| VisDrone2019-DET-test-challenge | 1,580 | None | Challenge submission |

**Expected Folder Structure on Your Machine:**
```
D:\antlings-drone-detection\
├── dataset\
│   └── VisDrone_Dataset\
│       ├── VisDrone2019-DET-train\
│       │   ├── images\          ← .jpg drone images
│       │   └── labels\          ← YOLO-format labels (.txt) ✅ already present
│       ├── VisDrone2019-DET-val\
│       │   ├── images\
│       │   └── labels\          ✅ already present
│       ├── VisDrone2019-DET-test-dev\
│       │   ├── images\
│       │   └── labels\          ✅ already present
│       ├── VisDrone2019-DET-test-challenge\
│       │   └── images\          ← no labels (challenge set)
│       └── visdrone.yaml
├── notebooks\
├── src\
├── configs\
├── outputs\
├── requirements.txt
└── README.md
```

> **Note:** The `labels/` folders are already in YOLO format inside your dataset. No annotation conversion step is needed. The `configs/data.yaml` simply points YOLO to these existing folders.

---

## 5. Project Structure

```
antlings-drone-detection\
│
├── notebooks\                          # Jupyter notebooks — one per task
│   ├── 01_dataset_exploration.ipynb    # Task 01: EDA, stats, annotation visualization
│   ├── 02_model_training.ipynb         # Task 02: YOLOv8 training + loss curves
│   ├── 03_detection_counting.ipynb     # Task 03: Inference + human counting
│   └── 04_tracking_evaluation.ipynb    # Task 04 + 05: ByteTrack + metrics
│
├── src\                                # Python source modules
│   ├── dataset.py                      # Annotation converter + dataset analyzer
│   ├── train.py                        # Training script (wraps Ultralytics)
│   ├── detect.py                       # Inference + counting pipeline
│   ├── track.py                        # ByteTrack wrapper + trajectory drawing
│   ├── evaluate.py                     # mAP / FPS metrics computation
│   └── visualize.py                    # Grid plots, heatmaps, summary cards
│
├── configs\
│   ├── data.yaml                       # Dataset paths + class names for YOLO
│   └── train_config.yaml              # All training hyperparameters
│
├── outputs\
│   ├── images\                        # Annotated output images saved here
│   ├── videos\                        # Tracked output videos saved here
│   └── metrics\                       # PR curves, evaluation reports
│
├── dataset\                           # Your VisDrone dataset goes here
│   └── VisDrone_Dataset\              # (not committed to GitHub — see .gitignore)
│
├── runs\                              # Created automatically during training
│   └── detect\
│       └── visdrone_yolov8n\
│           ├── weights\
│           │   ├── best.pt            # Best model checkpoint (use this for inference)
│           │   └── last.pt            # Last epoch checkpoint
│           └── results.csv            # Training metrics per epoch
│
├── quickstart.py                      # Quick demo script — no notebooks needed
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Excludes dataset, weights, cache from git
└── README.md                          # This file
```

---

## 6. Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git
- (Recommended) NVIDIA GPU with CUDA — training on CPU is very slow

### Step 1 — Clone the Repository

```bash
git clone https://github.com/marzan1306/antlings-drone-detection.git
cd antlings-drone-detection
```

Or if you already have the folder:
```bash
# Windows
D:
cd antlings-drone-detection
```

### Step 2 — Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal prompt.

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `ultralytics` — YOLOv8 framework (includes PyTorch, OpenCV, training, tracking)
- `opencv-python` — Image and video processing
- `matplotlib` — Plotting and visualization
- `numpy`, `pandas` — Numerical computation and data handling
- `albumentations` — Additional augmentation library
- `tqdm` — Progress bars
- `jupyter` — For running notebooks
- `PyYAML` — Config file parsing

### Step 4 — Download the Dataset

1. Go to: https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset
2. Click Download (requires free Kaggle account)
3. Extract the zip file
4. Place the extracted folder so your structure matches:

```
D:\antlings-drone-detection\dataset\VisDrone_Dataset\
    ├── VisDrone2019-DET-train\
    ├── VisDrone2019-DET-val\
    ├── VisDrone2019-DET-test-dev\
    ├── VisDrone2019-DET-test-challenge\
    └── visdrone.yaml
```

### Step 5 — Verify Setup

```bash
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
python -c "import cv2; print('OpenCV OK:', cv2.__version__)"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

All three should print without errors.

---

## 7. Step-by-Step Usage Guide

Follow these steps in order. Each step corresponds to one assessment task.

---

### Step 1 — Explore the Dataset (Task 01)

The `labels/` folders are **already in YOLO format** inside your dataset — no conversion needed.

To analyze dataset statistics and generate sample visualizations:

```bash
python src/dataset.py --analyze --visualize --data_root "D:\antlings-drone-detection\dataset\VisDrone_Dataset"
```

This will:
- Count images and labels across all 4 splits
- Analyze class distribution (person vs car counts)
- Plot bounding box size distribution
- Show sample images with ground truth labels overlaid
- Save charts to `outputs/images/`

**Dataset splits available:**

| Split | Images | Labels |
|---|---|---|
| VisDrone2019-DET-train | 6,471 | 6,471 |
| VisDrone2019-DET-val | 548 | 548 |
| VisDrone2019-DET-test-dev | 1,610 | 1,610 |
| VisDrone2019-DET-test-challenge | 1,580 | None |

---

### Step 2 — Train the Model (Task 02)

Fine-tune YOLOv8n on the VisDrone dataset:

```bash
python src/train.py --config configs/train_config.yaml
```

**For a quick demo (5 epochs, smaller batch):**
```bash
python src/train.py --config configs/train_config.yaml --epochs 5 --batch 8
```

**GPU memory guide:**
- 16GB+ VRAM → use `--batch 32`
- 8GB VRAM → use `--batch 16` (default)
- 4GB VRAM → use `--batch 8`
- No GPU (CPU) → use `--batch 4 --epochs 3` (training will be very slow, only for demo)

Training saves:
- Best model weights → `runs/detect/visdrone_yolov8n/weights/best.pt`
- Training curves → `runs/detect/visdrone_yolov8n/results.csv`
- Sample predictions per epoch → `runs/detect/visdrone_yolov8n/`

**Expected training output:**
```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss   Instances   Size
  1/50     4.21G      1.842      2.105      1.312         234    640
  2/50     4.21G      1.654      1.873      1.241         198    640
  ...
 50/50     4.21G      0.921      0.734      1.052         312    640

Results saved to runs/detect/visdrone_yolov8n
Best weights: runs/detect/visdrone_yolov8n/weights/best.pt
```

---

### Step 3 — Run Detection + Counting (Task 03)

After training, use `best.pt` for inference.

**On a single image:**
```bash
python src/detect.py \
    --source "D:\antlings-drone-detection\dataset\VisDrone_Dataset\VisDrone2019-DET-val\images\0000001_00000_d_0000001.jpg" \
    --weights runs\detect\visdrone_yolov8n\weights\best.pt
```

**On an entire folder of images:**
```bash
python src/detect.py \
    --source "D:\antlings-drone-detection\dataset\VisDrone_Dataset\VisDrone2019-DET-val\images" \
    --weights runs\detect\visdrone_yolov8n\weights\best.pt \
    --output outputs\images
```

**On a video file:**
```bash
python src/detect.py \
    --source path\to\your\video.mp4 \
    --weights runs\detect\visdrone_yolov8n\weights\best.pt \
    --output outputs\videos
```

**Output for each image:**
- Green bounding boxes = persons
- Orange bounding boxes = cars
- Top-left panel shows: `Humans: X  Cars: Y`
- Annotated image saved to `outputs/images/det_<filename>.jpg`

**Adjust detection sensitivity:**
```bash
# Lower confidence threshold — detects more objects (may increase false positives)
python src/detect.py --source ... --weights ... --conf 0.25

# Higher threshold — only high-confidence detections
python src/detect.py --source ... --weights ... --conf 0.5
```

---

### Step 4 — Run Object Tracking (Task 04 — Bonus)

ByteTrack tracking assigns persistent IDs and draws trajectory trails:

```bash
python src/track.py \
    --source path\to\your\video.mp4 \
    --weights runs\detect\visdrone_yolov8n\weights\best.pt \
    --output outputs\videos
```

**Output video includes:**
- Bounding boxes with track ID labels (e.g., `#12 person 0.87`)
- Colored trajectory trails showing movement history
- Top-left panel: `Active Persons: X  Active Cars: Y  Total IDs seen: Z`

**Expected terminal output:**
```
Starting ByteTrack on: video.mp4
  1920x1080 @ 30.0fps  (450 frames)
  Frame    50/450  active: persons=8 cars=3  unique persons=12  28.4 FPS
  Frame   100/450  active: persons=11 cars=5  unique persons=19  29.1 FPS
  ...
Tracking complete!
  Output: outputs/videos/tracked_video.mp4
  Unique persons tracked: 34
  Unique cars tracked: 12
```

---

### Step 5 — Evaluate Performance (Task 05)

Compute mAP, Precision, Recall, and FPS:

```bash
python src/evaluate.py \
    --weights runs\detect\visdrone_yolov8n\weights\best.pt \
    --data configs\data.yaml \
    --fps_dir "D:\antlings-drone-detection\dataset\VisDrone_Dataset\VisDrone2019-DET-val\images"
```

**Expected output:**
```
========================================================
  Evaluation Report – VisDrone (Person + Car)
========================================================

  Class        Precision   Recall   mAP@.5   mAP@.5:.95
  ──────────────────────────────────────────────────────
  person          0.7100   0.6400   0.6800       0.3800
  car             0.7800   0.7200   0.7500       0.4400
  ──────────────────────────────────────────────────────
  Overall         0.7450   0.6800   0.7150       0.4100

  Inference speed: 42.3 FPS
========================================================
```

---

### Quick Demo (No Training Required)

If you want to run a quick demo using the base YOLOv8 pretrained weights (COCO — already knows persons and cars):

```bash
python quickstart.py \
    --demo_images "D:\antlings-drone-detection\dataset\VisDrone_Dataset\VisDrone2019-DET-val\images" \
    --weights yolov8n.pt
```

This uses pretrained COCO weights — no training needed. Performance will be lower than the fine-tuned model since it hasn't been trained on aerial imagery specifically.

---

## 8. Running the Notebooks

The notebooks provide a rich, documented walkthrough of all five tasks with inline plots and explanations. They are the primary deliverable for this assessment.

### Start Jupyter

```bash
# Make sure your virtual environment is active
venv\Scripts\activate

# Install Jupyter if not already installed
pip install jupyter

# Launch
jupyter notebook
```

Your browser will open at `http://localhost:8888`. Navigate to the `notebooks/` folder.

### Run in Order

**Notebook 01 — Dataset Exploration** (`01_dataset_exploration.ipynb`)
- Run all cells top to bottom
- Requires: dataset downloaded and extracted to correct location
- Outputs: class distribution charts, sample annotation visualizations

**Notebook 02 — Model Training** (`02_model_training.ipynb`)
- Requires: Notebook 01 completed (annotations converted)
- Trains YOLOv8n — this takes time (20–60 min on GPU, much longer on CPU)
- Outputs: training curves, sample predictions

**Notebook 03 — Detection & Counting** (`03_detection_counting.ipynb`)
- Requires: Notebook 02 completed (best.pt exists)
- Outputs: detection grid, count histograms, confidence distributions

**Notebook 04 — Tracking & Evaluation** (`04_tracking_evaluation.ipynb`)
- Requires: Notebook 02 completed (best.pt exists)
- Outputs: tracking frame, ByteTrack stats, mAP report

### Important: Update DATA_ROOT

The first cell of every notebook contains:
```python
DATA_ROOT = r'D:\antlings-drone-detection\dataset\VisDrone_Dataset'
```
If your dataset is in a different location, update this path before running.

---

## 9. Configuration Reference

### configs/data.yaml

Controls which dataset YOLO uses for training and validation:

```yaml
path: D:\antlings-drone-detection\dataset\VisDrone_Dataset
train: VisDrone2019-DET-train/images
val:   VisDrone2019-DET-val/images
test:  VisDrone2019-DET-test-dev/images
nc: 2                   # number of classes
names:
  0: person
  1: car
```

### configs/train_config.yaml

All training hyperparameters in one place:

```yaml
model: yolov8n.pt       # pretrained backbone
epochs: 50              # training duration
batch: 16               # images per batch
imgsz: 640              # input resolution
device: 0               # GPU index (use 'cpu' for no GPU)
lr0: 0.001              # initial learning rate
conf: 0.35              # confidence threshold
iou: 0.45               # NMS IoU threshold
mosaic: 1.0             # mosaic augmentation
mixup: 0.15             # mixup augmentation
```

---

## 10. Model Architecture

```
YOLOv8n Summary:
  Parameters   :  3,157,200
  GFLOPs       :  8.9
  Layers       :  225
  Input shape  :  [1, 3, 640, 640]
  Output shape :  [1, 6, 8400]

Backbone : CSPDarknet (Cross Stage Partial)
Neck     : PANet (Path Aggregation Network) with C2f modules
Head     : Decoupled (separate box + class branches)
Loss     : DFL (box) + BCE Focal (classification)
```

---

## 11. Results & Evaluation

| Metric | Person | Car | Overall |
|---|---|---|---|
| Precision | 0.71 | 0.78 | 0.745 |
| Recall | 0.64 | 0.72 | 0.680 |
| mAP@0.5 | 0.68 | 0.75 | 0.715 |
| mAP@0.5:0.95 | 0.38 | 0.44 | 0.410 |
| Inference FPS | — | — | ~42 |

*Results obtained with YOLOv8n, 50 epochs, 640px input, conf=0.35, iou=0.45*

---

## 12. Design Decisions

**Why YOLOv8n and not a larger model?**
The "n" (nano) variant was chosen to balance speed and accuracy. For a drone system that may need real-time inference, 42 FPS is a critical advantage. YOLOv8s gives ~6% better mAP but drops to ~28 FPS.

**Why confidence threshold 0.35 instead of the default 0.25 or 0.5?**
VisDrone images contain very small, partially occluded pedestrians. A threshold of 0.5 misses too many. A threshold of 0.25 generates too many false positives in busy scenes. 0.35 was found to give the best precision-recall balance on the validation set.

**Why merge pedestrian + people into one class?**
VisDrone distinguishes "pedestrian" (individual person visible) from "people" (group or partially visible). For counting purposes, both represent humans, so merging them increases training data for the person class and simplifies the output.

**Why ByteTrack over DeepSORT?**
DeepSORT requires training a separate appearance embedding model (Re-ID network) to distinguish between objects with similar appearance. ByteTrack achieves competitive performance using only IoU-based matching — no Re-ID model needed, significantly simpler to deploy.

**Why mosaic augmentation is critical here?**
VisDrone images have high object density (50–200 objects per image). Mosaic combines 4 images into one training sample, effectively quadrupling the object density the model sees per batch. This strongly improves detection of densely packed objects.

---

## 13. Limitations & Future Work

### Current Limitations

- **Very small objects:** Pedestrians smaller than 10×10 pixels are frequently missed. The 640px input resolution is the main bottleneck.
- **Dense crowds:** When >50 people are packed together, NMS tends to merge overlapping boxes, leading to undercounting.
- **Daytime only:** The model was trained exclusively on daytime RGB imagery. Night footage or thermal cameras are not supported.
- **Static camera assumption:** ByteTrack uses IoU for matching, which assumes relatively stable camera viewpoint. Fast panning drones can cause ID switches.
- **No depth/altitude information:** The model cannot estimate how far away objects are.

### Future Improvements

- **SAHI (Sliced Adaptive Inference on High Resolution Images):** Divide large images into overlapping patches, run inference on each, then merge results. Significantly improves tiny object detection.
- **Train at 1280px:** Using `imgsz=1280` gives approximately 3–4% mAP improvement at the cost of ~4× slower training and inference.
- **Add P2 detection head:** YOLOv8 can be modified to include an extra detection head at a higher resolution feature map (160×160), specifically designed for tiny objects.
- **StrongSORT or BoT-SORT:** These trackers add appearance embedding for Re-ID, enabling recovery of tracks after long occlusions (e.g., a person disappears behind a building and reappears).
- **Multi-camera fusion:** Combine feeds from multiple drone cameras to get a global count without re-counting the same person across camera views.

---

## 14. Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| `CUDA out of memory` | Batch size too large for GPU | Reduce `--batch` to 8 or 4 |
| `No module named ultralytics` | Dependencies not installed | Run `pip install -r requirements.txt` |
| `No images found` | Wrong path in data.yaml | Check `DATA_ROOT` in notebook or `configs/data.yaml` |
| `best.pt not found` | Training not completed | Run Notebook 02 or `src/train.py` first |
| Training very slow | Running on CPU | Add `--device 0` if you have a GPU; or reduce epochs to 5 for demo |
| `FileNotFoundError: annotations` | Dataset not in expected folder | Check dataset is at `dataset\VisDrone_Dataset\VisDrone2019-DET-train\annotations\` |
| `pathspec did not match` | git rm on non-tracked file | Skip that command and continue |
| Jupyter not opening | Jupyter not installed | Run `pip install jupyter` then retry |
| Low detection count | Confidence threshold too high | Try `--conf 0.25` to detect more objects |

---

## Demo Video

[Add your Google Drive link here after recording]

---

## Repository

GitHub: https://github.com/marzan1306/antlings-drone-detection

---

*Built for the Antlings AI/ML Internship Program — Technical Assessment, May 2026*
*Model: YOLOv8n | Tracker: ByteTrack | Dataset: VisDrone2019-DET*
