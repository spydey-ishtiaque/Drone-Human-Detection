# Drone Human & Car Detection System
End-to-end Computer Vision Pipeline for a "human and car detection system" in drone imagery, including counting, visualization, evaluation, and optional tracking.
Used Model: YOLOv8.
Dataset: VisDrone2019-DET.

---

## Overview

The goal of the project was to build a a complete computer vision pipeline that processes drone images to detect humans and cars, count the total number of humans or cars visible, and visualize the results with annotated bounding boxes. An object tracking with persistent identity assignment across video frames is also implemented.

The pipeline runs entirely in a Kaggle notebook environment using a free GPU (GPU T4 x2), making it "reproducible" without any local hardware requirements!

**AI tools used during development:** Claude (Anthropic) and Codex were used throughout this project for code generation, debugging, and architectural guidance, as explicitly permitted by the assessment guidelines.

---

## Demo Video

>(https://drive.google.com/drive/folders/1SXTiVAibBvJsRvcNcytdS3zlhQBASyqS?usp=drive_link)

---

## Repository Structure

```
drone-human-detection/
├── notebooks/
│   └── visdrone_yolo_detection.ipynb   ← Full Kaggle notebook
├── outputs/
│   ├── Train_Set_Samples.png
│   ├── Validation_Set_Samples.png
│   ├── detection_results.png
│   ├── sahi_comparison.png
│   ├── tracking_preview.png            ← Insert after Cell 11
│   └── detection_log.csv
├── scripts/
│   └── detect.py                       ← Standalone inference script
└── README.md
```

---

## Task 1 — Dataset Understanding & Preprocessing

### Dataset Structure

The dataset used is **VisDrone2019-DET**, a benchmark drone detection dataset collected by the AISKYEYE team. It contains aerial images captured by drone cameras across 14 different cities in China, covering a wide variety of scenes including urban roads, pedestrian areas, and highways.

The dataset is divided into four splits:

| Split          | Images | Labels                              |
|---             |---     |---                                  |
| Train          | 6,471  | ✅                                  |
| Validation     | 548    | ✅                                  |
| Test-Dev       | 1,610  | ✅                                  |
| Test-Challenge | ~1,580 | ❌ (no labels, for submission only) |

Labels are provided in **YOLO format** — each `.txt` file contains one line per object in the image, structured as:

```
class_id  x_center  y_center  width  height
```

All five values are normalized between 0 and 1 relative to the image dimensions. This means no label conversion was necessary — the dataset was ready for YOLOv8 training without any preprocessing of the annotation files.

The dataset defines **10 object classes**:

|ID |     Class       | Detected as |
|---|      ---        |     ---     |
| 0 | pedestrian      | Human ✅   |
| 1 | people          | Human ✅   |
| 2 | bicycle         | —           |
| 3 | car             | Car ✅     |
| 4 | van             | Car ✅     |
| 5 | truck           | —           |
| 6 | tricycle        | —           |
| 7 | awning-tricycle | —           |
| 8 | bus             | —           |
| 9 | motor           | —           |

For this task, classes 0 and 1 (people  & pedestrian) are counted as **humans** and classes 3 and 4 as **cars**. The distinction between `pedestrian` (isolated individual) and `people` (dense cluster) is maintained during training to help the model learn richer visual features, but both are aggregated into a single human count at inference time. Similarly, `van` is grouped with `car` because the visual distinction between the two is minimal at drone altitude.

### Sample Visualizations

> 📷 **Insert `Train_Set_Samples.png` here**
> *(Download from Kaggle output and drag into the GitHub file editor, or reference as `outputs/Train_Set_Samples.png`)*

> 📷 **Insert `Validation_Set_Samples.png` here**

### Challenges Noticed in the Dataset

**Tiny object size.** The most significant challenge in this dataset is that humans often occupy fewer than 15×15 pixels in a 1920×1080 image. Standard YOLO inference at 640px resolution further compresses these objects, making them extremely difficult to detect reliably. This was directly addressed by implementing SAHI sliced inference at the detection stage.

**Class imbalance.** Cars and pedestrians appear far more frequently than classes like `awning-tricycle` or `tricycle`. This can cause the model to underperform on rare classes, but since we only care about humans and cars for this task, the imbalance is not a practical problem here.

**Variable altitude and angle.** Images are taken from different drone heights and angles, so the same object can appear very differently across images. Augmentation was used during training to expose the model to these variations.

**Dense crowds.** In busy scenes, bounding boxes overlap significantly. The `people` class (class 1) handles some of this by labeling dense clusters as a single annotation, but crowded scenes still challenge precise counting.

---

## Task 2 — Model Training

### Approach

Rather than training an object detection model from scratch — which would require significantly more data and compute time — a **YOLOv8s (small)** model pre-trained on the COCO dataset was fine-tuned on the VisDrone training set. This transfer learning approach means the model already understands basic visual features like edges, shapes, and textures from COCO, and the fine-tuning process adapts it specifically to aerial drone imagery.

YOLOv8 (You Only Look Once, version 8) is a single-stage detector — it predicts bounding boxes and class labels in a single forward pass through the network, making it fast enough for practical use while maintaining competitive accuracy.

### Training Configuration

| Parameter  | Value                        | Reason |
|---         |---                           |---|
| Base model | yolov8s.pt                   | Balance of speed and accuracy |
| Epochs     | 50 (30 initial + 20 resumed) | Allow sufficient convergence |
| Image size | 640px                        | Standard YOLO input resolution |
| Batch size | 16                           | Fits T4 GPU memory |
| Optimizer  | Auto (AdamW)                 | Chosen automatically by ultralytics |
| Platform   | Kaggle (NVIDIA T4 GPU)       | Free cloud GPU, ~40 min/run |

### Augmentation

The following augmentations were applied during training to improve generalization:

- **Mosaic augmentation** — combines 4 training images into a single sample. This is particularly effective for VisDrone because it exposes the model to many objects in a single training step.
- **HSV color jitter** — random shifts in hue, saturation, and brightness to simulate different lighting conditions.
- **Horizontal and vertical flips** — reasonable for aerial imagery where there is no fixed "up" orientation from the object's perspective.
- **Random scaling** — handled internally by the mosaic augmentation.

### Training Curves

> 📷 **Insert `results.png` from `/kaggle/working/runs/visdrone_resumed/results.png` here**
> *(Shows loss curves and mAP progression across epochs)*

### Sample Training Predictions

> 📷 **Insert a sample prediction image from the training run here**
> *(Found at `/kaggle/working/runs/visdrone_resumed/val_batch0_pred.jpg`)*

---

## Task 3 — Human & Car Detection with Human Counting

### Detection Pipeline

At inference time, the trained model processes each image and outputs bounding boxes with class IDs and confidence scores. A post-processing step filters the results to keep only human and car detections, draws annotated boxes, and displays a count banner at the top of each image.

```
Input Image
    ↓
YOLOv8s Inference (all 10 classes)
    ↓
Filter: keep classes {0, 1} as Human, {3, 4} as Car
    ↓
Draw bounding boxes + labels
    ↓
Add count banner (HUMANS: N | CARS: N)
    ↓
Output Annotated Image
```

Humans are shown in **red** boxes and cars in **blue** boxes. The confidence score is displayed alongside each label.

### Results on Test-Dev Set (1,610 images)

| Metric | Value |
|---|---|
| Total humans detected | 12,338 |
| Total cars detected | 25,699 |
| Average humans per image | 7.7 |
| Average cars per image | 16.0 |
| Processing time | ~40 seconds |

### Detection Visualization

> 📷 **Insert `detection_results.png` here**

### SAHI Sliced Inference

Because VisDrone objects are extremely small, standard inference misses many detections. **SAHI (Sliced Aided Hyper Inference)** addresses this by dividing each image into overlapping 640×640 tiles, running inference on each tile at full resolution, and merging the results back into the original image coordinates. This recovers a significant number of small human detections that standard inference would miss.

> 📷 **Insert `sahi_comparison.png` here**
> *(Left: standard inference. Right: SAHI. The difference in small-object detections is visible in crowded scenes.)*

---

## Task 4 — Object Tracking (Bonus)

### Why Tracking Matters for Counting

Simple per-frame detection counts objects in each frame independently. In a video, a person who appears in 100 consecutive frames would be counted 100 times. **Object tracking** solves this by assigning a persistent identity (a track ID) to each detected object and maintaining that ID as the object moves across frames. This allows accurate counting of unique individuals rather than per-frame detections.

### Implementation

Tracking was implemented using **ByteTrack**, which is built directly into the Ultralytics YOLOv8 library (`model.track(..., tracker='bytetrack.yaml')`). ByteTrack works by predicting where each tracked object should appear in the next frame using a motion model, then matching those predictions against new detections. Objects that temporarily disappear (due to occlusion or low confidence) are kept in a buffer and can be re-identified when they reappear.

Since the VisDrone DET dataset consists of individual images rather than video files, consecutive frames from the same sequence were reconstructed into a video. VisDrone image filenames encode the sequence ID and frame number (e.g. `0000001_00001_d_0000001.jpg`), making this reconstruction straightforward.

### Tracking Output

Each detection box in the tracking video shows:
- Object type (HUMAN or CAR)
- Persistent track ID (e.g. `HUMAN #12`)
- Class name and confidence score

The banner at the top of each frame displays the current-frame count of humans and cars visible at that moment.

> 📹 **Insert a screenshot from `tracked_detection_output.mp4` here as `outputs/tracking_preview.png`**

---

## Task 5 — Evaluation & Visualization

### Quantitative Results

| Metric | Value |
|---|---|
| mAP@50 | 0.3659 |
| mAP@50-95 | 0.2116 |
| Precision | 0.4992 |
| Recall | 0.3795 |

**mAP@50** (mean Average Precision at 50% IoU overlap threshold) is the primary metric for object detection. A score of 0.366 on VisDrone is a reasonable result for a small model trained with limited compute. State-of-the-art specialized models on this benchmark typically achieve 0.40–0.50 mAP@50 using significantly more training time and larger architectures.

**Precision** of 0.499 means roughly half of the boxes the model draws actually correspond to real objects. **Recall** of 0.380 means the model finds about 38% of all objects in the scene — the remaining 62% are missed, primarily small or partially occluded objects.

### Strengths

- The pipeline is fully end-to-end: from raw dataset images to annotated detections, counting, and tracked video output.
- SAHI sliced inference meaningfully improves small-object recall without any retraining.
- The system processes 1,610 test images in approximately 40 seconds on a T4 GPU, demonstrating practical inference speed.
- ByteTrack tracking provides more meaningful human counts in video scenarios than naive per-frame summation.

### Limitations

- The model misses a significant number of small, distant, or occluded humans — a fundamental challenge in low-altitude drone footage where people can be under 10 pixels tall.
- Precision of ~50% means there is a notable false positive rate, particularly in cluttered scenes where shadows or bicycles may be classified as humans.
- The model was trained on a relatively small number of epochs due to time constraints. Longer training, a larger model (YOLOv8m or YOLOv8l), or a higher input resolution (1280px) would likely improve mAP.
- Counting in dense crowd scenes (where the `people` class is used as a cluster label) is inherently approximate — a single `people` box may represent 5–20 individuals.

### Challenges Faced

The most significant practical challenge was the GPU compatibility issue encountered on Kaggle, where the installed PyTorch version (2.10.0+cu128) required CUDA capability ≥ sm_70 but the assigned P100 GPU (sm_60) did not meet this requirement. Switching the accelerator to a T4 GPU resolved this immediately.

A recurring challenge was managing file paths on Kaggle, where the dataset was mounted at a non-standard path (`/kaggle/input/datasets/banuprasadb/...`). This required careful path verification before running each cell.

A subtle variable-naming conflict between the detection cell and the SAHI cell (both defining `HUMAN_CLASSES` and `CAR_CLASSES` but using different types — integer sets vs string sets) caused zero-detection output that required systematic debugging to isolate.

---

## How to Run

### Requirements

```bash
pip install ultralytics sahi
```

### Inference on a Single Image

```python
from ultralytics import YOLO
import cv2

model = YOLO('outputs/best.pt')
results = model('your_image.jpg', conf=0.25)[0]

human_count = sum(1 for cls in results.boxes.cls if int(cls) in {0, 1})
car_count   = sum(1 for cls in results.boxes.cls if int(cls) in {3, 4})
print(f"Humans: {human_count} | Cars: {car_count}")
```

### Full Notebook

Open `notebooks/visdrone_yolo_detection.ipynb` in Kaggle and attach the [VisDrone dataset](https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset). Run all cells in order. GPU accelerator (T4) must be enabled.

---

## Acknowledgements

- Dataset: [VisDrone2019](https://github.com/VisDrone/VisDrone-Dataset) by the AISKYEYE Team, Tianjin University
- Model framework: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Sliced inference: [SAHI](https://github.com/obss/sahi) by OBSS
- AI tools used for code generation and debugging: [Claude](https://claude.ai) (Anthropic) and Codex
