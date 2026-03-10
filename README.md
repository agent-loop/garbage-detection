# 🚮 Smart Garbage Detection System

Detects garbage piles from a CCTV/webcam feed using a **custom-trained YOLOv5 model**, draws bounding boxes live, and uploads high-confidence detections to a **Flask** API for storage and dashboard use.

## 💥 What this repo contains

- **Backend (Python)**: YOLO inference + Flask API + SQLite storage

This README is **text-only** (no images) so it stays lightweight and zip-friendly.

## 🚀 Tech used

- **Model/Inference**: YOLOv5 (PyTorch), Ultralytics tooling
- **Computer Vision**: OpenCV, Pillow
- **API**: Flask + flask-cors
- **Storage**: SQLite

## 🧠 How the model was created (training summary)

The custom model was trained to detect **garbage piles**.

- **Data collection**: street/CCTV-like images gathered at multiple times of day
- **Annotation**: bounding boxes labeled using **LabelImg**
- **Augmentation**: flips/rotations and common image perturbations (blur/noise/exposure)
- **Model output**: weights file stored at:
  - `Backend/model/best1000.pt`

If you want to retrain or replace the model, put your new weights in `Backend/model/` and update `Backend/main.py` if the filename differs.

## 📂 Dataset creation & downloads

### 1) Folder structure

Recommended structure for the raw dataset:

```text
dataset/
  images/
    raw/          # original images from CCTV / internet
  labels/
    raw/          # LabelImg YOLO-format TXT files
```

After splitting into train/val/test for YOLOv5:

```text
dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

### 2) Download links (example)

Host your dataset zip anywhere (Drive, S3, etc.) and link it here:

- **Full dataset (images + labels)**: `<YOUR_DATASET_ZIP_URL>`
- **Trained weights (`best1000.pt`)**: `<YOUR_WEIGHTS_URL>` (optional backup)

After downloading:

- Extract the dataset under a top-level `dataset/` folder
- Place `best1000.pt` at `Backend/model/best1000.pt`

## 🏷️ Annotation with LabelImg (YOLO format)

### 1) Install LabelImg

```powershell
pip install labelImg
labelImg
```

On first run:

- Set **Open Dir** to your `dataset/images/raw` folder
- Set **Save Dir** to `dataset/labels/raw`
- In **PascalVOC / YOLO** menu, select **YOLO** format

Use a **single class** for this project:

- `garbage`

### 2) Basic annotation workflow

1. Draw a bounding box around each visible garbage pile
2. Assign class `garbage`
3. Save – LabelImg writes a `.txt` file next to the image:
   - Same filename as image, YOLO format: `class x_center y_center width height` (normalized 0–1)

Repeat until all images are labeled.

## 🔀 Split dataset into train / val / test

Once `dataset/images/raw` and `dataset/labels/raw` are ready, you can use this helper script to create YOLOv5-style splits:

```python
import os
import random
import shutil
from pathlib import Path

ROOT = Path("dataset")
IMG_RAW = ROOT / "images" / "raw"
LBL_RAW = ROOT / "labels" / "raw"

splits = {"train": 0.7, "val": 0.2, "test": 0.1}

def ensure_dirs():
    for split in splits:
        (ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

def main():
    ensure_dirs()
    images = [p for p in IMG_RAW.glob("*.jpg")] + [p for p in IMG_RAW.glob("*.png")]
    random.shuffle(images)

    n = len(images)
    n_train = int(n * splits["train"])
    n_val = int(n * splits["val"])

    split_indices = {
        "train": images[:n_train],
        "val": images[n_train : n_train + n_val],
        "test": images[n_train + n_val :],
    }

    for split, files in split_indices.items():
        for img_path in files:
            label_path = LBL_RAW / (img_path.stem + ".txt")
            if not label_path.exists():
                continue

            dest_img = ROOT / "images" / split / img_path.name
            dest_lbl = ROOT / "labels" / split / label_path.name

            shutil.copy2(img_path, dest_img)
            shutil.copy2(label_path, dest_lbl)

    print("Done. Images and labels split into train/val/test.")

if __name__ == "__main__":
    main()
```

Run it from the repo root (after creating the `dataset/` folders):

```powershell
python split_dataset.py
```

or simply paste the script into a one-off file and run it once.

## 📁 Project structure (what each file does)

### Backend

- **`Backend/main.py`**: simple webcam garbage detector
  - Runs YOLO on **every frame**
  - Shows **live bounding boxes**
  - Saves an annotated image to `Backend/data/` every **60 frames** when garbage is detected
- **`Backend/model/best1000.pt`**: custom garbage detection weights (required for best performance)
- **`Backend/data/`**: detector snapshots (created automatically, kept empty in git via `.gitkeep`)

## 🛠️ Run (Windows PowerShell)

### 1) Create venv + install deps

From the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Start live detection (webcam)

From the repo root, with venv activated:

```powershell
cd Backend
python main.py
```

`main.py` draws bounding boxes live. It **saves** an annotated image every **60 frames** when garbage is detected into `Backend/data/`.

## 📦 Copy to another PC (zip)

Before zipping, remove local/generated files:

- `.venv/`
- `Backend/core/media/*` (uploaded images)
- `Backend/data/*` (detector snapshots)
- `Backend/*.sqlite`, `Backend/*.db*` (optional; database files are auto-created)

After unzipping on the new PC:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

