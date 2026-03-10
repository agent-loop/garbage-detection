# 🚮 Smart Garbage Detection System

Detects garbage piles from a CCTV/webcam feed using a **custom-trained YOLOv5 model**, draws bounding boxes live, and uploads high-confidence detections to a **Flask** API for storage and dashboard use.

## 💥 What this repo contains

- **Backend (Python)**: YOLO inference + Flask API + SQLite storage
- **Frontend (React)**: dashboard UI (login/history/verified/delete views)
- **Android (Kotlin)**: mobile client (optional)

This README is **text-only** (no images) so it stays lightweight and zip-friendly.

## 🚀 Tech used

- **Model/Inference**: YOLOv5 (PyTorch), Ultralytics tooling
- **Computer Vision**: OpenCV, Pillow
- **API**: Flask + flask-cors
- **Storage**: SQLite
- **Frontend**: React
- **Android**: Kotlin (Jetpack libraries)

## 🧠 How the model was created (training summary)

The custom model was trained to detect **garbage piles**.

- **Data collection**: street/CCTV-like images gathered at multiple times of day
- **Annotation**: bounding boxes labeled using **LabelImg**
- **Augmentation**: flips/rotations and common image perturbations (blur/noise/exposure)
- **Model output**: weights file stored at:
  - `Backend/model/best1000.pt`

If you want to retrain or replace the model, put your new weights in `Backend/model/` and update `Backend/main.py` if the filename differs.

## 📁 Project structure (what each file does)

### Backend

- **`Backend/main.py`**: webcam/CCTV inference client
  - Runs YOLO on **every frame**
  - Shows **live bounding boxes**
  - Only **saves/sends** an annotated image once every **60 frames** and only when confidence is high
  - Sends detection image to the API endpoint `POST /add/<addr>/<mac>`
- **`Backend/api.py`**: Flask API server used by the detector + UI
  - Saves uploaded images to `Backend/core/media/`
  - Stores detection metadata in `Backend/car.db.sqlite`
  - Exposes endpoints to fetch/verify/delete rows and login helpers
- **`Backend/config.py`**: backend config
  - Defines `UPLOAD_FOLDER` and ensures it exists
  - Initializes `Backend/garbage.db.sqlite` table (legacy table used by some flows)
- **`Backend/host/api.py` / `Backend/host/config.py`**: alternate API/config copy (same purpose as above)
- **`Backend/model/best1000.pt`**: custom garbage detection weights (required for garbage-only performance)
- **`Backend/core/media/`**: upload destination (created automatically, kept empty in git via `.gitkeep`)
- **`Backend/data/`**: detector snapshots (created automatically, kept empty in git via `.gitkeep`)

### Frontend (React)

- **`Frontend/`**: React app that calls the backend API for data (login/history/verified/delete/dashboard components).

### Android

- **`App/`**: Android app source (Kotlin). Optional; not required to run the Python backend.

## 🛠️ Run (Windows PowerShell)

### 1) Create venv + install deps

From the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Start the backend API

In terminal 1:

```powershell
cd Backend
python api.py
```

### 3) Start live detection (webcam)

In terminal 2:

```powershell
cd Backend
python main.py
```

`main.py` draws bounding boxes live. It **uploads** an annotated image every **60 frames** only if confidence is high.

### 4) (Optional) Start the React UI

```powershell
cd Frontend
npm install
npm start
```

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

