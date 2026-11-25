# Explainable Chest X-Ray Classifier (FastAPI + PyTorch)

Research prototype — not a diagnostic tool.

## Setup (Python 3.12)
    python -m pip install -r requirements.txt

## Data
Put Kaggle COVID-19 Radiography images under `data/raw/{COVID,Normal,Viral Pneumonia}/`.
Then train (auto 70/15/15 stratified split into `data/{train,val,test}/`):

    python -m scripts.train --epochs 15

Output: `artifacts/calibrated_model.pth` (weights + temperature + class names).

## Run the web app
    uvicorn app.main:app --reload --port 8000

Open http://localhost:8000 → upload a chest X-ray → calibrated probabilities + Grad-CAM.

## Structure
- `app/main.py` — FastAPI routes (`GET /`, `POST /predict`)
- `app/ml/model.py` — DenseNet121 + temperature fit
- `app/ml/inference.py` — inference + Grad-CAM + base64 (singleton)
- `app/templates/*.html` — Jinja2 + HTMX
- `scripts/train.py` — training + calibration