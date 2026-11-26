# Explainable Chest X-Ray Classifier (FastAPI + PyTorch)

Research prototype — not a diagnostic tool. Detects COVID-19, Normal, and Pneumonia from chest X-rays with model calibration (temperature scaling) and Grad-CAM explainability.

## Data
This project uses the [Chest X-Ray COVID19 & Pneumonia Dataset](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia) from Kaggle.

To retrain the model, download the dataset and place it in the `data/` folder. The script will automatically create a validation split if one doesn't exist. The structure should look like this:

```text
data/
├── train/
│   ├── COVID19/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/