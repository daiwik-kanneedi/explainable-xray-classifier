# Explainable Chest X-Ray Classifier (FastAPI + PyTorch)

**Research prototype — not a diagnostic tool.** Detects COVID-19, Normal, and Pneumonia from chest X-rays with model calibration (temperature scaling) and Grad-CAM explainability.

## 🚀 Features
- **3-Class Classification**: Fine-tuned DenseNet121 for COVID-19, Normal, and Pneumonia.
- **Model Calibration**: Temperature scaling ensures the AI's confidence matches real-world accuracy.
- **Explainability**: Grad-CAM heatmaps highlight the exact lung regions driving the prediction.
- **Clinical PDF Reports**: Download a professional PDF with embedded images, probabilities, and actionable advice.
- **Modern UI**: FastAPI backend with a responsive Jinja2 + HTMX frontend (including Dark Mode).

## 📊 Data
This project uses the [Chest X-Ray COVID19 & Pneumonia Dataset](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia) from Kaggle.

To retrain the model, download the dataset and place it in the `data/` folder. The training script will automatically create a validation split (15%) from the training data. The structure should look like this:

```text
data/
├── train/
│   ├── COVID19/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/