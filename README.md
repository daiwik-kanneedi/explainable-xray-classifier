# Explainable Chest X-Ray Classifier 🫁

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.5-009688)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C)

An end-to-end Deep Learning web application for detecting **COVID-19, Pneumonia, and Normal** chest X-rays. 

> **Disclaimer:** This is a research prototype and is **not intended for clinical diagnosis**. Always consult a qualified radiologist for medical interpretation.

## ✨ Features

- **3-Class Classification:** Fine-tuned DenseNet121 architecture for high-accuracy detection.
- **Model Calibration:** Implements Temperature Scaling to ensure confidence scores reflect real-world accuracy (fixing deep learning overconfidence).
- **Explainability (XAI):** Integrates Grad-CAM to generate heatmaps highlighting the exact lung regions driving the prediction.
- **Clinical PDF Reports:** Generates a downloadable, timestamped PDF containing the X-rays, heatmaps, probabilities, and actionable next steps.
- **Modern UI:** A responsive, highly polished web interface with Dark/Light mode, built with FastAPI, Jinja2, and HTMX.

---

## ⚙️ Requirements

- **Python 3.10 - 3.12** (PyTorch is not yet fully stable on 3.13/3.14)
- **PyTorch** (CUDA-enabled GPU recommended for training, CPU is fine for inference)

---

## 🚀 Run on Windows

1. **Clone the repository**
   ```powershell
   git clone https://github.com/daiwik-kanneedi/explainable-xray-classifier.git
   cd explainable-xray-classifier