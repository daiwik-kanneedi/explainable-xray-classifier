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
   ```

2. **Create a virtual environment**
   ```powershell
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   ```powershell
   .venv\Scripts\Activate.ps1
   ```
   *(Note: If you get an execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first).*

4. **Install dependencies**
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

5. **Start the web app**
   ```powershell
   uvicorn app.main:app --reload --port 8000
   ```

6. **Open your browser and go to `http://127.0.0.1:8000`**

---

## 🍎 Run on macOS / Linux

1. **Clone the repository**
   ```bash
   git clone https://github.com/daiwik-kanneedi/explainable-xray-classifier.git
   cd explainable-xray-classifier
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   ```

3. **Activate the virtual environment**
   ```bash
   source .venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Start the web app**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

6. **Open your browser and go to `http://127.0.0.1:8000`**

---

## 🩺 First-time Walkthrough

1. **Upload an Image:** Click the upload box and select a chest X-ray (`.png`, `.jpg`, or `.jpeg`).
2. **Analyze:** Click the blue "Analyze Image" button. The model will process the image locally.
3. **Review Results:**
   - **Triage Header:** Instantly see a color-coded result (Green for Normal, Red for COVID/Pneumonia).
   - **Grad-CAM Heatmap:** View the original X-ray side-by-side with the AI's explanation heatmap.
   - **Probability Table:** See the difference between the *Raw* (uncalibrated) and *Calibrated* confidence scores.
4. **Download PDF:** Click "Download Clinical PDF Report" to save a formatted document with all the findings.

---

## 📊 Data & Training (Optional)

This project uses the [Chest X-Ray COVID19 & Pneumonia Dataset](https://kaggle.com) from Kaggle.

*Note: The trained model weights (`artifacts/calibrated_model.pth`) are already included in this repository. You only need the dataset if you want to retrain the model.*

To retrain the model:

1. Download the dataset and place it in the `data/` folder.
2. Ensure the structure looks like this:
   ```text
   data/
   ├── train/
   │   ├── COVID19/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   └── test/
   ```
3. Run the training script:
   ```bash
   python -m scripts.train --data-dir data --epochs 10
   ```
   *(The script will automatically create a 15% validation split if one doesn't exist).*

---

## 🧠 Detection Methodology

- **Architecture:** We use a DenseNet121 backbone pre-trained on ImageNet, modified for a 3-class output.
- **Class Imbalance:** The training loop utilizes a `WeightedRandomSampler` to ensure minority classes (like COVID-19) are adequately represented.
- **Calibration:** Deep neural networks are notoriously overconfident. We fit a single scalar temperature parameter $T$ on the validation set using LBFGS optimization to minimize Negative Log-Likelihood (NLL), drastically reducing Expected Calibration Error (ECE).
- **Explainability:** Grad-CAM extracts gradients from the final convolutional layer to produce coarse localization maps, ensuring the model is looking at lung parenchyma rather than text markers or edges.
