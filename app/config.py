"""Shared constants and paths."""
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "calibrated_model.pth"
DATA_DIR = BASE_DIR / "data"

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.75
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")