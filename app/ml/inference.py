"""Lazy-loaded predictor: image bytes -> calibrated probabilities + Grad-CAM overlay."""
from __future__ import annotations

import base64
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from app.config import CONFIDENCE_THRESHOLD, DEVICE, IMG_SIZE, MODEL_PATH
from app.ml.model import XRayClassifier

_PRE = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.CLAHE(clip_limit=2.0, p=1.0)])
_NORM = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def _b64_png(rgb_uint8: np.ndarray) -> str:
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


class Predictor:
    """Process-wide singleton; loads weights once on first access."""

    _instance: "Predictor | None" = None

    def __new__(cls) -> "Predictor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        self.class_names: list[str] = list(ckpt["class_names"])
        self.temperature: float = float(ckpt["temperature"])
        self.model = XRayClassifier(num_classes=len(self.class_names), pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(DEVICE).eval()

    def predict(self, file_bytes: bytes, opacity: float = 0.55) -> dict[str, Any]:
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        image = cv2.cvtColor(cv2.imdecode(arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        rgb = _PRE(image=image)["image"]
        tensor = _NORM(image=rgb)["image"].unsqueeze(0).to(DEVICE)

        # 1. Get predictions
        with torch.no_grad():
            logits = self.model(tensor)
            raw = logits.softmax(dim=1)[0].cpu().numpy()
            calibrated = (logits / self.temperature).softmax(dim=1)[0].cpu().numpy()
            target = int(calibrated.argmax())

        # 2. Generate Grad-CAM
        with GradCAM(model=self.model, target_layers=[self.model.backbone.features[-1]]) as cam:
            heatmap = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(target)])[0]
            
        overlay = show_cam_on_image(rgb.astype(np.float32) / 255.0, heatmap, use_rgb=True)
        blended = cv2.addWeighted(overlay, opacity, rgb, 1.0 - opacity, 0)

        # 3. Setup Explanations and Advice
        target_class = self.class_names[target]
        
        explanations = {
            "NORMAL": "Lungs appear clear with no obvious signs of infection or fluid.",
            "PNEUMONIA": "Cloudy areas (opacities) detected in the lungs, consistent with fluid or inflammation.",
            "COVID19": "Bilateral ground-glass opacities detected, a pattern commonly associated with viral infections."
        }
        
        advice_map = {
            "NORMAL": {
                "severity": "safe",
                "steps": ["Lungs appear clear.", "No obvious signs of active infection.", "Maintain routine health checkups."]
            },
            "PNEUMONIA": {
                "severity": "alert",
                "steps": ["Signs of lung inflammation/opacities detected.", "Please consult a doctor for a formal diagnosis.", "If experiencing shortness of breath or high fever, seek immediate medical care."]
            },
            "COVID19": {
                "severity": "alert",
                "steps": ["Bilateral opacities detected, consistent with viral infections.", "Isolate to prevent potential spread.", "Consult a healthcare provider for a PCR test and medical advice."]
            }
        }
        
        current_advice = advice_map.get(target_class, {"severity": "warning", "steps": ["Consult a radiologist."]})

        # 4. Return the final dictionary
        return {
            "image": _b64_png(rgb),
            "heatmap": _b64_png(blended),
            "predicted": target_class,
            "explanation": explanations.get(target_class, "Consult a radiologist for a detailed diagnosis."),
            "severity": current_advice["severity"],
            "advice": current_advice["steps"],
            "calibrated_confidence": float(calibrated[target]),
            "raw_confidence": float(raw[target]),
            "temperature": self.temperature,
            "probabilities": [
                {"name": n, "raw": float(r), "calibrated": float(c)}
                for n, r, c in zip(self.class_names, raw, calibrated)
            ],
            "low_confidence": bool(calibrated[target] < CONFIDENCE_THRESHOLD),
        }