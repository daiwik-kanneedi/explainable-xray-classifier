"""DenseNet121 classifier and temperature-scaling fit."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.optim as optim


class XRayClassifier(nn.Module):
    """ImageNet-pretrained DenseNet121 with a 3-way classification head."""

    def __init__(self, num_classes: int = 3, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "densenet121", pretrained=pretrained, num_classes=num_classes, drop_rate=0.3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit a scalar T by minimising cross-entropy of (logits / T) on validation data."""
    T = nn.Parameter(torch.ones(1, device=logits.device) * 1.5)
    lbfgs = optim.LBFGS([T], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")
    ce = nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        lbfgs.zero_grad()
        loss = ce(logits / T, labels)
        loss.backward()
        return loss

    lbfgs.step(closure)
    return float(T.item())