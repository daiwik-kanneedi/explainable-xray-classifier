"""Train DenseNet121, fit temperature on val, save calibrated artifact."""
from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm  # <-- Added progress bar

from app.config import ARTIFACTS_DIR, IMG_SIZE
from app.ml.model import XRayClassifier, fit_temperature

SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def _transforms(train: bool) -> A.Compose:
    aug = (
        [A.HorizontalFlip(p=0.5), A.Affine(scale=(0.95, 1.05), rotate=(-10, 10), p=0.5)]
        if train else []
    )
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.CLAHE(clip_limit=2.0, p=1.0),
        *aug,
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def _get_class_names(data_dir: Path) -> list[str]:
    base = data_dir / "train" if (data_dir / "train").is_dir() else data_dir / "raw"
    if not base.is_dir():
        raise FileNotFoundError(f"Could not find 'train' or 'raw' folder in {data_dir}")
    return sorted([
        d.name for d in base.iterdir() 
        if d.is_dir() and not d.name.startswith(".") and not d.name.startswith("__")
    ])


def _scan(root: Path, class_names: list[str]) -> tuple[list[Path], list[int]]:
    paths: list[Path] = []
    labels: list[int] = []
    for idx, name in enumerate(class_names):
        files = sorted(p for p in (root / name).glob("*") if p.suffix.lower() in IMAGE_EXTS)
        paths += files
        labels += [idx] * len(files)
    if not paths:
        raise FileNotFoundError(f"No images found under {root} for classes {class_names}")
    return paths, labels


def _prepare_data(data_dir: Path, class_names: list[str]) -> None:
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    if train_dir.is_dir() and test_dir.is_dir() and not val_dir.is_dir():
        print("Found train/ and test/ but no val/. Creating 15% validation split from train/...")
        for c_name in class_names:
            c_train = train_dir / c_name
            c_val = val_dir / c_name
            c_val.mkdir(parents=True, exist_ok=True)
            
            files = sorted([p for p in c_train.glob("*") if p.suffix.lower() in IMAGE_EXTS])
            if not files:
                continue
                
            val_size = max(1, math.ceil(len(files) * 0.15))
            if val_size >= len(files):
                val_size = len(files) - 1
                
            _, va_files = train_test_split(files, test_size=val_size, random_state=42)
            for p in va_files:
                shutil.move(str(p), str(c_val / p.name))
        print("Validation split created.")
        return

    if all((data_dir / s).is_dir() for s in SPLITS):
        print("Pre-split train/val/test folders found. Skipping auto-split.")
        return

    raw_dir = data_dir / "raw"
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"No raw directory found at {raw_dir}")
        
    paths, labels = _scan(raw_dir, class_names)
    tr_p, rest_p, tr_y, rest_y = train_test_split(
        paths, labels, train_size=0.70, stratify=labels, random_state=42
    )
    va_p, te_p, va_y, te_y = train_test_split(
        rest_p, rest_y, train_size=0.50, stratify=rest_y, random_state=42
    )
    for split, ps, ys in zip(SPLITS, (tr_p, va_p, te_p), (tr_y, va_y, te_y)):
        for p, y in zip(ps, ys):
            dest = data_dir / split / class_names[y]
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest / p.name)


class _Dataset(Dataset):
    def __init__(self, root: Path, split: str, class_names: list[str], transform: A.Compose) -> None:
        self.paths, self.targets = _scan(root / split, class_names)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        img = cv2.cvtColor(cv2.imread(str(self.paths[i]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return self.transform(image=img)["image"], self.targets[i]


@torch.no_grad()
def _collect(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--data-dir", type=str, default="data")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    data_dir = Path(args.data_dir)

    class_names = _get_class_names(data_dir)
    print(f"Detected classes: {class_names}")

    _prepare_data(data_dir, class_names)
    
    print("\nLoading datasets...")
    ds = {s: _Dataset(data_dir, s, class_names, _transforms(s == "train")) for s in SPLITS}
    
    counts = np.bincount(ds["train"].targets, minlength=len(class_names))
    weights = (1.0 / np.maximum(counts, 1))[ds["train"].targets]
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True
    )
    common = dict(batch_size=args.batch_size, num_workers=2, pin_memory=use_amp)
    train_loader = DataLoader(ds["train"], sampler=sampler, **common)
    val_loader = DataLoader(ds["val"], shuffle=False, **common)
    test_loader = DataLoader(ds["test"], shuffle=False, **common)

    print("Initializing DenseNet121 (Downloading weights if not cached)...")
    model = XRayClassifier(num_classes=len(class_names)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1 = -1.0
    best_path = ARTIFACTS_DIR / "best_model.pth"

    print("\nStarting Training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        
        # Added tqdm progress bar here
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.size(0)
            
            # Update progress bar with current loss
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        vl, vy = _collect(model, val_loader, device)
        f1 = f1_score(vy.numpy(), vl.argmax(1).numpy(), average="macro")
        sched.step()
        print(f"--> Epoch {epoch:02d} done | avg loss {running / len(ds['train']):.4f} | val macro-F1 {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_path)
            print("    [Saved new best model!]")

    print("\nLoading best model for calibration...")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    val_logits, val_labels = _collect(model, val_loader, device)
    temperature = fit_temperature(val_logits.to(device), val_labels.to(device))

    test_logits, test_labels = _collect(model, test_loader, device)
    probs = (test_logits / temperature).softmax(1).numpy()
    
    print(f"\nBest val macro-F1 {best_f1:.4f} | temperature T = {temperature:.4f}\n")
    print(classification_report(test_labels.numpy(), probs.argmax(1), target_names=class_names, digits=4))

    torch.save(
        {"state_dict": model.state_dict(), "temperature": temperature, "class_names": class_names},
        ARTIFACTS_DIR / "calibrated_model.pth",
    )
    print(f"Saved {ARTIFACTS_DIR / 'calibrated_model.pth'}")


if __name__ == "__main__":
    main()