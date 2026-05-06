"""
STAGE 2 TRAINING ONLY — Ischemic vs Hemorrhagic
=================================================
Run this ONLY — Stage 1 is already trained and saved.
Fixes the FileNotFoundError from step2_train.py

RUN ON COLAB:
  !python step2_train_only.py
"""

import os
import copy
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_ROOT   = "/content/data"
MODEL_DIR   = "/content/models"
BATCH_SIZE  = 32
NUM_EPOCHS  = 30
LR          = 1e-4
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# BUILD SEPARATE STAGE 2 DATA FOLDER
# (only ischemic + hemorrhagic, no normal)
# ─────────────────────────────────────────────
import shutil

S2_ROOT = "/content/data_s2"

print("\nBuilding Stage 2 dataset folder (stroke types only)...")
for split in ["train", "val", "test"]:
    for label in ["stroke_ischemic", "stroke_hemorrhagic"]:
        dst = Path(f"{S2_ROOT}/{split}/{label}")
        dst.mkdir(parents=True, exist_ok=True)

        src = Path(f"{DATA_ROOT}/{split}/{label}")
        if not src.exists():
            print(f"  WARNING: {src} not found — skipping")
            continue

        files = list(src.glob("*"))
        print(f"  Linking {split}/{label}: {len(files)} images")

        for f in files:
            dst_file = dst / f.name
            if not dst_file.exists():
                shutil.copy2(f, dst_file)

# Verify counts
print("\nStage 2 dataset counts:")
for split in ["train", "val", "test"]:
    for label in ["stroke_ischemic", "stroke_hemorrhagic"]:
        count = len(list(Path(f"{S2_ROOT}/{split}/{label}").glob("*")))
        print(f"  {split}/{label:<25}: {count}")

# ─────────────────────────────────────────────
# CHECK — if no hemorrhagic images exist
# ─────────────────────────────────────────────
hem_train = len(list(Path(f"{S2_ROOT}/train/stroke_hemorrhagic").glob("*")))
isc_train = len(list(Path(f"{S2_ROOT}/train/stroke_ischemic").glob("*")))

if hem_train == 0:
    print("\n" + "!"*55)
    print("  No hemorrhagic images found in data/train/stroke_hemorrhagic")
    print("  This means step1_autolabel classified ALL infact cases as ischemic.")
    print("  Possible reasons:")
    print("    1. Your dataset is purely ischemic (all patients have ischemic stroke)")
    print("    2. The hemorrhagic threshold needs lowering")
    print("\n  What to do:")
    print("    Option A — If all your cases ARE ischemic:")
    print("      Your Stage 2 is not needed. System works as:")
    print("      Normal vs Stroke (Stage 1 = done, 98% acc)")
    print("      + Grad-CAM region highlight (Step 3)")
    print("      Just skip to: !python step3_inference.py")
    print("\n    Option B — If you believe some cases are hemorrhagic:")
    print("      Lower threshold in step1_autolabel.py:")
    print("      Change MIN_HEMORRHAGIC_PX = 200  →  MIN_HEMORRHAGIC_PX = 80")
    print("      Then re-run step1_autolabel.py (no retraining needed)")
    print("!"*55)
    exit(0)

if isc_train == 0:
    print("\nERROR: No ischemic images found. Check step1_autolabel output.")
    exit(0)

# ─────────────────────────────────────────────
# LOAD DATASETS — from clean s2 folder
# ─────────────────────────────────────────────
s2_datasets = {
    s: datasets.ImageFolder(f"{S2_ROOT}/{s}",
                             transform=train_tf if s == "train" else val_tf)
    for s in ["train", "val", "test"]
}
s2_loaders = {
    s: DataLoader(ds, batch_size=BATCH_SIZE,
                  shuffle=(s == "train"), num_workers=2)
    for s, ds in s2_datasets.items()
}

s2_classes = s2_datasets["train"].classes
print(f"\nStage 2 classes: {s2_classes}")

# ─────────────────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────────────────
def build_model(num_classes):
    model = timm.create_model("efficientnet_b4", pretrained=True)
    in_f  = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_f, num_classes)
    )
    return model.to(DEVICE)

# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
def train_model(model, loaders, criterion, optimizer, scheduler,
                epochs, save_path, class_names):
    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    history  = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}  lr={scheduler.get_last_lr()[0]:.2e}")
        print("─" * 50)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            loss_sum, correct, total = 0.0, 0, 0

            for inputs, labels in loaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    out   = model(inputs)
                    loss  = criterion(out, labels)
                    preds = out.argmax(1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                loss_sum += loss.item() * inputs.size(0)
                correct  += (preds == labels).sum().item()
                total    += inputs.size(0)

            ep_loss = loss_sum / total
            ep_acc  = correct  / total
            history[f"{phase}_loss"].append(ep_loss)
            history[f"{phase}_acc"].append(ep_acc)
            print(f"  {phase:5s}  loss={ep_loss:.4f}  acc={ep_acc:.4f}")

            if phase == "val" and ep_acc > best_acc:
                best_acc = ep_acc
                best_wts = copy.deepcopy(model.state_dict())
                torch.save(best_wts, save_path)
                print(f"  *** Best saved (acc={best_acc:.4f}) ***")

        scheduler.step()

    print(f"\nBest val accuracy: {best_acc:.4f}")
    model.load_state_dict(best_wts)

    # Test evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loaders["test"]:
            out = model(inputs.to(DEVICE))
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n── Test Set Evaluation ─────────────────")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))

    report = classification_report(all_labels, all_preds,
                                    target_names=class_names, output_dict=True)
    rpath  = save_path.replace(".pth", "_report.json")
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {rpath}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"],   label="val")
    axes[1].set_title("Accuracy"); axes[1].legend()
    fig.suptitle("Stage 2 — Ischemic vs Hemorrhagic")
    ppath = save_path.replace(".pth", "_history.png")
    plt.savefig(ppath, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved: {ppath}")

    return model, history


# ─────────────────────────────────────────────
# TRAIN STAGE 2
# ─────────────────────────────────────────────
print("\n" + "═"*55)
print("  STAGE 2 — Ischemic vs Hemorrhagic")
print("═"*55)

model     = build_model(num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

train_model(
    model, s2_loaders, criterion, optimizer, scheduler,
    NUM_EPOCHS,
    save_path=f"{MODEL_DIR}/stage2_ischemic_vs_hemorrhagic.pth",
    class_names=s2_classes
)

print("\nStage 2 training complete!")
print("Next: run step3_inference.py")
