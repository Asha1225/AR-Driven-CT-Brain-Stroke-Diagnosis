"""
STEP 2 — MODEL TRAINING
========================
Trains TWO EfficientNet-B4 models:
  Model A (Stage 1): Normal vs Stroke
  Model B (Stage 2): Stroke_ischemic vs Stroke_hemorrhagic

HOW TO RUN ON COLAB:
  !pip install timm
  !python step2_train.py

Saved models:
  models/stage1_normal_vs_stroke.pth
  models/stage2_ischemic_vs_hemorrhagic.pth
"""

import os
import time
import copy
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
import json

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


def build_model(num_classes):
    """EfficientNet-B4 with custom classifier head."""
    model = timm.create_model("efficientnet_b4", pretrained=True)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    return model.to(DEVICE)


def train_model(model, dataloaders, criterion, optimizer, scheduler,
                num_epochs, save_path, class_names):
    best_acc   = 0.0
    best_wts   = copy.deepcopy(model.state_dict())
    history    = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}  lr={scheduler.get_last_lr()[0]:.2e}")
        print("─" * 50)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss, running_correct, total = 0.0, 0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss    = criterion(outputs, labels)
                    preds   = outputs.argmax(dim=1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss    += loss.item() * inputs.size(0)
                running_correct += (preds == labels).sum().item()
                total           += inputs.size(0)

            epoch_loss = running_loss / total
            epoch_acc  = running_correct / total
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)
            print(f"  {phase:5s}  loss={epoch_loss:.4f}  acc={epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_wts = copy.deepcopy(model.state_dict())
                torch.save(best_wts, save_path)
                print(f"  *** Best model saved (acc={best_acc:.4f}) ***")

        scheduler.step()

    print(f"\nTraining complete. Best val accuracy: {best_acc:.4f}")
    model.load_state_dict(best_wts)

    # Final evaluation on test set
    evaluate_model(model, dataloaders["test"], class_names, save_path)

    return model, history


def evaluate_model(model, test_loader, class_names, save_path):
    """Full evaluation — confusion matrix + classification report."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\n── Test Set Evaluation ─────────────────")
    print(classification_report(all_labels, all_preds,
                                 target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix:")
    print(cm)

    # Save report
    report = classification_report(all_labels, all_preds,
                                    target_names=class_names,
                                    output_dict=True)
    report_path = save_path.replace(".pth", "_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {report_path}")


def plot_history(history, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"],   label="val")
    axes[1].set_title("Accuracy"); axes[1].legend()
    fig.suptitle(title)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved: {save_path}")


# ════════════════════════════════════════════
# STAGE 1 — Normal vs Stroke
# ════════════════════════════════════════════
print("\n" + "═"*55)
print("  STAGE 1 — Normal vs Stroke classifier")
print("═"*55)

# Merge ischemic + hemorrhagic into one "stroke" folder for Stage 1
import shutil
from pathlib import Path

for split in ["train", "val", "test"]:
    stroke_dir = Path(f"{DATA_ROOT}_s1/{split}/stroke")
    stroke_dir.mkdir(parents=True, exist_ok=True)
    normal_dir = Path(f"{DATA_ROOT}_s1/{split}/normal")
    normal_dir.mkdir(parents=True, exist_ok=True)

    # Copy normal
    src_normal = Path(f"{DATA_ROOT}/{split}/normal")
    if src_normal.exists():
        for f in src_normal.iterdir():
            shutil.copy2(f, normal_dir / f.name)

    # Merge both stroke types
    for stroke_type in ["stroke_ischemic", "stroke_hemorrhagic"]:
        src = Path(f"{DATA_ROOT}/{split}/{stroke_type}")
        if src.exists():
            for f in src.iterdir():
                dst_name = f"{stroke_type}_{f.name}"
                shutil.copy2(f, stroke_dir / dst_name)

# Build dataloaders for stage 1
s1_datasets = {
    s: datasets.ImageFolder(f"{DATA_ROOT}_s1/{s}",
                             transform=train_tf if s == "train" else val_tf)
    for s in ["train", "val", "test"]
}
s1_loaders = {
    s: DataLoader(ds, batch_size=BATCH_SIZE,
                  shuffle=(s == "train"), num_workers=2)
    for s, ds in s1_datasets.items()
}

s1_classes = s1_datasets["train"].classes
print(f"Stage 1 classes: {s1_classes}")
for s, ds in s1_datasets.items():
    print(f"  {s}: {len(ds)} images")

s1_model     = build_model(num_classes=2)
s1_criterion = nn.CrossEntropyLoss()
s1_optimizer = Adam(s1_model.parameters(), lr=LR, weight_decay=1e-4)
s1_scheduler = CosineAnnealingLR(s1_optimizer, T_max=NUM_EPOCHS)

s1_model, s1_history = train_model(
    s1_model, s1_loaders, s1_criterion, s1_optimizer, s1_scheduler,
    NUM_EPOCHS,
    save_path=f"{MODEL_DIR}/stage1_normal_vs_stroke.pth",
    class_names=s1_classes
)
plot_history(s1_history, "Stage 1 — Normal vs Stroke",
             f"{MODEL_DIR}/stage1_history.png")


# ════════════════════════════════════════════
# STAGE 2 — Ischemic vs Hemorrhagic
# ════════════════════════════════════════════
print("\n" + "═"*55)
print("  STAGE 2 — Ischemic vs Hemorrhagic classifier")
print("═"*55)

s2_datasets = {}
for split in ["train", "val", "test"]:
    isch_dir = Path(f"{DATA_ROOT}/{split}/stroke_ischemic")
    hem_dir  = Path(f"{DATA_ROOT}/{split}/stroke_hemorrhagic")

    if not isch_dir.exists() or not hem_dir.exists():
        print(f"WARNING: Missing stroke folders in {split}. Skipping.")
        continue

    s2_datasets[split] = datasets.ImageFolder(
        f"{DATA_ROOT}/{split}",
        transform=train_tf if split == "train" else val_tf,
        # Only load ischemic and hemorrhagic (exclude normal)
        is_valid_file=lambda p: (
            "stroke_ischemic" in p or "stroke_hemorrhagic" in p
        )
    )

s2_loaders = {
    s: DataLoader(ds, batch_size=BATCH_SIZE,
                  shuffle=(s == "train"), num_workers=2)
    for s, ds in s2_datasets.items()
}

if s2_datasets:
    s2_classes = s2_datasets["train"].classes
    print(f"Stage 2 classes: {s2_classes}")
    for s, ds in s2_datasets.items():
        print(f"  {s}: {len(ds)} images")

    s2_model     = build_model(num_classes=2)
    s2_criterion = nn.CrossEntropyLoss()
    s2_optimizer = Adam(s2_model.parameters(), lr=LR, weight_decay=1e-4)
    s2_scheduler = CosineAnnealingLR(s2_optimizer, T_max=NUM_EPOCHS)

    s2_model, s2_history = train_model(
        s2_model, s2_loaders, s2_criterion, s2_optimizer, s2_scheduler,
        NUM_EPOCHS,
        save_path=f"{MODEL_DIR}/stage2_ischemic_vs_hemorrhagic.pth",
        class_names=s2_classes
    )
    plot_history(s2_history, "Stage 2 — Ischemic vs Hemorrhagic",
                 f"{MODEL_DIR}/stage2_history.png")
else:
    print("No stroke type data found. Check step1 output.")

print("\nAll models trained. Next: run step3_inference.py")
