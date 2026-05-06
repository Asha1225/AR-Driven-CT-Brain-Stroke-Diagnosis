"""
STEP 5 — RESEARCH PAPER METRICS
=================================
Generates all figures and tables needed for your paper:
  - Confusion matrix plots (Stage 1 + Stage 2)
  - AUC-ROC curves (Stage 1 + Stage 2)
  - Sensitivity, Specificity, F1 per class
  - Full metrics summary table
  - Sample Grad-CAM visualization grid

RUN ON COLAB:
  !python step5_metrics.py

Output saved to: /content/paper_figures/
"""

import os, json, warnings
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import timm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, auc, ConfusionMatrixDisplay)
from pathlib import Path
import cv2
from PIL import Image
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_DIR   = "/content/models"
DATA_ROOT   = "/content/data"
S2_ROOT     = "/content/data_s2"
OUT_DIR     = "/content/paper_figures"
IMG_SIZE    = 224
BATCH_SIZE  = 32
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

S1_CLASSES  = ["normal", "stroke"]
S2_CLASSES  = ["stroke_hemorrhagic", "stroke_ischemic"]

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Device: {DEVICE}")
print(f"Figures will be saved to: {OUT_DIR}\n")

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
def load_model(path, num_classes):
    m = timm.create_model("efficientnet_b4", pretrained=False)
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(m.classifier.in_features, num_classes)
    )
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m.to(DEVICE)

s1_model = load_model(f"{MODEL_DIR}/stage1_normal_vs_stroke.pth", 2)
s2_model = load_model(f"{MODEL_DIR}/stage2_ischemic_vs_hemorrhagic.pth", 2)
print("Models loaded.")

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ─────────────────────────────────────────────
# EVALUATE MODEL ON TEST SET
# ─────────────────────────────────────────────
def evaluate(model, data_root, class_names):
    ds     = datasets.ImageFolder(f"{data_root}/test", transform=val_tf)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            out   = model(inputs.to(DEVICE))
            probs = out.softmax(1).cpu().numpy()
            preds = out.argmax(1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return (np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs))

print("\nEvaluating Stage 1 on test set...")
s1_labels, s1_preds, s1_probs = evaluate(s1_model, DATA_ROOT, S1_CLASSES)

print("Evaluating Stage 2 on test set...")
s2_labels, s2_preds, s2_probs = evaluate(s2_model, S2_ROOT, S2_CLASSES)

# ─────────────────────────────────────────────
# 1. CONFUSION MATRICES
# ─────────────────────────────────────────────
def plot_cm(labels, preds, class_names, title, save_path):
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5,4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")

plot_cm(s1_labels, s1_preds, S1_CLASSES,
        "Stage 1 — Normal vs Stroke",
        f"{OUT_DIR}/fig1_confusion_stage1.png")

plot_cm(s2_labels, s2_preds, S2_CLASSES,
        "Stage 2 — Ischemic vs Hemorrhagic",
        f"{OUT_DIR}/fig2_confusion_stage2.png")

# ─────────────────────────────────────────────
# 2. AUC-ROC CURVES
# ─────────────────────────────────────────────
def plot_roc(labels, probs, class_names, title, save_path):
    fig, ax = plt.subplots(figsize=(5,5))
    colors  = ["#E24B4A","#378ADD","#1D9E75","#EF9F27"]

    for i, (cls, col) in enumerate(zip(class_names, colors)):
        y_bin = (labels == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, probs[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=2,
                label=f"{cls} (AUC = {roc_auc:.3f})")

    ax.plot([0,1],[0,1], "k--", lw=1)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")

plot_roc(s1_labels, s1_probs, S1_CLASSES,
         "AUC-ROC — Stage 1 Normal vs Stroke",
         f"{OUT_DIR}/fig3_roc_stage1.png")

plot_roc(s2_labels, s2_probs, S2_CLASSES,
         "AUC-ROC — Stage 2 Ischemic vs Hemorrhagic",
         f"{OUT_DIR}/fig4_roc_stage2.png")

# ─────────────────────────────────────────────
# 3. METRICS SUMMARY TABLE
# ─────────────────────────────────────────────
def print_metrics(labels, preds, class_names, stage_name):
    print(f"\n{'═'*55}")
    print(f"  {stage_name}")
    print(f"{'═'*55}")
    print(classification_report(labels, preds, target_names=class_names))
    cm = confusion_matrix(labels, preds)
    print("Confusion matrix:")
    print(cm)

    # Per-class sensitivity + specificity
    print("\nSensitivity (Recall) and Specificity per class:")
    for i, cls in enumerate(class_names):
        tp = cm[i,i]
        fn = cm[i,:].sum() - tp
        fp = cm[:,i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens = tp/(tp+fn) if (tp+fn)>0 else 0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0
        print(f"  {cls:<26} Sensitivity={sens:.4f}  Specificity={spec:.4f}")

print_metrics(s1_labels, s1_preds, S1_CLASSES, "STAGE 1 — Normal vs Stroke")
print_metrics(s2_labels, s2_preds, S2_CLASSES, "STAGE 2 — Ischemic vs Hemorrhagic")

# ─────────────────────────────────────────────
# 4. TRAINING HISTORY PLOTS (combined)
# ─────────────────────────────────────────────
def plot_combined_history(save_path):
    s1_path = f"{MODEL_DIR}/stage1_normal_vs_stroke_report.json"
    s2_path = f"{MODEL_DIR}/stage2_ischemic_vs_hemorrhagic_report.json"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, path, title in [
        (axes[0], s1_path, "Stage 1 — Normal vs Stroke"),
        (axes[1], s2_path, "Stage 2 — Ischemic vs Hemorrhagic"),
    ]:
        if Path(path).exists():
            with open(path) as f:
                r = json.load(f)
            # Print key metrics on the plot
            acc = r.get("accuracy", 0)
            ax.text(0.5, 0.5,
                    f"Test Accuracy: {acc*100:.1f}%\n\nSee confusion matrix\nand ROC curve figures.",
                    ha="center", va="center", fontsize=12,
                    transform=ax.transAxes)
            ax.set_title(title, fontweight="bold")
            ax.axis("off")
        else:
            ax.set_title(f"{title}\n(report not found)")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")

plot_combined_history(f"{OUT_DIR}/fig5_accuracy_summary.png")

# ─────────────────────────────────────────────
# 5. GRAD-CAM SAMPLE GRID (for paper figure)
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.grads = None
        self.acts  = None
        tgt = model.blocks[-1]
        tgt.register_forward_hook(
            lambda m,i,o: setattr(self,'acts',o.detach()))
        tgt.register_full_backward_hook(
            lambda m,gi,go: setattr(self,'grads',go[0].detach()))

    def generate(self, tensor, class_idx=None):
        self.model.zero_grad()
        out = self.model(tensor)
        if class_idx is None:
            class_idx = out.argmax(1).item()
        out[0, class_idx].backward()
        w   = self.grads.mean(dim=[0,2,3])
        cam = self.acts[0].cpu().numpy()
        for i, wi in enumerate(w.cpu().numpy()):
            cam[i] *= wi
        cam = np.maximum(cam.mean(0), 0)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        probs = out.softmax(1)[0].detach().cpu().numpy()
        return cam, probs

infer_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def make_gradcam_grid(save_path, n_samples=3):
    """Create a grid: normal | hemorrhagic | ischemic with Grad-CAM overlays."""
    categories = [
        ("normal",             DATA_ROOT, S1_CLASSES,  s1_model, "Normal"),
        ("stroke_hemorrhagic", S2_ROOT,   S2_CLASSES,  s2_model, "Hemorrhagic Stroke"),
        ("stroke_ischemic",    S2_ROOT,   S2_CLASSES,  s2_model, "Ischemic Stroke"),
    ]

    fig = plt.figure(figsize=(12, 4*n_samples))
    gs  = gridspec.GridSpec(n_samples, len(categories)*2,
                            wspace=0.05, hspace=0.3)

    for col_idx, (label, data_root, classes, model, title) in enumerate(categories):
        folder = Path(f"{data_root}/test/{label}")
        imgs   = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        if not imgs:
            print(f"  No images found for {label}, skipping.")
            continue
        imgs = imgs[:n_samples]

        for row_idx, img_path in enumerate(imgs):
            pil   = Image.open(img_path).convert("RGB")
            cv_img= cv2.imread(str(img_path))
            if cv_img is None:
                cv_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            t     = infer_tf(pil).unsqueeze(0).to(DEVICE)

            cam, probs = GradCAM(model).generate(t)
            h, w  = cv_img.shape[:2]
            cam_r = cv2.resize(cam, (w,h))
            hmap  = cv2.applyColorMap((cam_r*255).astype(np.uint8), cv2.COLORMAP_JET)
            over  = cv2.addWeighted(cv_img, 0.55, hmap, 0.45, 0)
            over_rgb = cv2.cvtColor(over, cv2.COLOR_BGR2RGB)

            # Original
            ax1 = fig.add_subplot(gs[row_idx, col_idx*2])
            ax1.imshow(pil)
            ax1.axis("off")
            if row_idx == 0:
                ax1.set_title(f"{title}\nOriginal", fontsize=9, fontweight="bold")

            # Overlay
            ax2 = fig.add_subplot(gs[row_idx, col_idx*2+1])
            ax2.imshow(over_rgb)
            ax2.axis("off")
            conf = round(float(probs[classes.index(label) if label in classes else 0])*100, 1)
            if row_idx == 0:
                ax2.set_title(f"Grad-CAM\n({conf}%)", fontsize=9)

    plt.suptitle("Grad-CAM Visualization — CT Brain Stroke Detection",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")

print("\nGenerating Grad-CAM sample grid for paper...")
make_gradcam_grid(f"{OUT_DIR}/fig6_gradcam_grid.png", n_samples=3)

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "═"*55)
print("  ALL FIGURES SAVED TO:", OUT_DIR)
print("═"*55)
print("  fig1_confusion_stage1.png   — Confusion matrix S1")
print("  fig2_confusion_stage2.png   — Confusion matrix S2")
print("  fig3_roc_stage1.png         — AUC-ROC curve S1")
print("  fig4_roc_stage2.png         — AUC-ROC curve S2")
print("  fig5_accuracy_summary.png   — Accuracy summary")
print("  fig6_gradcam_grid.png       — Grad-CAM sample grid")
print("\nUse these directly in your research paper.")
