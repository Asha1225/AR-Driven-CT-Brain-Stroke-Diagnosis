"""
STEP 1 — AUTO-LABELING (matches your exact Drive structure)
=============================================================
Your Drive structure:
  MyDrive/
  ├── CT BRAIN INFACT CASES/
  │     ├── 1/   (patient 1 — all slices are stroke)
  │     ├── 2/
  │     └── ... (15 patients)
  └── CT BRAIN NORMAL CASES/
        ├── 1/
        └── ...

Logic:
  - CT BRAIN NORMAL CASES → ALL slices labeled "normal" (folder tells us)
  - CT BRAIN INFACT CASES → intensity check per slice:
        bright white region (180-255) inside brain → stroke_hemorrhagic
        no bright region                           → stroke_ischemic
        (all infact = stroke, we only detect type)

Output: data/train | val | test / normal | stroke_ischemic | stroke_hemorrhagic

RUN ON COLAB:
  from google.colab import drive
  drive.mount('/content/drive')
  !python step1_autolabel.py
"""

import os
import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG — matches your exact Drive folder names
# ─────────────────────────────────────────────
INFACT_ROOT       = "/content/drive/MyDrive/CT BRAIN INFACT CASES"
NORMAL_ROOT       = "/content/drive/MyDrive/CT BRAIN NORMAL CASES"
OUTPUT_ROOT       = "/content/data"

HEMORRHAGIC_LOW   = 180   # bright white = blood leak
HEMORRHAGIC_HIGH  = 255
BRAIN_MASK_THRESH = 15    # ignore black background
MIN_HEMORRHAGIC_PX= 200   # min pixels to confirm hemorrhagic

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# CREATE OUTPUT FOLDERS
# ─────────────────────────────────────────────
for split in ["train", "val", "test"]:
    for label in ["normal", "stroke_ischemic", "stroke_hemorrhagic"]:
        os.makedirs(f"{OUTPUT_ROOT}/{split}/{label}", exist_ok=True)
print("Output folders created.\n")


def get_brain_mask(gray):
    _, mask = cv2.threshold(gray, BRAIN_MASK_THRESH, 255, cv2.THRESH_BINARY)
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    return mask


def is_hemorrhagic(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    gray       = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brain_mask = get_brain_mask(gray)
    brain_only = cv2.bitwise_and(gray, gray, mask=brain_mask)
    hem_mask   = cv2.inRange(brain_only, HEMORRHAGIC_LOW, HEMORRHAGIC_HIGH)
    return cv2.countNonZero(hem_mask) > MIN_HEMORRHAGIC_PX


def split_list(items):
    items = list(items)
    np.random.shuffle(items)
    n = len(items)
    t = int(n * TRAIN_RATIO)
    v = int(n * (TRAIN_RATIO + VAL_RATIO))
    return items[:t], items[t:v], items[v:]


def safe_copy(src, dst_folder):
    src      = Path(src)
    patient  = src.parent.name
    dst_name = f"p{patient}_{src.name}"
    dst      = Path(dst_folder) / dst_name
    c = 1
    while dst.exists():
        dst = Path(dst_folder) / f"{Path(dst_name).stem}_{c}{src.suffix}"
        c += 1
    shutil.copy2(src, dst)


# ── NORMAL CASES ─────────────────────────────
print("── Processing NORMAL cases ─────────────")
normal_imgs = (list(Path(NORMAL_ROOT).rglob("*.jpg")) +
               list(Path(NORMAL_ROOT).rglob("*.JPG")) +
               list(Path(NORMAL_ROOT).rglob("*.png")))
print(f"  Found: {len(normal_imgs)} normal images")

for split_name, files in zip(["train","val","test"], split_list(normal_imgs)):
    for f in tqdm(files, desc=f"  normal → {split_name}"):
        safe_copy(f, f"{OUTPUT_ROOT}/{split_name}/normal")


# ── INFACT (STROKE) CASES ────────────────────
print("\n── Processing INFACT stroke cases ──────")
infact_imgs = (list(Path(INFACT_ROOT).rglob("*.jpg")) +
               list(Path(INFACT_ROOT).rglob("*.JPG")) +
               list(Path(INFACT_ROOT).rglob("*.png")))
print(f"  Found: {len(infact_imgs)} stroke images")
print("  Detecting hemorrhagic vs ischemic...")

ischemic_imgs, hemorrhagic_imgs = [], []
for p in tqdm(infact_imgs, desc="  Classifying"):
    (hemorrhagic_imgs if is_hemorrhagic(p) else ischemic_imgs).append(p)

print(f"\n  Ischemic    : {len(ischemic_imgs):4d} slices")
print(f"  Hemorrhagic : {len(hemorrhagic_imgs):4d} slices")

for label, imgs in [("stroke_ischemic", ischemic_imgs),
                    ("stroke_hemorrhagic", hemorrhagic_imgs)]:
    for split_name, files in zip(["train","val","test"], split_list(imgs)):
        for f in tqdm(files, desc=f"  {label} → {split_name}"):
            safe_copy(f, f"{OUTPUT_ROOT}/{split_name}/{label}")


# ── SUMMARY ──────────────────────────────────
print("\n── Final dataset summary ───────────────")
total = 0
for split in ["train", "val", "test"]:
    for label in ["normal", "stroke_ischemic", "stroke_hemorrhagic"]:
        count = len(list(Path(f"{OUTPUT_ROOT}/{split}/{label}").glob("*")))
        total += count
        print(f"  data/{split}/{label:<22}: {count:4d}")
print(f"\n  Total: {total} images")

if len(hemorrhagic_imgs) < 50:
    print("\nNOTE: Few hemorrhagic images found. If unexpected,")
    print("  lower MIN_HEMORRHAGIC_PX to 100 and re-run.")
print("\nNext: run step2_train.py")
