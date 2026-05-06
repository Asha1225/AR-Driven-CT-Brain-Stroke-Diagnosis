"""
STEP 3 — INFERENCE + GRAD-CAM + PIXEL ANALYSIS
================================================
Runs full 3-stage diagnosis on any CT image or folder.

USAGE on Colab:
  # Single image:
  !python step3_inference.py --image /content/data/test/stroke_hemorrhagic/p1_1.2.840...jpg

  # Entire test folder:
  !python step3_inference.py --folder /content/data/test/stroke_hemorrhagic/
  !python step3_inference.py --folder /content/data/test/normal/
  !python step3_inference.py --folder /content/data/test/stroke_ischemic/

  # Your original Drive folder (new unseen cases):
  !python step3_inference.py --folder "/content/drive/MyDrive/CT BRAIN INFACT CASES/"
"""

import os, cv2, csv, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_DIR   = "/content/models"
RESULTS_DIR = "/content/results"
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class order MUST match what ImageFolder used during training (alphabetical)
S1_CLASSES = ["normal", "stroke"]
S2_CLASSES = ["stroke_hemorrhagic", "stroke_ischemic"]

for d in [f"{RESULTS_DIR}/normal",
          f"{RESULTS_DIR}/stroke_ischemic",
          f"{RESULTS_DIR}/stroke_hemorrhagic"]:
    os.makedirs(d, exist_ok=True)

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

print("Loading models...")
s1_model = load_model(f"{MODEL_DIR}/stage1_normal_vs_stroke.pth", 2)
s2_model = load_model(f"{MODEL_DIR}/stage2_ischemic_vs_hemorrhagic.pth", 2)
print(f"Models loaded. Device: {DEVICE}\n")

infer_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ─────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.grads = None
        self.acts  = None
        tgt = model.blocks[-1]
        tgt.register_forward_hook(
            lambda m,i,o: setattr(self, 'acts', o.detach()))
        tgt.register_full_backward_hook(
            lambda m,gi,go: setattr(self, 'grads', go[0].detach()))

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

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def overlay_heatmap(orig_cv, cam, alpha=0.45):
    h, w    = orig_cv.shape[:2]
    cam_r   = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap((cam_r*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_cv, 1-alpha, heatmap, alpha, 0)
    return overlay, cam_r

def pixel_analysis(cam_r, threshold=0.5):
    mask     = (cam_r >= threshold).astype(np.uint8)
    s_px     = int(mask.sum())
    t_px     = cam_r.size
    area_pct = round(s_px / t_px * 100, 2)
    return s_px, t_px, area_pct, mask

def draw_labels(canvas, s1_label, s1_conf, s2_label, s2_conf,
                stroke_px, area_pct, orig_w):
    font = cv2.FONT_HERSHEY_SIMPLEX
    h    = canvas.shape[0]

    def put(text, x, y, color=(255,255,255)):
        cv2.putText(canvas, text, (x,y), font, 0.45, (0,0,0), 2)
        cv2.putText(canvas, text, (x,y), font, 0.45, color,   1)

    put("Original CT",           10,        20)
    put("Grad-CAM overlay",      orig_w+30, 20)

    if s1_label == "normal":
        put(f"NORMAL  ({s1_conf}%)", 10, h-15, (80,220,80))
    else:
        put(f"STROKE DETECTED ({s1_conf}%)", 10, h-50, (80,80,255))
        color = (80,80,255) if "ischemic" in s2_label else (80,200,255)
        lbl   = "ISCHEMIC" if "ischemic" in s2_label else "HEMORRHAGIC"
        put(f"Type: {lbl}  ({s2_conf}%)",       10,        h-22, color)
        put(f"Region: {stroke_px}px  {area_pct}%", orig_w+30, h-22)

# ─────────────────────────────────────────────
# MAIN INFERENCE FUNCTION
# ─────────────────────────────────────────────
def infer_single(img_path, writer=None):
    img_path = Path(img_path)
    if not img_path.exists():
        print(f"  NOT FOUND: {img_path}"); return None

    pil_img = Image.open(img_path).convert("RGB")
    orig_cv = cv2.imread(str(img_path))
    if orig_cv is None:
        orig_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    tensor  = infer_tf(pil_img).unsqueeze(0).to(DEVICE)

    # Stage 1
    cam1, p1 = GradCAM(s1_model).generate(tensor)
    s1_idx   = int(np.argmax(p1))
    s1_label = S1_CLASSES[s1_idx]
    s1_conf  = round(float(p1[s1_idx])*100, 2)

    # Stage 2 (only for stroke slices)
    if s1_label == "stroke":
        cam2, p2 = GradCAM(s2_model).generate(tensor)
        s2_idx   = int(np.argmax(p2))
        s2_label = S2_CLASSES[s2_idx]
        s2_conf  = round(float(p2[s2_idx])*100, 2)
        active   = cam2
    else:
        s2_label, s2_conf, active = "N/A", 0.0, cam1

    # Overlay + pixel count
    overlay, cam_r           = overlay_heatmap(orig_cv, active)
    s_px, t_px, area_pct, _ = pixel_analysis(cam_r)

    # Side-by-side result image
    h, w   = orig_cv.shape[:2]
    canvas = np.zeros((h, w*2+20, 3), dtype=np.uint8)
    canvas[:, :w]    = orig_cv
    canvas[:, w+20:] = overlay
    draw_labels(canvas, s1_label, s1_conf, s2_label, s2_conf,
                s_px, area_pct, w)

    # Save
    folder = (f"{RESULTS_DIR}/normal" if s1_label=="normal"
              else f"{RESULTS_DIR}/{s2_label}")
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_p  = f"{folder}/{img_path.stem}_{ts}.png"
    cv2.imwrite(out_p, canvas)

    # Console output
    print(f"  {img_path.name}")
    print(f"    Stage 1 : {s1_label.upper():<8}  conf={s1_conf}%")
    if s1_label == "stroke":
        print(f"    Stage 2 : {s2_label.upper():<26}  conf={s2_conf}%")
        print(f"    Region  : {s_px} px  ({area_pct}% of image)")
    print(f"    Saved   : {out_p}")

    if writer:
        writer.writerow({
            "filename":          img_path.name,
            "stage1_result":     s1_label,
            "stage1_confidence": s1_conf,
            "stroke_type":       s2_label,
            "stroke_type_conf":  s2_conf,
            "stroke_pixels":     s_px,
            "total_pixels":      t_px,
            "area_percent":      area_pct,
            "output_file":       out_p,
        })
    return dict(s1=s1_label, s1c=s1_conf, s2=s2_label, s2c=s2_conf,
                px=s_px, pct=area_pct, out=out_p)

# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  type=str, help="Single image path")
    parser.add_argument("--folder", type=str, help="Folder of images")
    args = parser.parse_args()

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"{RESULTS_DIR}/results_{ts}.csv"
    fields   = ["filename","stage1_result","stage1_confidence",
                "stroke_type","stroke_type_conf",
                "stroke_pixels","total_pixels","area_percent","output_file"]

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        if args.image:
            infer_single(args.image, w)

        elif args.folder:
            exts = ["*.jpg","*.JPG","*.jpeg","*.JPEG","*.png","*.PNG"]
            imgs = []
            for ext in exts:
                imgs.extend(Path(args.folder).rglob(ext))
            print(f"Found {len(imgs)} images in {args.folder}\n")
            for img in sorted(imgs):
                infer_single(img, w)
        else:
            print("Usage:")
            print("  python step3_inference.py --image myct.jpg")
            print("  python step3_inference.py --folder /path/to/folder/")

    print(f"\nDone. CSV report: {csv_path}")
    print(f"Results saved to: {RESULTS_DIR}/")
