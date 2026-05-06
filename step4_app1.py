"""
STEP 4 — STREAMLIT WEB APP (FULLY FIXED)
==========================================
FIXES:
  1. Brain mask applied — heatmap shown ONLY inside brain region
  2. Normal images → NO heatmap, NO contour, NO stroke region shown
  3. Stroke images → heatmap clipped to brain boundary only
  4. Scanner border / white arc completely removed from display
  5. GradCAM hook leak fixed
  6. use_column_width → use_container_width (deprecation fix)
  7. Auto-detects Colab vs local model path

HOW TO RUN ON COLAB:
  !pip install streamlit timm opencv-python-headless -q
  !npm install -g localtunnel -q
"""

import os, cv2, warnings
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CT Brain Stroke Diagnosis",
    page_icon="🧠",
    layout="wide"
)

# ─────────────────────────────────────────────
# CONFIG — auto-detect Colab vs local
# ─────────────────────────────────────────────
MODEL_DIR   = "/content/models" if os.path.exists("/content/models") else "./models"
RESULTS_DIR = "/content/results" if os.path.exists("/content") else "./results"
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
S1_CLASSES  = ["normal", "stroke"]
S2_CLASSES  = ["stroke_hemorrhagic", "stroke_ischemic"]

os.makedirs(RESULTS_DIR, exist_ok=True)
for sub in ["normal", "stroke_ischemic", "stroke_hemorrhagic"]:
    os.makedirs(f"{RESULTS_DIR}/{sub}", exist_ok=True)


# ─────────────────────────────────────────────
# BRAIN MASK — removes everything outside brain
# ─────────────────────────────────────────────
def get_brain_mask(img_bgr):
    """
    Returns binary mask (255=brain, 0=outside).
    Removes black background + white scanner arc at edges.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Remove pure black background
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    # Remove bright white scanner arc near edges (top 10% / bottom 10%)
    border = int(min(h, w) * 0.10)
    edge_zone = np.zeros_like(gray, dtype=bool)
    edge_zone[:border, :]  = True
    edge_zone[-border:, :] = True
    edge_zone[:, :border]  = True
    edge_zone[:, -border:] = True
    thresh[edge_zone & (gray > 200)] = 0  # kill bright whites at edges

    # Morphological close to fill gaps in brain
    kernel = np.ones((15, 15), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)

    # Keep only largest contour = brain
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)

    return mask


def apply_brain_mask(img_bgr, brain_mask):
    """Zero out everything outside the brain."""
    result = img_bgr.copy()
    result[brain_mask == 0] = 0
    return result


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    def _load(path, n):
        m = timm.create_model("efficientnet_b4", pretrained=False)
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(m.classifier.in_features, n)
        )
        m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        m.eval()
        return m.to(DEVICE)

    s1_path = f"{MODEL_DIR}/stage1_normal_vs_stroke.pth"
    s2_path = f"{MODEL_DIR}/stage2_ischemic_vs_hemorrhagic.pth"

    if not os.path.exists(s1_path):
        raise FileNotFoundError(f"Stage 1 model not found: {s1_path}")

    s1 = _load(s1_path, 2)
    s2 = _load(s2_path, 2) if os.path.exists(s2_path) else None
    return s1, s2


# ─────────────────────────────────────────────
# GRAD-CAM (fixed — no hook leak)
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model    = model
        self.grads    = None
        self.acts     = None
        self._handles = []
        self._register()

    def _register(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        tgt = self.model.blocks[-1]
        self._handles = [
            tgt.register_forward_hook(
                lambda m, i, o: setattr(self, 'acts', o.detach())),
            tgt.register_full_backward_hook(
                lambda m, gi, go: setattr(self, 'grads', go[0].detach()))
        ]

    def generate(self, tensor, class_idx=None):
        self.model.zero_grad()
        out = self.model(tensor)
        if class_idx is None:
            class_idx = out.argmax(1).item()
        out[0, class_idx].backward()
        w   = self.grads.mean(dim=[0, 2, 3])
        cam = self.acts[0].cpu().numpy().copy()
        for i, wi in enumerate(w.cpu().numpy()):
            cam[i] *= wi
        cam = np.maximum(cam.mean(0), 0)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        probs = out.softmax(1)[0].detach().cpu().numpy()
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return cam, probs


infer_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# MAIN INFERENCE
# ─────────────────────────────────────────────
def run_inference(pil_img, s1_model, s2_model):
    orig_np = np.array(pil_img.convert("RGB"))
    orig_cv = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
    h, w    = orig_cv.shape[:2]
    tensor  = infer_tf(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)

    # Get brain mask — removes arc + background
    brain_mask     = get_brain_mask(orig_cv)
    brain_only_bgr = apply_brain_mask(orig_cv, brain_mask)
    brain_only_rgb = cv2.cvtColor(brain_only_bgr, cv2.COLOR_BGR2RGB)

    # Stage 1: Normal vs Stroke
    cam1, p1 = GradCAM(s1_model).generate(tensor)
    s1_idx   = int(np.argmax(p1))
    s1_label = S1_CLASSES[s1_idx]
    s1_conf  = round(float(p1[s1_idx]) * 100, 2)

    # ── NORMAL: return immediately with NO heatmap ──
    if s1_label == "normal":
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_p = f"{RESULTS_DIR}/normal/result_{ts}.png"
        Image.fromarray(orig_np).save(save_p)
        return dict(
            s1_label="normal", s1_conf=s1_conf,
            s2_label="N/A",    s2_conf=0.0,
            brain_only_rgb=brain_only_rgb,
            heatmap_rgb=None,
            contour_rgb=None,
            stroke_px=0, total_px=int((brain_mask > 0).sum()),
            area_pct=0.0, save_path=save_p, p2=None
        )

    # Stage 2: Stroke type
    p2 = None
    s2_label = "N/A"
    s2_conf  = 0.0

    if s2_model is not None:
        cam2, p2 = GradCAM(s2_model).generate(tensor)
        s2_idx   = int(np.argmax(p2))
        s2_label = S2_CLASSES[s2_idx]
        s2_conf  = round(float(p2[s2_idx]) * 100, 2)
        active_cam = cam2
    else:
        active_cam = cam1

    # Resize CAM to original image size
    cam_r = cv2.resize(active_cam, (w, h))

    # ── CRITICAL: mask CAM to brain region only ──
    brain_mask_f = brain_mask.astype(np.float32) / 255.0
    cam_brain    = cam_r * brain_mask_f
    if cam_brain.max() > 0:
        cam_brain = cam_brain / cam_brain.max()

    # Heatmap overlay — only inside brain
    heatmap    = cv2.applyColorMap((cam_brain * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay    = orig_cv.copy()
    brain_bool = brain_mask > 0
    overlay[brain_bool] = cv2.addWeighted(orig_cv, 0.4, heatmap, 0.6, 0)[brain_bool]
    heatmap_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Stroke region contour — brain only
    stroke_mask    = (cam_brain >= 0.5).astype(np.uint8)
    stroke_px      = int(stroke_mask.sum())
    brain_px       = int((brain_mask > 0).sum())
    area_pct       = round(stroke_px / max(brain_px, 1) * 100, 2)
    contours, _    = cv2.findContours(stroke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img    = brain_only_rgb.copy()
    cv2.drawContours(contour_img, contours, -1, (255, 50, 50), 2)

    # Save result
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"{RESULTS_DIR}/{s2_label if s2_label != 'N/A' else 'stroke_ischemic'}"
    os.makedirs(folder, exist_ok=True)
    combined = np.hstack([orig_np, heatmap_rgb])
    save_p   = f"{folder}/result_{ts}.png"
    Image.fromarray(combined).save(save_p)

    return dict(
        s1_label=s1_label, s1_conf=s1_conf,
        s2_label=s2_label, s2_conf=s2_conf,
        brain_only_rgb=brain_only_rgb,
        heatmap_rgb=heatmap_rgb,
        contour_rgb=contour_img,
        stroke_px=stroke_px, total_px=brain_px,
        area_pct=area_pct, save_path=save_p, p2=p2
    )


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown("""
**AR-Driven CT Brain Stroke Diagnosis**

**Model:** EfficientNet-B4 (fine-tuned)

**Stage 1** — Normal vs Stroke
**Stage 2** — Ischemic vs Hemorrhagic
**Stage 3** — Grad-CAM (brain only)

---
**Ischemic stroke**
Blocked blood supply
Dark/hypodense region on CT

**Hemorrhagic stroke**
Ruptured blood vessel
Bright/hyperdense region on CT

---
*Dataset: Real CT brain images
collected at medical college*
    """)
    st.caption(f"Device: {'GPU ✅' if torch.cuda.is_available() else 'CPU'}")


# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
st.title("🧠 AR-Driven CT Brain Stroke Diagnosis")
st.markdown("*Upload a CT brain slice — stroke detection with brain-region-only highlighting.*")
st.markdown("---")

s1_model = s2_model = None
models_ok = False
try:
    s1_model, s2_model = load_models()
    models_ok = True
    if s2_model:
        st.success(f"✅ Both models loaded — {'GPU' if DEVICE.type=='cuda' else 'CPU'}")
    else:
        st.warning("⚠️ Stage 1 only loaded — stroke type classification unavailable")
except FileNotFoundError as e:
    st.error(f"❌ {e}")
    st.info("Run step2_train.py on Colab first, then relaunch.")
except Exception as e:
    st.error(f"❌ Unexpected error: {e}")

uploaded = st.file_uploader(
    "Upload a CT brain slice (JPG or PNG)",
    type=["jpg", "jpeg", "png"],
    disabled=not models_ok
)

if uploaded and models_ok:
    pil_img = Image.open(uploaded)

    with st.spinner("Analyzing CT scan..."):
        r = run_inference(pil_img, s1_model, s2_model)

    st.markdown("---")

    # ── Result banner ──
    if r["s1_label"] == "normal":
        st.success(f"✅  NORMAL  —  Confidence: {r['s1_conf']}%")
        st.markdown("No stroke detected. Brain appears normal.")
    else:
        st.error(f"⚠️  STROKE DETECTED  —  Confidence: {r['s1_conf']}%")
        if r["s2_label"] == "stroke_ischemic":
            st.info(f"🔵  ISCHEMIC STROKE  —  Confidence: {r['s2_conf']}%")
            st.caption("Blood supply blocked → dark/hypodense region on CT.")
        elif r["s2_label"] == "stroke_hemorrhagic":
            st.warning(f"🔴  HEMORRHAGIC STROKE  —  Confidence: {r['s2_conf']}%")
            st.caption("Blood vessel ruptured → bright/hyperdense region on CT.")
        elif s2_model is None:
            st.info("ℹ️ Stroke type not classified — Stage 2 model not available.")

    st.markdown("---")

    # ── Images ──
    if r["s1_label"] == "normal":
        # NORMAL: only show original + brain region — NO heatmap at all
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Original CT")
            st.image(pil_img, use_container_width=True)
        with c2:
            st.subheader("Brain region (masked)")
            st.image(r["brain_only_rgb"], use_container_width=True)
            st.caption("✅ No stroke region found inside brain")
    else:
        # STROKE: 3 columns — original, heatmap (brain only), contour
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Original CT")
            st.image(pil_img, use_container_width=True)
        with c2:
            st.subheader("Stroke heatmap (brain only)")
            st.image(r["heatmap_rgb"], use_container_width=True)
            st.caption("🔴 Red/yellow = stroke region inside brain only")
        with c3:
            st.subheader("Stroke boundary")
            st.image(r["contour_rgb"], use_container_width=True)
            st.caption("Red contour = stroke region boundary inside brain")

    # ── Metrics (stroke only) ──
    if r["s1_label"] == "stroke":
        st.markdown("---")
        st.subheader("Quantitative analysis")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Stage 1 confidence",      f"{r['s1_conf']}%")
        m2.metric("Stage 2 confidence",      f"{r['s2_conf']}%" if r['p2'] is not None else "N/A")
        m3.metric("Stroke pixels (brain)",   f"{r['stroke_px']:,}")
        m4.metric("% of brain affected",     f"{r['area_pct']}%")

        if r["p2"] is not None:
            st.markdown("---")
            st.subheader("Stage 2 class probabilities")
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.barh(["Hemorrhagic", "Ischemic"],
                    [r["p2"][0]*100, r["p2"][1]*100],
                    color=["#E24B4A", "#378ADD"])
            ax.set_xlim(0, 100)
            ax.set_xlabel("Confidence (%)")
            ax.axvline(50, color="gray", linestyle="--", linewidth=0.8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Download ──
    st.markdown("---")
    with open(r["save_path"], "rb") as f:
        st.download_button(
            label="⬇️  Download result image",
            data=f,
            file_name=f"stroke_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png"
        )
    st.caption(f"Auto-saved to: `{r['save_path']}`")

elif not models_ok:
    st.info("👆 Fix the model error above, then relaunch.")
else:
    st.info("👆 Upload a CT brain slice image to begin analysis.")
