"""
STEP 4 — STREAMLIT WEB APP (WITH 3D VIEWER)
=============================================
TWO MODES:
  1. Single slice  — upload 1 CT image → instant diagnosis + heatmap
  2. Patient 3D    — upload ALL slices of one patient → auto-selects best
                     slices, diagnoses each, builds interactive 3D brain model

HOW TO RUN LOCALLY (PC):
  pip install streamlit timm opencv-python torch torchvision plotly -q
  streamlit run step4_app.py

  Models must be in ./models/ folder next to this file.

HOW TO RUN ON COLAB:
  !pip install streamlit timm opencv-python-headless plotly -q
  Then use localtunnel or cloudflared tunnel.
"""

import os, cv2, warnings, io, base64
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm
import streamlit as st
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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
# CONFIG
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
# BRAIN MASK
# ─────────────────────────────────────────────
def get_brain_mask(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    border = int(min(h, w) * 0.10)
    edge = np.zeros_like(gray, dtype=bool)
    edge[:border, :] = edge[-border:, :] = True
    edge[:, :border] = edge[:, -border:] = True
    thresh[edge & (gray > 200)] = 0
    k = np.ones((15, 15), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    if cnts:
        cv2.drawContours(mask, [max(cnts, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return mask





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
# GRAD-CAM
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.grads = self.acts = None
        self._h = []
        tgt = model.blocks[-1]
        self._h = [
            tgt.register_forward_hook(lambda m,i,o: setattr(self,'acts',o.detach())),
            tgt.register_full_backward_hook(lambda m,gi,go: setattr(self,'grads',go[0].detach()))
        ]

    def generate(self, tensor, class_idx=None):
        self.model.zero_grad()
        out = self.model(tensor)
        if class_idx is None:
            class_idx = out.argmax(1).item()
        out[0, class_idx].backward()
        w   = self.grads.mean(dim=[0,2,3])
        cam = self.acts[0].cpu().numpy().copy()
        for i, wi in enumerate(w.cpu().numpy()):
            cam[i] *= wi
        cam = np.maximum(cam.mean(0), 0)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        probs = out.softmax(1)[0].detach().cpu().numpy()
        for h in self._h: h.remove()
        self._h.clear()
        return cam, probs


infer_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


# ─────────────────────────────────────────────
# SINGLE SLICE INFERENCE
# ─────────────────────────────────────────────
def infer_slice(pil_img, s1_model, s2_model):
    orig_np  = np.array(pil_img.convert("RGB"))
    orig_cv  = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
    h, w     = orig_cv.shape[:2]
    tensor   = infer_tf(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    brain_mask = get_brain_mask(orig_cv)

    # Stage 1
    cam1, p1 = GradCAM(s1_model).generate(tensor)
    s1_idx   = int(np.argmax(p1))
    s1_label = S1_CLASSES[s1_idx]
    s1_conf  = round(float(p1[s1_idx])*100, 2)

    if s1_label == "normal":
        brain_rgb = (cv2.cvtColor(orig_cv, cv2.COLOR_BGR2RGB) *
                     (brain_mask[:,:,None]/255)).astype(np.uint8)
        return dict(s1_label="normal", s1_conf=s1_conf,
                    s2_label="N/A", s2_conf=0.0,
                    brain_only_rgb=brain_rgb,
                    heatmap_rgb=None, contour_rgb=None,
                    stroke_px=0, brain_px=int((brain_mask>0).sum()),
                    area_pct=0.0, cam_brain=np.zeros((h,w)), p2=None)

    # Stage 2
    p2 = None; s2_label = "N/A"; s2_conf = 0.0
    if s2_model is not None:
        cam2, p2 = GradCAM(s2_model).generate(tensor)
        s2_idx   = int(np.argmax(p2))
        s2_label = S2_CLASSES[s2_idx]
        s2_conf  = round(float(p2[s2_idx])*100, 2)
        active   = cam2
    else:
        active = cam1

    cam_r       = cv2.resize(active, (w, h))
    brain_f     = brain_mask.astype(np.float32)/255.0
    cam_brain   = cam_r * brain_f
    if cam_brain.max() > 0:
        cam_brain /= cam_brain.max()

    heatmap     = cv2.applyColorMap((cam_brain*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay     = orig_cv.copy()
    bb          = brain_mask > 0
    overlay[bb] = cv2.addWeighted(orig_cv, 0.4, heatmap, 0.6, 0)[bb]
    heatmap_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    stroke_mask = (cam_brain >= 0.5).astype(np.uint8)
    stroke_px   = int(stroke_mask.sum())
    brain_px    = int((brain_mask>0).sum())
    area_pct    = round(stroke_px/max(brain_px,1)*100, 2)
    cnts, _     = cv2.findContours(stroke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    brain_rgb   = (orig_np * (brain_mask[:,:,None]/255)).astype(np.uint8)
    contour_img = brain_rgb.copy()
    cv2.drawContours(contour_img, cnts, -1, (255,50,50), 2)

    return dict(s1_label=s1_label, s1_conf=s1_conf,
                s2_label=s2_label, s2_conf=s2_conf,
                brain_only_rgb=brain_rgb,
                heatmap_rgb=heatmap_rgb, contour_rgb=contour_img,
                stroke_px=stroke_px, brain_px=brain_px,
                area_pct=area_pct, cam_brain=cam_brain, p2=p2)


# ─────────────────────────────────────────────
# SORT SLICES BY FILENAME (anatomical order)
# ─────────────────────────────────────────────
def sort_slices_by_position(files):
    def num_key(f):
        nums = [int(x) for x in
                f.name.replace("_",".").replace("-",".").split(".")
                if x.isdigit()]
        return nums[-1] if nums else 0
    try:
        return sorted(files, key=num_key)
    except Exception:
        return sorted(files, key=lambda f: f.name)


# ─────────────────────────────────────────────
# BUILD REAL-BRAIN-SHAPED 3D MODEL
# ─────────────────────────────────────────────
def build_brain_3d(results, gray_slices, brain_masks):
    """
    Build a real brain-shaped 3D model using brain masks as the surface.
    Every slice is included. Stroke region shown in red inside the brain shape.

    Approach:
    - Brain surface = isosurface of stacked brain masks (actual brain boundary)
    - Stroke region = scatter3d points inside brain where CAM >= 0.5
    - Result: brain-shaped transparent surface + red dots for stroke
    """
    n    = len(gray_slices)
    SIZE = 80  # downsample to 80x80 for performance, keeps brain shape

    # Build volumes: brain mask volume + stroke CAM volume
    mask_vol   = np.zeros((n, SIZE, SIZE), dtype=np.float32)
    stroke_vol = np.zeros((n, SIZE, SIZE), dtype=np.float32)
    gray_vol   = np.zeros((n, SIZE, SIZE), dtype=np.float32)

    for i in range(n):
        mask_vol[i]   = cv2.resize(brain_masks[i].astype(np.float32)/255.0,
                                   (SIZE, SIZE))
        gray_vol[i]   = cv2.resize(gray_slices[i].astype(np.float32)/255.0,
                                   (SIZE, SIZE))
        if results[i]["s1_label"] == "stroke":
            cam = results[i]["cam_brain"]
            stroke_vol[i] = cv2.resize(cam.astype(np.float32), (SIZE, SIZE))

    # Coordinates
    zi, yi, xi = np.mgrid[0:n, 0:SIZE, 0:SIZE]
    # Normalize z to same scale as x,y so brain looks proportional
    z_scale = SIZE / max(n, 1)
    zi_scaled = zi.astype(np.float32) * z_scale

    fig = go.Figure()

    # ── Brain surface (isosurface of mask volume) ──
    # This gives the actual brain SHAPE, not a rectangle
    fig.add_trace(go.Isosurface(
        x=xi.flatten(),
        y=yi.flatten(),
        z=zi_scaled.flatten(),
        value=mask_vol.flatten(),
        isomin=0.45,
        isomax=1.0,
        surface_count=2,
        opacity=0.12,
        colorscale=[[0, "rgb(160,160,160)"], [1, "rgb(210,210,210)"]],
        showscale=False,
        name="Brain surface",
        caps=dict(x_show=False, y_show=False, z_show=False),
    ))

    # ── Stroke region — red scatter points inside brain ──
    # Only show voxels where stroke CAM >= 0.5 AND inside brain mask
    stroke_inside = (stroke_vol >= 0.5) & (mask_vol >= 0.45)
    if stroke_inside.sum() > 0:
        sz = zi_scaled[stroke_inside]
        sy = yi[stroke_inside]
        sx = xi[stroke_inside]
        sv = stroke_vol[stroke_inside]  # intensity for color

        # Limit to max 3000 points for performance
        if len(sx) > 3000:
            idx = np.random.choice(len(sx), 3000, replace=False)
            sx, sy, sz, sv = sx[idx], sy[idx], sz[idx], sv[idx]

        fig.add_trace(go.Scatter3d(
            x=sx, y=sy, z=sz,
            mode="markers",
            marker=dict(
                size=2.5,
                color=sv,
                colorscale=[[0,"rgba(255,100,0,0.5)"],
                            [1,"rgba(255,0,0,1.0)"]],
                opacity=0.85,
                showscale=False,
            ),
            name="Stroke region",
        ))

    # ── Camera angle — show brain from front-top ──
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False,
                       showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False,
                       showticklabels=False, title=""),
            zaxis=dict(showgrid=False, zeroline=False,
                       showticklabels=False, title="Slice (inferior→superior)"),
            bgcolor="rgb(5,5,15)",
            camera=dict(
                eye=dict(x=1.4, y=1.4, z=0.8),
                center=dict(x=0, y=0, z=0)
            ),
            aspectmode="data",
        ),
        paper_bgcolor="rgb(5,5,15)",
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
        title=dict(
            text="3D Brain — Gray=brain surface | Red=stroke region (rotate/zoom freely)",
            font=dict(color="white", size=13),
            x=0.5
        ),
        legend=dict(
            font=dict(color="white", size=12),
            bgcolor="rgba(0,0,0,0.6)",
            x=0.01, y=0.99
        )
    )
    return fig


def build_slice_summary_chart(results):
    """Bar chart of stroke confidence per slice."""
    labels = [f"Slice {i+1}" for i in range(len(results))]
    confs  = [r["s1_conf"] if r["s1_label"]=="stroke" else 0 for r in results]
    colors = ["#E24B4A" if c > 50 else "#4CAF50" for c in confs]

    fig, ax = plt.subplots(figsize=(max(6, len(labels)*0.3), 3))
    ax.bar(labels, confs, color=colors, width=0.7)
    ax.axhline(50, color="white", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Stroke confidence (%)", color="white")
    ax.set_xlabel("Slice", color="white")
    ax.set_title("Stroke detection per slice", color="white")
    ax.tick_params(colors="white", labelsize=7)
    ax.set_facecolor("#111")
    fig.patch.set_facecolor("#111")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    if len(labels) > 20:
        ax.set_xticks([])
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("🧠 CT Stroke Diagnosis")
    st.markdown("""
**Model:** EfficientNet-B4

**Stage 1** — Normal vs Stroke
**Stage 2** — Ischemic vs Hemorrhagic
**Grad-CAM** — Brain region only

---
**Modes:**
- Single slice → quick diagnosis
- Multi-slice → full 3D brain model

---
**Ischemic:** dark/hypodense region
**Hemorrhagic:** bright/hyperdense

---
*Note: Model trained on one hospital
scanner. Different scanners may give
less accurate results.*
    """)
    st.caption(f"Device: {'GPU ✅' if torch.cuda.is_available() else 'CPU'}")

# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
st.title("🧠 AR-Driven CT Brain Stroke Diagnosis")
st.markdown("*Single slice diagnosis + multi-slice 3D brain model with stroke region highlighting.*")
st.markdown("---")

# Load models
s1_model = s2_model = None
models_ok = False
try:
    s1_model, s2_model = load_models()
    models_ok = True
    tag = "GPU ✅" if DEVICE.type=="cuda" else "CPU"
    if s2_model:
        st.success(f"✅ Both models loaded — {tag}")
    else:
        st.warning(f"⚠️ Stage 1 only — {tag}")
except FileNotFoundError as e:
    st.error(f"❌ {e}")
    st.info("Copy both `.pth` files into a `models/` folder next to `step4_app.py`, then relaunch.")
except Exception as e:
    st.error(f"❌ {e}")

# ── Mode selector ─────────────────────────────
mode = st.radio(
    "Select mode",
    ["🔬 Single slice — quick diagnosis",
     "🧠 Multi-slice — full patient 3D model"],
    horizontal=True,
    disabled=not models_ok
)

st.markdown("---")

# ══════════════════════════════════════════════
# MODE 1 — SINGLE SLICE
# ══════════════════════════════════════════════
if mode.startswith("🔬") and models_ok:
    uploaded = st.file_uploader(
        "Upload one CT brain slice (JPG or PNG)",
        type=["jpg","jpeg","png"]
    )

    if uploaded:
        pil_img = Image.open(uploaded)
        with st.spinner("Analyzing..."):
            r = infer_slice(pil_img, s1_model, s2_model)

        st.markdown("---")

        if r["s1_label"] == "normal":
            st.success(f"✅  NORMAL  —  Confidence: {r['s1_conf']}%")
            st.markdown("No stroke detected.")
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Original CT")
                st.image(pil_img, use_container_width=True)
            with c2:
                st.subheader("Brain region (masked)")
                st.image(r["brain_only_rgb"], use_container_width=True)
                st.caption("✅ No stroke region found")
        else:
            st.error(f"⚠️  STROKE DETECTED  —  Confidence: {r['s1_conf']}%")
            if r["s2_label"] == "stroke_ischemic":
                st.info(f"🔵  ISCHEMIC STROKE  —  Confidence: {r['s2_conf']}%")
                st.caption("Blood supply blocked → dark/hypodense region.")
            elif r["s2_label"] == "stroke_hemorrhagic":
                st.warning(f"🔴  HEMORRHAGIC STROKE  —  Confidence: {r['s2_conf']}%")
                st.caption("Blood vessel ruptured → bright/hyperdense region.")
            elif s2_model is None:
                st.info("ℹ️ Stroke type not classified — Stage 2 model unavailable.")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.subheader("Original CT")
                st.image(pil_img, use_container_width=True)
            with c2:
                st.subheader("Stroke heatmap (brain only)")
                st.image(r["heatmap_rgb"], use_container_width=True)
                st.caption("Red/yellow = stroke region inside brain")
            with c3:
                st.subheader("Stroke boundary")
                st.image(r["contour_rgb"], use_container_width=True)
                st.caption("Red contour = stroke boundary")

            st.markdown("---")
            st.subheader("Quantitative analysis")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Stage 1 confidence",    f"{r['s1_conf']}%")
            m2.metric("Stage 2 confidence",    f"{r['s2_conf']}%" if r['p2'] is not None else "N/A")
            m3.metric("Stroke pixels (brain)", f"{r['stroke_px']:,}")
            m4.metric("% of brain affected",   f"{r['area_pct']}%")

            if r["p2"] is not None:
                st.markdown("---")
                fig, ax = plt.subplots(figsize=(5,2))
                ax.barh(["Hemorrhagic","Ischemic"],
                        [r["p2"][0]*100, r["p2"][1]*100],
                        color=["#E24B4A","#378ADD"])
                ax.set_xlim(0,100)
                ax.set_xlabel("Confidence (%)")
                ax.axvline(50, color="gray", linestyle="--", linewidth=0.8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()


# ══════════════════════════════════════════════
# MODE 2 — MULTI-SLICE 3D
# ══════════════════════════════════════════════
elif mode.startswith("🧠") and models_ok:
    st.subheader("Upload all CT slices for one patient")
    st.info(
        "Upload ALL slices for one patient. "
        "Every slice will be diagnosed. "
        "Then all slices are stacked to build a real brain-shaped 3D model "
        "with the stroke region highlighted in red inside the brain."
    )

    uploaded_files = st.file_uploader(
        "Upload all CT slices (JPG or PNG) — select multiple files at once",
        type=["jpg","jpeg","png"],
        accept_multiple_files=True
    )

    if uploaded_files and len(uploaded_files) >= 2:
        # Sort slices in anatomical order by filename
        sorted_files = sort_slices_by_position(uploaded_files)
        st.info(f"📂 {len(sorted_files)} slices found — diagnosing ALL slices...")

        # ── Diagnose every single slice ───────
        results      = []
        gray_slices  = []
        brain_masks  = []
        stroke_count = 0

        progress = st.progress(0, text="Running diagnosis on all slices...")
        for i, f in enumerate(sorted_files):
            f.seek(0)
            pil  = Image.open(f).convert("RGB")
            bgr  = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

            r = infer_slice(pil, s1_model, s2_model)
            results.append(r)

            # Grayscale + brain mask for 3D
            gray_slices.append(np.array(pil.convert("L")))
            brain_masks.append(get_brain_mask(bgr))

            if r["s1_label"] == "stroke":
                stroke_count += 1

            progress.progress(
                (i+1)/len(sorted_files),
                text=f"Slice {i+1}/{len(sorted_files)} — {r['s1_label'].upper()} ({r['s1_conf']}%)"
            )

        # ── Summary ───────────────────────────
        st.markdown("---")
        total_slices = len(results)
        normal_count = total_slices - stroke_count
        stroke_pct   = round(stroke_count/total_slices*100, 1)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total slices", total_slices)
        c2.metric("Stroke slices", f"{stroke_count} ({stroke_pct}%)")
        c3.metric("Normal slices", normal_count)

        if stroke_pct >= 20:
            stroke_rs = [r for r in results if r["s1_label"]=="stroke"]
            avg_conf  = np.mean([r["s1_conf"] for r in stroke_rs])
            types     = [r["s2_label"] for r in stroke_rs if r["s2_label"]!="N/A"]
            majority  = max(set(types), key=types.count) if types else "Unknown"
            c4.metric("Verdict", "STROKE")
            st.error(
                f"⚠️ STROKE DETECTED in {stroke_pct}% of slices — "
                f"Type: {majority.replace('stroke_','').upper()} — "
                f"Avg confidence: {avg_conf:.1f}%"
            )
        elif stroke_pct > 0:
            c4.metric("Verdict", "Possible stroke")
            st.warning(f"⚠️ Possible stroke in {stroke_pct}% of slices — verify manually.")
        else:
            c4.metric("Verdict", "NORMAL")
            st.success("✅ No stroke detected across all slices.")

        # ── Per-slice bar chart ───────────────
        st.markdown("---")
        st.subheader("Stroke detection across all slices")
        chart_fig = build_slice_summary_chart(results)
        st.pyplot(chart_fig)
        plt.close()

        # ── Sample stroke slices ──────────────
        st.markdown("---")
        st.subheader("Stroke slices with heatmap")
        stroke_idx = [(i,r) for i,r in enumerate(results) if r["s1_label"]=="stroke"]
        if stroke_idx:
            # Show up to 4 evenly spaced stroke slices
            show_n  = min(4, len(stroke_idx))
            step    = max(1, len(stroke_idx)//show_n)
            show    = [stroke_idx[i*step] for i in range(show_n)]
            cols    = st.columns(show_n)
            for col, (idx, r) in zip(cols, show):
                with col:
                    st.caption(f"Slice {idx+1} — {r['s1_conf']}%")
                    if r["heatmap_rgb"] is not None:
                        st.image(r["heatmap_rgb"], use_container_width=True)
        else:
            st.info("No stroke slices detected.")

        # ── 3D Brain Model ────────────────────
        st.markdown("---")
        st.subheader("🧠 Interactive 3D Brain Model")
        st.caption(
            "Brain shape built from brain masks of ALL slices. "
            "Gray = brain surface | Red = stroke region. "
            "Drag to rotate, scroll to zoom."
        )

        if len(gray_slices) < 5:
            st.warning("Need at least 5 slices to build 3D model.")
        else:
            with st.spinner("Building brain-shaped 3D model... (~20 seconds)"):
                fig_3d = build_brain_3d(results, gray_slices, brain_masks)
            st.plotly_chart(fig_3d, use_container_width=True)

        # ── CSV report ────────────────────────
        st.markdown("---")
        import csv, io as sio
        buf = sio.StringIO()
        w   = csv.writer(buf)
        w.writerow(["slice","filename","s1_label","s1_conf",
                    "s2_label","s2_conf","stroke_px","brain_px","area_pct"])
        for i, (f, r) in enumerate(zip(sorted_files, results)):
            w.writerow([i+1, f.name, r["s1_label"], r["s1_conf"],
                        r["s2_label"], r["s2_conf"],
                        r["stroke_px"], r["brain_px"], r["area_pct"]])
        st.download_button(
            "⬇️ Download full slice-by-slice report (CSV)",
            buf.getvalue(),
            file_name=f"patient_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    elif uploaded_files and len(uploaded_files) == 1:
        st.warning("Only 1 file uploaded. Use Single Slice mode, or upload all slices.")
    else:
        st.info("👆 Upload all CT slices for one patient to begin.")

elif not models_ok:
    st.info("Fix the model error above first.")
