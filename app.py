"""
CIFAR-10 Image Classifier
Streamlit interface with model selector, confidence scores, and Grad-CAM overlay.
"""

import numpy as np
import cv2
import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
from typing import Optional

from src.model import CustomCNN
from src.gradcam import GradCAM

# -- Constants --------------------------------------------

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CLASS_EMOJI = {
    "airplane": "✈️", "automobile": "🚗", "bird": "🐦",
    "cat": "🐱",  "deer": "🦌",  "dog": "🐶",
    "frog": "🐸",  "horse": "🐴", "ship": "🚢", "truck": "🚛",
}

MODEL_PATHS = {
    "Custom CNN":  "src/models/custom_cnn_best.pth",
    "ResNet-18":   "src/models/resnet18_best.pth",
}

MODEL_ACCURACY = {
    "Custom CNN": "~75% test accuracy",
    "ResNet-18":  "~88% test accuracy",
}

TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -- Model loading --------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model(model_name: str) -> Optional[torch.nn.Module]:
    """
    Load a trained model from disk. Cached so it only loads once per session.
    """
    path = MODEL_PATHS[model_name]
    if not Path(path).exists():
        return None

    if model_name == "Custom CNN":
        model = CustomCNN()
    else:
        model = tv_models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 10)

    state = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# -- Inference --------------------------------------------

def predict(
    model: torch.nn.Module,
    image: Image.Image,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """
    Run inference and return top-k predictions with confidence scores.
    """
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze()

    top_probs, top_idxs = torch.topk(probs, top_k)
    return [
        (CLASSES[idx.item()], prob.item())
        for idx, prob in zip(top_idxs, top_probs)
    ]


# -- Grad-CAM --------------------------------------------

def compute_gradcam_overlay(
    model:      torch.nn.Module,
    model_name: str,
    image:      Image.Image,
    class_idx:  int,
) -> Optional[np.ndarray]:
    """
    Compute Grad-CAM heatmap and overlay it on the original image.
    """
    target_layer = (
        model.conv2               # last conv layer in CustomCNN
        if model_name == "Custom CNN"
        else model.layer4[-1].conv2  # last conv inside the last ResNet-18 block
    )

    try:
        gcam    = GradCAM(model, target_layer)
        tensor  = TRANSFORM(image).unsqueeze(0).to(DEVICE)
        heatmap = gcam.generate(tensor, class_idx)

        # Resize heatmap to original image size
        orig_w, orig_h = image.size
        heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))

        # Colorise and blend
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        orig_array    = np.array(image.convert("RGB"))

        overlay = cv2.addWeighted(orig_array, 0.55, heatmap_rgb, 0.45, 0)
        return overlay

    except Exception:
        return None


# -- Page layout --------------------------------------------

st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 CIFAR-10 Image Classifier")
st.caption(
    "Upload any image — the model classifies it into one of 10 categories "
    "and shows where it is looking via Grad-CAM."
)

# -- Sidebar --------------------------------------------

with st.sidebar:
    st.header("Settings")

    model_name = st.radio(
        "Model",
        options=list(MODEL_PATHS.keys()),
        help="Custom CNN is faster. ResNet-18 is more accurate.",
    )
    st.caption(MODEL_ACCURACY[model_name])
    st.divider()

    show_gradcam = st.toggle("Show Grad-CAM heatmap", value=True)
    top_k        = st.slider("Top-K predictions", min_value=1, max_value=10, value=3)
    st.divider()

    st.markdown("**Classes**")
    for cls in CLASSES:
        st.caption(f"{CLASS_EMOJI[cls]} {cls}")

    st.divider()
    st.caption(f"Running on: `{'GPU' if DEVICE.type == 'cuda' else 'CPU'}`")

# -- Load model --------------------------------------------

with st.spinner(f"Loading {model_name}..."):
    model = load_model(model_name)

if model is None:
    st.error(
        f"Model weights not found at `{MODEL_PATHS[model_name]}`. "
        "Run `python src/train.py` first to train the models."
    )
    st.stop()

# -- Upload --------------------------------------------

uploaded = st.file_uploader(
    "Upload an image (jpg, png, jpeg)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.info("Upload an image above to get started.")
    st.stop()

# -- Run inference --------------------------------------------

image = Image.open(uploaded).convert("RGB")

with st.spinner("Classifying..."):
    predictions = predict(model, image, top_k=top_k)

top_class, top_conf = predictions[0]

# -- Layout: image | predictions | gradcam --------------------------------------------

col_img, col_pred, col_cam = st.columns([1, 1, 1])

with col_img:
    st.subheader("Input image")
    st.image(image, use_container_width=True)
    st.caption(f"{image.size[0]}×{image.size[1]}px · {uploaded.name}")

with col_pred:
    st.subheader("Predictions")
    st.metric(
        label="Top prediction",
        value=f"{CLASS_EMOJI[top_class]} {top_class}",
        delta=f"{top_conf:.1%} confidence",
    )
    st.divider()

    for i, (cls, conf) in enumerate(predictions):
        label = f"{CLASS_EMOJI[cls]} {cls}"
        col_label, col_bar = st.columns([1, 2])
        with col_label:
            weight = "**" if i == 0 else ""
            st.markdown(f"{weight}{label}{weight}")
        with col_bar:
            st.progress(conf, text=f"{conf:.1%}")

with col_cam:
    st.subheader("Grad-CAM")
    if show_gradcam:
        with st.spinner("Computing Grad-CAM..."):
            class_idx = CLASSES.index(top_class)
            overlay   = compute_gradcam_overlay(model, model_name, image, class_idx)

        if overlay is not None:
            st.image(overlay, use_container_width=True)
            st.caption(
                f"Regions the model focused on to predict "
                f"**{top_class}** — red = high attention, blue = low attention."
            )
        else:
            st.warning("Grad-CAM failed for this image. Try another.")
    else:
        st.info("Enable Grad-CAM in the sidebar to see model attention.")

# -- Full confidence table --------------------------------------------

with st.expander("Full confidence scores — all 10 classes"):
    all_preds = predict(model, image, top_k=10)
    for cls, conf in all_preds:
        st.progress(conf, text=f"{CLASS_EMOJI[cls]} {cls}: {conf:.2%}")
