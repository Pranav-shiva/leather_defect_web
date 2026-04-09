"""
Leather Defect Detection — Streamlit App
All 3 models shown side by side

FOLDER STRUCTURE:
    Streamlit/
    ├── app.py
    ├── mobilenetv2.weights.h5
    ├── efficientnetb0.weights.h5
    ├── resnet50.weights.h5
    └── class_names.json

RUN:
    streamlit run app.py
"""

import os, json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ── Page ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Leather Defect Detector",
    page_icon="🔍",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────
IMG_SIZE    = 224
NUM_CLASSES = 6

if os.path.exists("class_names.json"):
    with open("class_names.json") as f:
        CLASS_NAMES = json.load(f)
else:
    CLASS_NAMES = [
        "Folding marks", "Grain off", "Growth marks",
        "loose grains",  "non defective", "pinhole",
    ]

CLASS_COLORS = {
    "Folding marks":  "#E74C3C",
    "Grain off":      "#E67E22",
    "Growth marks":   "#F39C12",
    "loose grains":   "#9B59B6",
    "non defective":  "#27AE60",
    "pinhole":        "#2980B9",
}

# Model configs — architecture must match exactly what was trained in Colab
MODELS = {
    "MobileNetV2": {
        "weights":    "mobilenetv2.weights.h5",
        "accuracy":   " ",
        "backbone_fn": lambda: tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False, weights=None),
        "preprocess":  tf.keras.applications.mobilenet_v2.preprocess_input,
    },
    "EfficientNetB0": {
        "weights":    "efficientnetb0.weights.h5",
        "accuracy":   " ",
        "backbone_fn": lambda: tf.keras.applications.EfficientNetB0(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False, weights=None,
            ),
        "preprocess":  lambda x: x,   # EfficientNet handles scaling
    },
    "ResNet50": {
        "weights":    "resnet50.weights.h5",
        "accuracy":   " ",
        "backbone_fn": lambda: tf.keras.applications.ResNet50(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False, weights=None),
        "preprocess":  tf.keras.applications.resnet50.preprocess_input,
    },
}


# ── Build & load model (cached) ───────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(name):
    cfg = MODELS[name]
    if not os.path.exists(cfg["weights"]):
        return None
    backbone = cfg["backbone_fn"]()
    backbone.trainable = False
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x   = backbone(inp, training=False)
    x   = tf.keras.layers.GlobalAveragePooling2D()(x)
    x   = tf.keras.layers.Dense(256, activation="relu")(x)
    x   = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.load_weights(cfg["weights"])
    return model


# ── Preprocessing ─────────────────────────────────────────────────
def preprocess(pil_image, name):
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    arr = MODELS[name]["preprocess"](arr)
    return np.expand_dims(arr, axis=0)


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Leather Defect Detector")
    st.markdown("**Research Project**")
    st.markdown("---")
    st.markdown("**Models:**")
    for name, cfg in MODELS.items():
        exists = os.path.exists(cfg["weights"])
        icon   = "✅" if exists else "❌"
        st.markdown(f"{icon} {name} — {cfg['accuracy']}")
    st.markdown("---")
    st.markdown("**Defect Classes:**")
    for cls in CLASS_NAMES:
        color = CLASS_COLORS.get(cls, "#888")
        st.markdown(
            f"<span style='color:{color};font-size:15px'>●</span> {cls}",
            unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Upload a leather image to compare all 3 models.")


# ── Main ──────────────────────────────────────────────────────────
st.title("🔍 Leather Defect Detection")
st.markdown("Upload a leather surface image — all 3 models predict simultaneously.")

uploaded = st.file_uploader(
    "Choose a leather image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded is None:
    st.info("👆 Upload a leather surface image to get started.")
    st.stop()

pil_image = Image.open(uploaded)

# Show uploaded image centered
col_l, col_mid, col_r = st.columns([1, 2, 1])
with col_mid:
    st.image(pil_image, caption="Uploaded Image", use_container_width=True)

st.markdown("---")
st.subheader("📊 All Model Predictions")

# One column per model
cols = st.columns(3)

for col, (model_name, cfg) in zip(cols, MODELS.items()):
    with col:
        st.markdown(
            f"<h3 style='text-align:center'>{model_name}</h3>",
            unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align:center;color:#888;font-size:13px'>"
            f"Test accuracy: {cfg['accuracy']}</p>",
            unsafe_allow_html=True)

        # Load model
        with st.spinner(f"Loading {model_name}..."):
            model = load_model(model_name)

        if model is None:
            st.error(
                f"❌ `{cfg['weights']}` not found.\n\n"
                f"Download it from Colab and place it in this folder.")
            continue

        # Predict
        with st.spinner("Predicting..."):
            tensor = preprocess(pil_image, model_name)
            probs  = model.predict(tensor, verbose=0)[0]

        top_idx  = int(np.argmax(probs))
        top_cls  = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else str(top_idx)
        top_conf = float(probs[top_idx]) * 100
        color    = CLASS_COLORS.get(top_cls, "#888")

        # Result card
        st.markdown(
            f"""
            <div style="
                background:{color}22;
                border:2px solid {color};
                border-radius:10px;
                padding:14px;
                text-align:center;
                margin:10px 0;">
                <div style="font-size:11px;color:#888;letter-spacing:1px;">
                    PREDICTION
                </div>
                <div style="font-size:20px;font-weight:bold;
                            color:{color};margin:6px 0;">
                    {top_cls}
                </div>
                <div style="font-size:14px;color:#555;">
                    {top_conf:.1f}% confidence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if top_cls == "non defective":
            st.success("✅ No defect")
        else:
            st.warning(f"⚠️ {top_cls}")

        # Probability bars
        st.markdown("**Class probabilities:**")
        for i in np.argsort(probs)[::-1]:
            cls  = CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i)
            prob = float(probs[i])
            col2 = CLASS_COLORS.get(cls, "#888")
            bold = "**" if i == top_idx else ""
            c1, c2, c3 = st.columns([3, 4, 2])
            with c1:
                st.markdown(f"<small>{bold}{cls}{bold}</small>",
                            unsafe_allow_html=True)
            with c2:
                st.progress(float(prob))
            with c3:
                st.markdown(
                    f"<small style='color:{col2};font-weight:bold'>"
                    f"{prob*100:.1f}%</small>",
                    unsafe_allow_html=True)

st.markdown("---")
st.caption("Leather Defect Detection | MobileNetV2 · EfficientNetB0 · ResNet50 | Research Project")
