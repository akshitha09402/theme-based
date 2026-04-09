import io
import json
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import requests


MODEL_PATH = Path("models/breed_model.tflite")
LABELS_PATH = Path("models/labels.txt")
CLASS_INDICES_PATH = Path("models/class_indices.json")
IMG_SIZE = 192


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


def _load_labels(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError("labels.txt is empty.")
    return labels


def _load_labels_from_class_indices(path: Path) -> list[str]:
    """
    Expects Keras-style mapping: {"Ayrshire": 0, "Brown_Swiss": 1, ...}
    Produces an index-aligned list: labels[index] == class_name
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError("class_indices.json must be a non-empty JSON object.")

    items: list[tuple[str, int]] = []
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, int):
            raise ValueError("class_indices.json must map strings to integers.")
        items.append((k, v))

    max_idx = max(v for _, v in items)
    labels = [""] * (max_idx + 1)
    for name, idx in items:
        if idx < 0 or idx >= len(labels):
            raise ValueError("class_indices.json contains out-of-range index values.")
        if labels[idx]:
            raise ValueError(f"Duplicate index {idx} in class_indices.json.")
        labels[idx] = name

    if any(not x for x in labels):
        raise ValueError("class_indices.json indices must be contiguous starting at 0.")
    return labels


@st.cache_resource(show_spinner=False)
def _load_interpreter(model_path: Path):
    interpreter = None
    errors: list[str] = []

    # Preferred runtime for hosted environments where full TensorFlow is unavailable.
    try:
        from ai_edge_litert.interpreter import Interpreter as LiteRtInterpreter

        interpreter = LiteRtInterpreter(model_path=str(model_path))
    except Exception as e:
        errors.append(f"ai-edge-litert: {e}")

    # Fallback to tflite-runtime if available.
    if interpreter is None:
        try:
            from tflite_runtime.interpreter import Interpreter as TFLiteRuntimeInterpreter

            interpreter = TFLiteRuntimeInterpreter(model_path=str(model_path))
        except Exception as e:
            errors.append(f"tflite-runtime: {e}")

    # Local/dev fallback when TensorFlow is installed.
    if interpreter is None:
        try:
            import tensorflow as tf  # type: ignore

            interpreter = tf.lite.Interpreter(model_path=str(model_path))
        except Exception as e:
            errors.append(f"tensorflow: {e}")

    if interpreter is None:
        raise RuntimeError(
            "No compatible TFLite interpreter found. Install one of: "
            "`ai-edge-litert`, `tflite-runtime`, or `tensorflow`.\n"
            + "\n".join(errors)
        )

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def _preprocess_image(image: Image.Image, preprocessing: str) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = np.array(image, dtype=np.float32)
    x = np.expand_dims(x, axis=0)  # (1, H, W, 3)

    if preprocessing == "efficientnet":
        # TensorFlow-free path for hosted environments: keep identity behavior.
        # For many TF EfficientNet pipelines, this is equivalent.
        pass
    elif preprocessing == "scale_0_1":
        x = x / 255.0
    elif preprocessing == "raw_0_255":
        pass
    else:
        raise ValueError("Unknown preprocessing option.")

    return x


def _predict(interpreter, input_details, output_details, x: np.ndarray) -> np.ndarray:
    # Ensure dtype matches model input
    input_dtype = input_details[0]["dtype"]
    if x.dtype != input_dtype:
        x = x.astype(input_dtype)

    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(output_details[0]["index"])
    y = np.squeeze(y)
    return y


def _top_k(probs: np.ndarray, labels: list[str], k: int) -> list[Prediction]:
    k = max(1, min(int(k), len(labels)))
    idxs = np.argsort(probs)[::-1][:k]
    return [Prediction(labels[i], float(probs[i])) for i in idxs]


@st.cache_data(show_spinner=False, ttl=60 * 60)
def _wikipedia_summary(title: str) -> dict:
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
    r = requests.get(url, timeout=10, headers={"User-Agent": "breed-detector-streamlit/1.0"})
    if r.status_code != 200:
        return {"ok": False, "status_code": r.status_code}
    data = r.json()
    return {
        "ok": True,
        "title": data.get("title"),
        "extract": data.get("extract"),
        "content_urls": data.get("content_urls", {}),
    }


def _google_search_link(query: str) -> str:
    return f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}"


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def _feed_items_from_google(breed_label: str) -> dict:
    """
    Attempts to extract common feed items from Google result snippets.
    Falls back gracefully if scraping is blocked/unavailable.
    """
    breed_name = breed_label.replace("_", " ").strip()
    query = f"best feed for {breed_name} cattle buffalo breed"
    url = _google_search_link(query)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }

    # Canonical item names mapped to possible snippet keywords.
    keyword_map = {
        "Green fodder (Napier/Berseem/Lucerne)": ["green fodder", "napier", "berseem", "lucerne", "alfalfa"],
        "Dry fodder (hay/straw)": ["dry fodder", "hay", "straw"],
        "Silage (maize/sorghum)": ["silage", "maize silage", "corn silage", "sorghum silage"],
        "Concentrate mix (grains + oil cake)": ["concentrate", "grain", "maize", "barley", "oil cake", "cottonseed cake"],
        "Mineral mixture": ["mineral mixture", "mineral mix"],
        "Salt": ["salt", "lick salt"],
        "Clean drinking water": ["water", "clean water", "fresh water"],
    }

    try:
        r = requests.get(url, timeout=10, headers=headers)
        if r.status_code != 200:
            return {"ok": False, "items": [], "query": query, "status": r.status_code}

        html = re.sub(r"<[^>]+>", " ", r.text).lower()
        html = re.sub(r"\s+", " ", html)

        found_items: list[str] = []
        for item_name, keys in keyword_map.items():
            if any(k in html for k in keys):
                found_items.append(item_name)

        return {"ok": True, "items": found_items, "query": query, "status": 200}
    except requests.RequestException:
        return {"ok": False, "items": [], "query": query, "status": None}


st.set_page_config(page_title="Breed Detector (TFLite)", page_icon="🐄", layout="centered")

st.title("Cattle/Buffalo Breed Detector")
st.caption("Upload an image → get breed prediction + confidence (TFLite).")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.75, step=0.01)
    top_k = st.slider("Show top‑k", min_value=3, max_value=10, value=5, step=1)
    st.subheader("Input preprocessing")
    preprocessing = st.selectbox(
        "Preprocessing mode",
        options=[
            "efficientnet",
            "scale_0_1",
            "raw_0_255",
        ],
        index=0,
        help=(
            "If predictions look consistently wrong, this is often the reason. "
            "Try the three modes and keep the one that gives the best results."
        ),
    )
    st.subheader("Label mapping")
    labels_source = st.radio(
        "Labels source",
        options=["auto", "labels.txt", "class_indices.json"],
        index=0,
        help=(
            "If predictions are consistently shifted (A shows as B), it's almost always a label-order issue. "
            "`class_indices.json` is the most reliable if you have it from training."
        ),
    )


if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    st.error("Missing model files.")
    st.markdown(
        f"""
Place these files, then reload:

- `{MODEL_PATH.as_posix()}`
- `{LABELS_PATH.as_posix()}`
"""
    )
    st.stop()


def _resolve_labels() -> list[str]:
    if labels_source == "labels.txt":
        return _load_labels(LABELS_PATH)
    if labels_source == "class_indices.json":
        return _load_labels_from_class_indices(CLASS_INDICES_PATH)

    # auto
    if CLASS_INDICES_PATH.exists():
        return _load_labels_from_class_indices(CLASS_INDICES_PATH)
    return _load_labels(LABELS_PATH)


labels = _resolve_labels()
interpreter, input_details, output_details = _load_interpreter(MODEL_PATH)

with st.expander("Debug: model + labels", expanded=False):
    st.write("Labels count:", len(labels))
    st.write("First 15 labels:", labels[:15])
    st.write("TFLite input dtype/shape:", input_details[0]["dtype"], input_details[0]["shape"].tolist())
    st.write("TFLite output dtype/shape:", output_details[0]["dtype"], output_details[0]["shape"].tolist())

uploaded = st.file_uploader("Upload an image (jpg/png/webp)", type=["jpg", "jpeg", "png", "webp"])

if uploaded is None:
    st.info("Waiting for an image upload.")
    st.stop()

raw_bytes = uploaded.read()
image = Image.open(io.BytesIO(raw_bytes))

st.image(image, caption="Uploaded image", use_container_width=True)

with st.spinner("Running inference..."):
    x = _preprocess_image(image, preprocessing=preprocessing)
    probs = _predict(interpreter, input_details, output_details, x)

if probs.ndim != 1 or probs.shape[0] != len(labels):
    st.error(
        f"Model output shape {tuple(probs.shape)} does not match labels count ({len(labels)}). "
        "Make sure you are using the correct `labels.txt` for this model."
    )
    st.stop()

top = _top_k(probs, labels, k=top_k)
best = top[0]

col1, col2 = st.columns(2)
with col1:
    st.metric("Prediction", best.label)
with col2:
    st.metric("Confidence", f"{best.confidence * 100:.2f}%")

if best.confidence < threshold:
    st.warning(
        f"Low confidence (below {threshold:.2f}). Showing top‑{len(top)} predictions instead of accepting a single answer."
    )

st.subheader("Top predictions")
st.dataframe(
    {
        "label": [p.label for p in top],
        "confidence": [round(p.confidence, 6) for p in top],
        "confidence_%": [round(p.confidence * 100, 2) for p in top],
    },
    use_container_width=True,
    hide_index=True,
)

st.subheader("Feed items for this breed")
st.caption("Shows likely feed items using Google search snippets.")

feed = _feed_items_from_google(best.label)
if feed.get("ok") and feed.get("items"):
    st.write(f"Recommended feed items for **{best.label.replace('_', ' ')}**:")
    for item in feed["items"]:
        st.markdown(f"- {item}")
else:
    st.warning(
        "Could not fetch reliable feed items from Google right now "
        "(network/rate-limit/anti-bot). Showing a safe generic list."
    )
    generic_items = [
        "Green fodder (Napier/Berseem/Lucerne)",
        "Dry fodder (hay/straw)",
        "Silage (maize/sorghum)",
        "Concentrate mix (grains + oil cake)",
        "Mineral mixture + salt",
        "Clean drinking water",
    ]
    st.write(f"Generic feed items for **{best.label.replace('_', ' ')}**:")
    for item in generic_items:
        st.markdown(f"- {item}")

st.markdown(f"[Open Google feed search]({_google_search_link(feed['query'])})")

with st.expander("Breed description (reference summary)", expanded=False):
    # Try Wikipedia for a short description; keep it optional and fully in-app.
    wiki = _wikipedia_summary(best.label.replace("_", " "))
    if wiki.get("ok") and (wiki.get("extract") or "").strip():
        st.write(wiki.get("title", best.label))
        st.write(wiki.get("extract"))
    else:
        st.info(
            "No reference summary found for this exact breed label. "
            "You can still use the nutrition panel above."
        )

