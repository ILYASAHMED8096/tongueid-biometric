from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
import streamlit as st
from sklearn.preprocessing import StandardScaler

from tongueid.features import extract_features
from tongueid.metrics import cosine_similarity
from tongueid.embeddings import ResNetEmbedder, EmbedConfig

DATA_ROOT = Path("data/processed")


def list_users(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("person_")])


def iter_images(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in folder.glob("*") if p.suffix.lower() in exts])


def read_uploaded_file(uploaded) -> np.ndarray | None:
    if uploaded is None:
        return None
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-12)


@st.cache_resource
def get_embedder():
    return ResNetEmbedder(EmbedConfig(model_name="resnet18", device="cpu"))


def build_handcrafted_scaler(root: Path, users: list[str]) -> StandardScaler:
    # Fit scaler on all enrollment data (simple global fit)
    feats = []
    for u in users:
        for p in iter_images(root / u):
            img = cv2.imread(str(p))
            if img is None:
                continue
            feats.append(extract_features(img))
    X = np.vstack(feats)
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def build_handcrafted_template(user_dir: Path, scaler: StandardScaler) -> np.ndarray:
    feats = []
    for p in iter_images(user_dir):
        img = cv2.imread(str(p))
        if img is None:
            continue
        f = extract_features(img)
        f = scaler.transform(f.reshape(1, -1)).squeeze(0)
        feats.append(l2norm(f))
    X = np.vstack(feats)
    return l2norm(X.mean(axis=0))


def build_deep_template(user_dir: Path, embedder: ResNetEmbedder) -> np.ndarray:
    embs = []
    for p in iter_images(user_dir):
        img = cv2.imread(str(p))
        if img is None:
            continue
        embs.append(embedder.embed(img))  # already L2-normalized
    X = np.vstack(embs)
    return l2norm(X.mean(axis=0))


st.set_page_config(page_title="TongueID Demo", layout="centered")
st.title("TongueID — Verification Demo")
st.caption("Public dataset workflow (data/ is gitignored). Choose Handcrafted (scaled) or Deep (ResNet18).")

users = list_users(DATA_ROOT)
if not users:
    st.error("No users found under data/processed. Generate person_XX folders first.")
    st.stop()

method = st.selectbox("Feature method", ["Handcrafted (scaled)", "Deep embeddings (ResNet18)"])

col1, col2 = st.columns(2)
with col1:
    claimed_user = st.selectbox("Claimed identity", users)
with col2:
    threshold = st.slider("Decision threshold (accept if score ≥ threshold)", 0.0, 1.0, 0.25, 0.01)

uploaded = st.file_uploader("Upload a probe image (ROI image preferred)", type=["png", "jpg", "jpeg", "bmp", "webp"])

# Prepare method-specific objects
if method == "Handcrafted (scaled)":
    scaler = build_handcrafted_scaler(DATA_ROOT, users)
else:
    embedder = get_embedder()

if uploaded:
    img = read_uploaded_file(uploaded)
    if img is None:
        st.error("Could not decode image.")
        st.stop()

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Probe image", use_container_width=True)

    user_dir = DATA_ROOT / claimed_user

    if method == "Handcrafted (scaled)":
        template = build_handcrafted_template(user_dir, scaler)
        probe = extract_features(img)
        probe = scaler.transform(probe.reshape(1, -1)).squeeze(0)
        probe = l2norm(probe)
    else:
        template = build_deep_template(user_dir, embedder)
        probe = embedder.embed(img)  # already L2-normalized

    score = cosine_similarity(probe, template)
    decision = "ACCEPT ✅" if score >= threshold else "REJECT ❌"

    st.subheader("Result")
    st.write(f"**Method:** {method}")
    st.write(f"**Similarity score:** `{score:.4f}`")
    st.write(f"**Decision:** {decision}")

    with st.expander("How it works"):
        st.write(
            "- Build a user template by averaging features/embeddings from that user's folder.\n"
            "- Extract features/embeddings from the uploaded probe image.\n"
            "- Compute cosine similarity between probe and template.\n"
            "- If score ≥ threshold → ACCEPT."
        )
