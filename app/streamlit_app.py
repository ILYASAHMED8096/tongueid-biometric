from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
import streamlit as st

from tongueid.features import extract_features
from tongueid.metrics import cosine_similarity

DATA_ROOT = Path("data/processed")  # expects person_XX folders


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
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def build_template_for_user(user_dir: Path) -> np.ndarray:
    feats = []
    for p in iter_images(user_dir):
        img = cv2.imread(str(p))
        if img is None:
            continue
        feats.append(extract_features(img))
    if not feats:
        raise RuntimeError(f"No images found in {user_dir}")
    X = np.vstack(feats)
    return X.mean(axis=0)


st.set_page_config(page_title="TongueID Demo", layout="centered")
st.title("TongueID — Biometric Verification Demo (Public Dataset Workflow)")
st.caption("Uploads are processed locally; data/ is gitignored. Uses cosine similarity over engineered features.")

users = list_users(DATA_ROOT)
if not users:
    st.error("No users found under data/processed. Run your dataset scripts first.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    claimed_user = st.selectbox("Claimed identity", users)
with col2:
    threshold = st.slider("Decision threshold (accept if score ≥ threshold)", 0.0, 1.0, 0.65, 0.01)

uploaded = st.file_uploader("Upload a probe image (ROI image preferred)", type=["png", "jpg", "jpeg", "bmp", "webp"])

if uploaded:
    img = read_uploaded_file(uploaded)
    if img is None:
        st.error("Could not decode image.")
        st.stop()

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Probe image", use_container_width=True)

    probe_feat = extract_features(img)

    user_dir = DATA_ROOT / claimed_user
    template = build_template_for_user(user_dir)

    score = cosine_similarity(probe_feat, template)
    decision = "ACCEPT ✅" if score >= threshold else "REJECT ❌"

    st.subheader("Result")
    st.write(f"**Similarity score:** `{score:.4f}`")
    st.write(f"**Decision:** {decision}")

    with st.expander("How it works"):
        st.write(
            "- For each user folder, we build a template feature vector by averaging features.\n"
            "- We extract features from the uploaded probe image.\n"
            "- We compute cosine similarity between probe and template.\n"
            "- If score ≥ threshold → ACCEPT, else REJECT."
        )
