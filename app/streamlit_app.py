from __future__ import annotations
from pathlib import Path
import sqlite3
import time
import os
import hashlib
import hmac
from typing import Optional, Tuple, List

import numpy as np
import cv2
import streamlit as st
from sklearn.preprocessing import StandardScaler

from tongueid.metrics import cosine_similarity
from tongueid.features import extract_features
from tongueid.embeddings import ResNetEmbedder, EmbedConfig


# -------------------------
# Paths (all local, gitignored)
# -------------------------
ENROLL_ROOT = Path("data/enrolled")
DB_PATH = ENROLL_ROOT / "users.db"
ENROLL_ROOT.mkdir(parents=True, exist_ok=True)


# -------------------------
# Password hashing (PBKDF2)
# -------------------------
def hash_password(password: str, salt: bytes | None = None, iterations: int = 200_000) -> Tuple[bytes, bytes]:
    if salt is None:
        salt = os.urandom(16)
    pw_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, dklen=32)
    return salt, pw_hash


def verify_password(password: str, salt: bytes, expected_hash: bytes) -> bool:
    test_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000, dklen=32)
    return hmac.compare_digest(test_hash, expected_hash)


# -------------------------
# Database helpers
# -------------------------
def db_connect():
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            unique_key TEXT PRIMARY KEY,
            username   TEXT NOT NULL,
            salt       BLOB NOT NULL,
            pw_hash    BLOB NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    return con


def user_exists(unique_key: str) -> bool:
    with db_connect() as con:
        cur = con.execute("SELECT 1 FROM users WHERE unique_key = ?", (unique_key,))
        return cur.fetchone() is not None


def create_user(unique_key: str, username: str, password: str) -> None:
    salt, pw_hash = hash_password(password)
    created_at = time.strftime("%Y-%m-%d %H:%M:%S")
    with db_connect() as con:
        con.execute(
            "INSERT INTO users(unique_key, username, salt, pw_hash, created_at) VALUES(?,?,?,?,?)",
            (unique_key, username, salt, pw_hash, created_at),
        )


def get_user(unique_key: str) -> Optional[Tuple[str, bytes, bytes, str]]:
    with db_connect() as con:
        cur = con.execute("SELECT username, salt, pw_hash, created_at FROM users WHERE unique_key = ?", (unique_key,))
        row = cur.fetchone()
        return row if row else None


def list_users() -> List[Tuple[str, str]]:
    with db_connect() as con:
        cur = con.execute("SELECT unique_key, username FROM users ORDER BY created_at DESC")
        return cur.fetchall()


# -------------------------
# Image IO
# -------------------------
def save_uploaded_images(unique_key: str, files) -> int:
    user_dir = ENROLL_ROOT / unique_key
    user_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for f in files:
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            continue
        out_path = user_dir / f"{int(time.time() * 1000)}_{saved}.png"
        cv2.imwrite(str(out_path), img)
        saved += 1
    return saved


def iter_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not folder.exists():
        return []
    return sorted([p for p in folder.glob("*") if p.suffix.lower() in exts])


def read_uploaded_file(uploaded) -> Optional[np.ndarray]:
    if uploaded is None:
        return None
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-12)


# -------------------------
# Feature/Embedding templates
# -------------------------
@st.cache_resource
def get_embedder():
    return ResNetEmbedder(EmbedConfig(model_name="resnet18", device="cpu"))


def build_global_scaler_for_enrolled() -> StandardScaler:
    # Fit scaler over ALL enrolled images (handcrafted mode)
    feats = []
    for unique_key, _username in list_users():
        user_dir = ENROLL_ROOT / unique_key
        for p in iter_images(user_dir):
            img = cv2.imread(str(p))
            if img is None:
                continue
            feats.append(extract_features(img))

    if not feats:
        # fallback: scaler that does nothing meaningful (avoid crash)
        scaler = StandardScaler()
        scaler.fit(np.zeros((2, 10), dtype=np.float32))
        return scaler

    X = np.vstack(feats)
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def build_template(unique_key: str, mode: str, scaler: Optional[StandardScaler] = None) -> np.ndarray:
    user_dir = ENROLL_ROOT / unique_key
    images = iter_images(user_dir)
    if not images:
        raise RuntimeError(f"No enrolled images found for {unique_key}.")

    if mode == "Deep (ResNet18)":
        embedder = get_embedder()
        embs = []
        for p in images:
            img = cv2.imread(str(p))
            if img is None:
                continue
            embs.append(embedder.embed(img))  # already normalized
        X = np.vstack(embs)
        return l2norm(X.mean(axis=0))

    # Handcrafted (scaled)
    if scaler is None:
        scaler = build_global_scaler_for_enrolled()

    feats = []
    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            continue
        f = extract_features(img)
        f = scaler.transform(f.reshape(1, -1)).squeeze(0)
        feats.append(l2norm(f))
    X = np.vstack(feats)
    return l2norm(X.mean(axis=0))


def extract_probe_vector(img_bgr: np.ndarray, mode: str, scaler: Optional[StandardScaler] = None) -> np.ndarray:
    if mode == "Deep (ResNet18)":
        embedder = get_embedder()
        return embedder.embed(img_bgr)

    # Handcrafted (scaled)
    if scaler is None:
        scaler = build_global_scaler_for_enrolled()
    f = extract_features(img_bgr)
    f = scaler.transform(f.reshape(1, -1)).squeeze(0)
    return l2norm(f)


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="TongueID Enroll + Verify", layout="centered")
st.title("TongueID — Enroll & Verify (Local Demo)")
st.caption("Stores everything locally under data/enrolled (gitignored). Passwords are stored as hashes (not plain text).")

tab_enroll, tab_verify = st.tabs(["🧾 Enroll (Register)", "✅ Verify (Compare)"])


with tab_enroll:
    st.subheader("Enroll a new user")

    username = st.text_input("Username (display name)", key="enroll_username")
    unique_key = st.text_input("Unique Key (must be unique)", key="enroll_unique_key", help="Example: ilyas01 or user_123")
    password = st.text_input("Password", type="password", key="enroll_password")
    password2 = st.text_input("Confirm Password", type="password", key="enroll_password2")

    files = st.file_uploader(
    "Upload tongue images (PNG/JPG). You can upload multiple images.",
    key="enroll_images",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    accept_multiple_files=True
)

    if st.button("Create Account + Save Images", key="btn_enroll"):
        uk = unique_key.strip()
        un = username.strip()

        if not un:
            st.error("Username is required.")
        elif not uk:
            st.error("Unique Key is required.")
        elif len(uk) < 4:
            st.error("Unique Key must be at least 4 characters.")
        elif user_exists(uk):
            st.error("This Unique Key already exists. Please choose another one.")
        elif not password or len(password) < 6:
            st.error("Password must be at least 6 characters.")
        elif password != password2:
            st.error("Passwords do not match.")
        elif not files or len(files) < 2:
            st.error("Please upload at least 2 images for enrollment.")
        else:
            create_user(uk, un, password)
            saved = save_uploaded_images(uk, files)
            st.success(f"User created ✅  Saved {saved} image(s) to {ENROLL_ROOT / uk}")

    st.markdown("---")
    st.subheader("Existing enrolled users")
    rows = list_users()
    if rows:
        st.write([{ "unique_key": r[0], "username": r[1]} for r in rows])
    else:
        st.info("No users enrolled yet.")


with tab_verify:
    st.subheader("Verify a user (compare probe vs enrolled template)")

    mode = st.selectbox("Comparison Mode", ["Deep (ResNet18)", "Handcrafted (scaled)"])

    default_thr = 0.90 if mode == "Deep (ResNet18)" else 0.25
    threshold = st.slider("Threshold (accept if score ≥ threshold)", 0.0, 1.0, float(default_thr), 0.01)

    unique_key = st.text_input("Unique Key", key="verify_unique_key")
    password = st.text_input("Password", type="password", key="verify_password")
    uploaded = st.file_uploader("Upload probe image", key="verify_probe", type=["png", "jpg", "jpeg", "bmp", "webp"])

    if st.button("Verify", key="btn_verify"):
        uk = unique_key.strip()
        if not uk:
            st.error("Enter Unique Key.")
            st.stop()

        user = get_user(uk)
        if not user:
            st.error("User not found.")
            st.stop()

        username, salt, pw_hash, created_at = user
        if not verify_password(password, salt, pw_hash):
            st.error("Invalid password.")
            st.stop()

        img = read_uploaded_file(uploaded)
        if img is None:
            st.error("Upload a valid probe image.")
            st.stop()

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Probe image", use_container_width=True)

        scaler = build_global_scaler_for_enrolled() if mode == "Handcrafted (scaled)" else None
        template = build_template(uk, mode, scaler=scaler)
        probe = extract_probe_vector(img, mode, scaler=scaler)

        score = cosine_similarity(probe, template)
        decision = "ACCEPT ✅" if score >= threshold else "REJECT ❌"

        st.subheader("Result")
        st.write(f"**User:** {username}  (`{uk}`)")
        st.write(f"**Mode:** {mode}")
        st.write(f"**Similarity score:** `{score:.4f}`")
        st.write(f"**Threshold:** `{threshold:.2f}`")
        st.write(f"**Decision:** {decision}")

        with st.expander("What is happening?"):
            st.write(
                "- You login with Unique Key + password.\n"
                "- The system builds a template from your enrolled images.\n"
                "- It extracts a vector from the probe image.\n"
                "- It compares probe vs template using cosine similarity.\n"
                "- If score ≥ threshold → accept."
            )
