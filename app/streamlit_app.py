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
# Local storage (gitignored)
# -------------------------
ENROLL_ROOT = Path("data/enrolled")
DB_PATH = ENROLL_ROOT / "users.db"
ENROLL_ROOT.mkdir(parents=True, exist_ok=True)


# -------------------------
# Admin credentials
# Recommended: set env vars before running:
#   setx TONGUEID_ADMIN_KEY "admin"
#   setx TONGUEID_ADMIN_PASSWORD "StrongPassword123"
# Then restart terminal.
# -------------------------
ADMIN_KEY = os.getenv("TONGUEID_ADMIN_KEY", "admin")
ADMIN_PASSWORD = os.getenv("TONGUEID_ADMIN_PASSWORD", "admin123")  # change via env for real use


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
# (includes migration to add "role" column if missing)
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
            created_at TEXT NOT NULL,
            role       TEXT NOT NULL DEFAULT 'user'
        );
        """
    )

    # ---- migration: add role if DB existed without it
    cols = [r[1] for r in con.execute("PRAGMA table_info(users);").fetchall()]
    if "role" not in cols:
        con.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user';")

    return con


def user_exists(unique_key: str) -> bool:
    with db_connect() as con:
        cur = con.execute("SELECT 1 FROM users WHERE unique_key = ?", (unique_key,))
        return cur.fetchone() is not None


def create_user(unique_key: str, username: str, password: str, role: str = "user") -> None:
    salt, pw_hash = hash_password(password)
    created_at = time.strftime("%Y-%m-%d %H:%M:%S")
    with db_connect() as con:
        con.execute(
            "INSERT INTO users(unique_key, username, salt, pw_hash, created_at, role) VALUES(?,?,?,?,?,?)",
            (unique_key, username, salt, pw_hash, created_at, role),
        )


def get_user(unique_key: str) -> Optional[Tuple[str, bytes, bytes, str, str]]:
    with db_connect() as con:
        cur = con.execute(
            "SELECT username, salt, pw_hash, created_at, role FROM users WHERE unique_key = ?",
            (unique_key,),
        )
        row = cur.fetchone()
        return row if row else None


def list_users() -> List[Tuple[str, str, str, str]]:
    with db_connect() as con:
        cur = con.execute("SELECT unique_key, username, created_at, role FROM users ORDER BY created_at DESC")
        return cur.fetchall()


def ensure_admin_account():
    """
    Create admin account in DB if it doesn't exist.
    Admin user is identified by ADMIN_KEY and has role='admin'.
    """
    if user_exists(ADMIN_KEY):
        # If exists, ensure role is admin (in case of old DB)
        with db_connect() as con:
            con.execute("UPDATE users SET role='admin' WHERE unique_key = ?", (ADMIN_KEY,))
        return

    create_user(ADMIN_KEY, "Master Admin", ADMIN_PASSWORD, role="admin")


# -------------------------
# Image helpers
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
    feats = []
    for unique_key, _username, _created, _role in list_users():
        # Skip admin images (admin usually has none)
        if unique_key == ADMIN_KEY:
            continue
        user_dir = ENROLL_ROOT / unique_key
        for p in iter_images(user_dir):
            img = cv2.imread(str(p))
            if img is None:
                continue
            feats.append(extract_features(img))

    if not feats:
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
            embs.append(embedder.embed(img))
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

    if scaler is None:
        scaler = build_global_scaler_for_enrolled()
    f = extract_features(img_bgr)
    f = scaler.transform(f.reshape(1, -1)).squeeze(0)
    return l2norm(f)


# -------------------------
# Session / Auth
# -------------------------
def login(unique_key: str, password: str) -> bool:
    user = get_user(unique_key)
    if not user:
        return False
    username, salt, pw_hash, created_at, role = user
    if not verify_password(password, salt, pw_hash):
        return False

    st.session_state["logged_in"] = True
    st.session_state["unique_key"] = unique_key
    st.session_state["username"] = username
    st.session_state["role"] = role
    return True


def logout():
    st.session_state["logged_in"] = False
    st.session_state["unique_key"] = ""
    st.session_state["username"] = ""
    st.session_state["role"] = ""


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="TongueID Login + Verify", layout="centered")
st.title("TongueID — Login, Enroll & Verify (Local Demo)")
st.caption("All data stored locally in data/enrolled (gitignored).")

ensure_admin_account()

# init session state
if "logged_in" not in st.session_state:
    logout()

# Top right logout
if st.session_state["logged_in"]:
    cols = st.columns([3, 1])
    with cols[0]:
        st.success(f"Logged in as **{st.session_state['username']}** (`{st.session_state['unique_key']}`) — role: **{st.session_state['role']}**")
    with cols[1]:
        if st.button("Logout", key="btn_logout"):
            logout()
            st.rerun()


# -------------------------
# Landing page: Login / Register
# -------------------------
if not st.session_state["logged_in"]:
    tab_login, tab_register = st.tabs(["🔐 Login", "🧾 Register (Enroll)"])

    with tab_login:
        st.subheader("Login")
        with st.form("login_form"):
            uk = st.text_input("Unique Key", key="login_unique_key")
            pw = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login")

        if submitted:
            if login(uk.strip(), pw):
                st.success("Login successful ✅")
                st.rerun()
            else:
                st.error("Invalid unique key or password.")

        st.markdown("---")
        st.info(
            f"Admin login (default): unique key = `{ADMIN_KEY}` and password = `{ADMIN_PASSWORD}`.\n\n"
            "⚠️ Change admin password using environment variables for real use."
        )

    with tab_register:
        st.subheader("Register / Enroll")

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
            elif uk == ADMIN_KEY:
                st.error("This Unique Key is reserved for the admin. Choose another.")
            elif user_exists(uk):
                st.error("This Unique Key already exists. Please choose another one.")
            elif not password or len(password) < 6:
                st.error("Password must be at least 6 characters.")
            elif password != password2:
                st.error("Passwords do not match.")
            elif not files or len(files) < 2:
                st.error("Please upload at least 2 images for enrollment.")
            else:
                create_user(uk, un, password, role="user")
                saved = save_uploaded_images(uk, files)
                st.success(f"User created ✅  Saved {saved} image(s) to {ENROLL_ROOT / uk}")

    st.stop()


# -------------------------
# After login: pages based on role
# -------------------------
role = st.session_state["role"]

if role == "admin":
    # Admin dashboard
    st.header("🛡️ Admin Dashboard")

    st.subheader("All enrolled users")
    rows = [r for r in list_users()]  # (unique_key, username, created_at, role)

    q = st.text_input("Search (Unique Key or Username)", key="admin_search").strip().lower()
    table_rows = []
    for uk, un, created, r in rows:
        if q and (q not in uk.lower() and q not in un.lower()):
            continue
        table_rows.append({"Unique Key": uk, "Username": un, "Created At": created, "Role": r})

    if table_rows:
        st.table(table_rows)
    else:
        st.info("No users match your search.")

    st.markdown("---")
    st.subheader("Verify any user (admin)")
    # Exclude admin account from verify dropdown
    user_choices = [r[0] for r in rows if r[0] != ADMIN_KEY and r[3] == "user"]
    if not user_choices:
        st.warning("No normal users enrolled yet.")
        st.stop()

    selected_user = st.selectbox("Select user to verify", user_choices, key="admin_selected_user")
    mode = st.selectbox("Comparison Mode", ["Deep (ResNet18)", "Handcrafted (scaled)"], key="admin_mode")

    default_thr = 0.90 if mode == "Deep (ResNet18)" else 0.25
    threshold = st.slider("Threshold", 0.0, 1.0, float(default_thr), 0.01, key="admin_threshold")
    uploaded = st.file_uploader("Upload probe image", type=["png", "jpg", "jpeg", "bmp", "webp"], key="admin_probe")

    if st.button("Verify as Admin", key="btn_admin_verify"):
        img = read_uploaded_file(uploaded)
        if img is None:
            st.error("Upload a valid probe image.")
            st.stop()

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Probe image", use_container_width=True)

        scaler = build_global_scaler_for_enrolled() if mode == "Handcrafted (scaled)" else None
        template = build_template(selected_user, mode, scaler=scaler)
        probe = extract_probe_vector(img, mode, scaler=scaler)

        score = cosine_similarity(probe, template)
        decision = "ACCEPT ✅" if score >= threshold else "REJECT ❌"

        st.subheader("Result")
        st.write(f"**User Key:** `{selected_user}`")
        st.write(f"**Mode:** {mode}")
        st.write(f"**Score:** `{score:.4f}`  |  **Threshold:** `{threshold:.2f}`")
        st.write(f"**Decision:** {decision}")

else:
    # Normal user verify page
    st.header("✅ Verify (Logged-in User)")
    st.write(f"You are verifying for **{st.session_state['username']}** (`{st.session_state['unique_key']}`)")

    mode = st.selectbox("Comparison Mode", ["Deep (ResNet18)", "Handcrafted (scaled)"], key="user_mode")
    default_thr = 0.90 if mode == "Deep (ResNet18)" else 0.25
    threshold = st.slider("Threshold (accept if score ≥ threshold)", 0.0, 1.0, float(default_thr), 0.01, key="user_threshold")
    uploaded = st.file_uploader("Upload probe image", type=["png", "jpg", "jpeg", "bmp", "webp"], key="user_probe")

    if st.button("Verify", key="btn_user_verify"):
        uk = st.session_state["unique_key"]

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
        st.write(f"**Mode:** {mode}")
        st.write(f"**Similarity score:** `{score:.4f}`")
        st.write(f"**Threshold:** `{threshold:.2f}`")
        st.write(f"**Decision:** {decision}")
