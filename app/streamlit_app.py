from __future__ import annotations

from pathlib import Path
import sqlite3
import time
import os
import hashlib
import hmac
import shutil
from typing import Optional, Tuple, List
import base64
import secrets
import smtplib
from email.message import EmailMessage

import numpy as np
import cv2
import streamlit as st
from sklearn.preprocessing import StandardScaler

from tongueid.metrics import cosine_similarity
from tongueid.features import extract_features
from tongueid.embeddings import ResNetEmbedder, EmbedConfig


# =========================================================
# 0) BRAND
# =========================================================
APP_NAME = "LinguaNet"
TAGLINE = "A Neural Network-Based Tongue Biometric Recognition System"


# =========================================================
# 1) UI (Left strip background + centered content on right)
# =========================================================
def inject_ui_css():
    st.markdown(
        """
        <style>
        :root{
            --leftw: 30vw;
            --gap: 2rem;
            --contentw: 920px;
        }

        .stApp { background: #ffffff !important; }
        html, body { color: #0f172a !important; }

        div[data-testid="stAppViewContainer"]::before{
            content:"";
            position: fixed;
            top: 0; left: 0;
            width: var(--leftw);
            height: 100vh;
            background: #f3f5ff;
            border-right: 1px solid rgba(15,23,42,0.08);
            z-index: 0;
            pointer-events: none;
        }

        .left-panel{
            position: fixed;
            top: 0;
            left: 0;
            width: var(--leftw);
            height: 100vh;
            z-index: 2;
            pointer-events: none;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .left-panel img{
            width: min(520px, 90%);
            height: auto;
            image-rendering: auto;
            transform: translateZ(0);
            filter:
                drop-shadow(0 18px 45px rgba(15,23,42,0.18))
                saturate(1.05)
                contrast(1.05);
            border-radius: 18px;
            background: transparent;
        }

        div[data-testid="stMainBlockContainer"]{
            position: relative;
            z-index: 1;
            padding-left: calc(var(--leftw) + var(--gap)) !important;
            padding-right: var(--gap) !important;
            padding-top: 2.2rem !important;
            padding-bottom: 2rem !important;
            max-width: none !important;
        }

        .right-center{
            max-width: var(--contentw);
            margin: 0 auto;
        }

        .soft-divider{
            height: 1px;
            background: linear-gradient(
                90deg, transparent,
                rgba(34,197,94,0.25),
                rgba(124,58,237,0.25),
                rgba(236,72,153,0.25),
                transparent
            );
            margin: 12px 0 14px 0;
        }

        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea{
            background: #ffffff !important;
            border: 1px solid rgba(15,23,42,0.14) !important;
            border-radius: 14px !important;
            color: #0f172a !important;
            padding: 12px 12px !important;
        }

        .stButton > button{
            border-radius: 14px !important;
            border: 1px solid rgba(15,23,42,0.14) !important;
            background: linear-gradient(90deg, rgba(124,58,237,0.96), rgba(34,197,94,0.90)) !important;
            color: white !important;
            font-weight: 900 !important;
            padding: 0.55rem 0.9rem !important;
            box-shadow: 0 10px 26px rgba(124,58,237,0.18) !important;
        }

        button[data-baseweb="tab"]{
            font-weight: 900 !important;
            border-radius: 14px !important;
        }

        footer { visibility: hidden; }

        @media (max-width: 900px){
            :root{ --leftw: 0vw; --gap: 1.25rem; --contentw: 100%; }
            .left-panel{ display:none; }
            div[data-testid="stAppViewContainer"]::before{ display:none; }
            div[data-testid="stMainBlockContainer"]{
                padding-left: var(--gap) !important;
                padding-right: var(--gap) !important;
            }
            .right-center{ max-width: 100%; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def open_center():
    st.markdown('<div class="right-center">', unsafe_allow_html=True)


def close_center():
    st.markdown("</div>", unsafe_allow_html=True)


def img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# =========================================================
# 2) Local storage
# =========================================================
ENROLL_ROOT = Path("data/enrolled")
DB_PATH = ENROLL_ROOT / "users.db"
ENROLL_ROOT.mkdir(parents=True, exist_ok=True)

ADMIN_KEY = os.getenv("TONGUEID_ADMIN_KEY", "admin")
ADMIN_PASSWORD = os.getenv("TONGUEID_ADMIN_PASSWORD", "admin123")


# =========================================================
# 3) Email (SMTP)
# =========================================================
SMTP_HOST = os.getenv("TONGUEID_SMTP_HOST", "")
SMTP_PORT = int(os.getenv("TONGUEID_SMTP_PORT", "587"))
SMTP_USER = os.getenv("TONGUEID_SMTP_USER", "")
SMTP_PASS = os.getenv("TONGUEID_SMTP_PASS", "")


def send_email(to_email: str, subject: str, body: str) -> None:
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS):
        raise RuntimeError("SMTP env vars not set. Set TONGUEID_SMTP_HOST/USER/PASS (and PORT).")

    msg = EmailMessage()
    msg["From"] = SMTP_USER
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)


# =========================================================
# 4) Password hashing + time helpers
# =========================================================
def hash_password(password: str, salt: bytes | None = None, iterations: int = 200_000) -> Tuple[bytes, bytes]:
    if salt is None:
        salt = os.urandom(16)
    pw_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, dklen=32)
    return salt, pw_hash


def verify_password(password: str, salt: bytes, expected_hash: bytes) -> bool:
    test_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000, dklen=32)
    return hmac.compare_digest(test_hash, expected_hash)


def _now_ts() -> int:
    return int(time.time())


def _parse_ts(s: Optional[str]) -> int:
    try:
        return int(s) if s else 0
    except Exception:
        return 0


# =========================================================
# 5) DB helpers (users + audit logs + lockouts + templates)
# =========================================================
def db_connect():
    con = sqlite3.connect(DB_PATH)

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            unique_key TEXT PRIMARY KEY,
            username   TEXT NOT NULL,
            email      TEXT NOT NULL DEFAULT '',
            salt       BLOB NOT NULL,
            pw_hash    BLOB NOT NULL,
            created_at TEXT NOT NULL,
            role       TEXT NOT NULL DEFAULT 'user',

            reset_code_hash BLOB,
            reset_code_salt BLOB,
            reset_expires_at TEXT,
            reset_tries INTEGER NOT NULL DEFAULT 0,

            failed_login_count INTEGER NOT NULL DEFAULT 0,
            lock_until_ts INTEGER NOT NULL DEFAULT 0
        );
        """
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            event_type TEXT NOT NULL,
            actor_key TEXT,
            target_key TEXT,
            details TEXT
        );
        """
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS templates (
            unique_key TEXT NOT NULL,
            mode TEXT NOT NULL,
            template BLOB NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (unique_key, mode)
        );
        """
    )

    cols = [r[1] for r in con.execute("PRAGMA table_info(users);").fetchall()]

    def _add(colname: str, ddl: str):
        if colname not in cols:
            con.execute(ddl)

    _add("role", "ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user';")
    _add("email", "ALTER TABLE users ADD COLUMN email TEXT NOT NULL DEFAULT '';")
    _add("reset_code_hash", "ALTER TABLE users ADD COLUMN reset_code_hash BLOB;")
    _add("reset_code_salt", "ALTER TABLE users ADD COLUMN reset_code_salt BLOB;")
    _add("reset_expires_at", "ALTER TABLE users ADD COLUMN reset_expires_at TEXT;")
    _add("reset_tries", "ALTER TABLE users ADD COLUMN reset_tries INTEGER NOT NULL DEFAULT 0;")
    _add("failed_login_count", "ALTER TABLE users ADD COLUMN failed_login_count INTEGER NOT NULL DEFAULT 0;")
    _add("lock_until_ts", "ALTER TABLE users ADD COLUMN lock_until_ts INTEGER NOT NULL DEFAULT 0;")

    return con


def log_event(event_type: str, actor_key: str | None = None, target_key: str | None = None, details: str = "") -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with db_connect() as con:
        con.execute(
            "INSERT INTO audit_logs(ts, event_type, actor_key, target_key, details) VALUES(?,?,?,?,?)",
            (ts, event_type, actor_key, target_key, details),
        )


def get_audit_logs(limit: int = 200) -> list[tuple]:
    with db_connect() as con:
        return con.execute(
            "SELECT ts, event_type, actor_key, target_key, details FROM audit_logs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()


def user_exists(unique_key: str) -> bool:
    with db_connect() as con:
        return con.execute("SELECT 1 FROM users WHERE unique_key = ?", (unique_key,)).fetchone() is not None


def create_user(unique_key: str, username: str, email: str, password: str, role: str = "user") -> None:
    salt, pw_hash = hash_password(password)
    created_at = time.strftime("%Y-%m-%d %H:%M:%S")
    with db_connect() as con:
        con.execute(
            "INSERT INTO users(unique_key, username, email, salt, pw_hash, created_at, role) VALUES(?,?,?,?,?,?,?)",
            (unique_key, username, email, salt, pw_hash, created_at, role),
        )


def get_user(unique_key: str) -> Optional[Tuple[str, str, bytes, bytes, str, str]]:
    with db_connect() as con:
        row = con.execute(
            "SELECT username, email, salt, pw_hash, created_at, role FROM users WHERE unique_key = ?",
            (unique_key,),
        ).fetchone()
        return row if row else None


def list_users() -> List[Tuple[str, str, str, str, str]]:
    with db_connect() as con:
        return con.execute(
            "SELECT unique_key, username, email, created_at, role FROM users ORDER BY created_at DESC"
        ).fetchall()


def ensure_admin_account():
    if user_exists(ADMIN_KEY):
        with db_connect() as con:
            con.execute("UPDATE users SET role='admin' WHERE unique_key = ?", (ADMIN_KEY,))
        return
    create_user(ADMIN_KEY, "Master Admin", "", ADMIN_PASSWORD, role="admin")


def delete_user(unique_key: str) -> None:
    if unique_key == ADMIN_KEY:
        raise ValueError("Admin account cannot be deleted.")
    with db_connect() as con:
        con.execute("DELETE FROM users WHERE unique_key = ?", (unique_key,))
        con.execute("DELETE FROM templates WHERE unique_key = ?", (unique_key,))
    user_dir = ENROLL_ROOT / unique_key
    if user_dir.exists() and user_dir.is_dir():
        shutil.rmtree(user_dir, ignore_errors=True)


def set_user_password(unique_key: str, new_password: str) -> None:
    salt, pw_hash = hash_password(new_password)
    with db_connect() as con:
        con.execute("UPDATE users SET salt=?, pw_hash=? WHERE unique_key=?", (salt, pw_hash, unique_key))


# ---------- Lockout helpers ----------
def is_locked(unique_key: str) -> tuple[bool, int]:
    with db_connect() as con:
        row = con.execute("SELECT lock_until_ts FROM users WHERE unique_key=?", (unique_key,)).fetchone()
    if not row:
        return False, 0
    lock_until = int(row[0] or 0)
    return (_now_ts() < lock_until), lock_until


def format_lock_remaining(lock_until_ts: int) -> str:
    remaining = max(0, int(lock_until_ts) - _now_ts())
    mins = remaining // 60
    secs = remaining % 60
    return f"{mins} min {secs} sec" if mins > 0 else f"{secs} sec"


def record_login_failure(unique_key: str) -> None:
    with db_connect() as con:
        row = con.execute("SELECT failed_login_count FROM users WHERE unique_key=?", (unique_key,)).fetchone()
        count = int(row[0] or 0) + 1 if row else 1

        lock_until = 0
        if count >= 5:
            lock_until = _now_ts() + 10 * 60  # 10 minutes
            count = 0

        con.execute(
            "UPDATE users SET failed_login_count=?, lock_until_ts=? WHERE unique_key=?",
            (count, lock_until, unique_key),
        )


def record_login_success(unique_key: str) -> None:
    with db_connect() as con:
        con.execute("UPDATE users SET failed_login_count=0, lock_until_ts=0 WHERE unique_key=?", (unique_key,))


# =========================================================
# 5b) Template storage helpers
# =========================================================
def _np_to_blob(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def _blob_to_np(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


def save_template(unique_key: str, mode: str, vec: np.ndarray) -> None:
    updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    blob = _np_to_blob(vec)
    with db_connect() as con:
        con.execute(
            """
            INSERT INTO templates(unique_key, mode, template, updated_at)
            VALUES(?,?,?,?)
            ON CONFLICT(unique_key, mode) DO UPDATE SET
              template=excluded.template,
              updated_at=excluded.updated_at
            """,
            (unique_key, mode, blob, updated_at),
        )


def load_template(unique_key: str, mode: str) -> Optional[np.ndarray]:
    with db_connect() as con:
        row = con.execute(
            "SELECT template FROM templates WHERE unique_key=? AND mode=?",
            (unique_key, mode),
        ).fetchone()
    if not row:
        return None
    return _blob_to_np(row[0])


def build_and_store_templates_for_user(unique_key: str) -> None:
    deep_vec = build_template(unique_key, mode="Deep (ResNet18)")
    save_template(unique_key, "Deep (ResNet18)", deep_vec)

    scaler = build_global_scaler_for_enrolled()
    hand_vec = build_template(unique_key, mode="Handcrafted (scaled)", scaler=scaler)
    save_template(unique_key, "Handcrafted (scaled)", hand_vec)


# =========================================================
# 6) OTP Reset helpers
# =========================================================
def set_reset_code(unique_key: str, code: str, ttl_seconds: int = 600) -> None:
    salt, code_hash = hash_password(code)
    expires_at = str(_now_ts() + ttl_seconds)
    with db_connect() as con:
        con.execute(
            """
            UPDATE users
            SET reset_code_hash=?, reset_code_salt=?, reset_expires_at=?, reset_tries=0
            WHERE unique_key=?
            """,
            (code_hash, salt, expires_at, unique_key),
        )


def verify_reset_code(unique_key: str, code: str) -> bool:
    with db_connect() as con:
        row = con.execute(
            "SELECT reset_code_hash, reset_code_salt, reset_expires_at, reset_tries FROM users WHERE unique_key=?",
            (unique_key,),
        ).fetchone()

    if not row:
        return False

    code_hash, code_salt, expires_at, tries = row
    tries = tries or 0

    if code_hash is None or code_salt is None:
        return False
    if tries >= 5:
        return False
    if _now_ts() > _parse_ts(expires_at):
        return False

    ok = verify_password(code, code_salt, code_hash)

    with db_connect() as con:
        con.execute("UPDATE users SET reset_tries = reset_tries + 1 WHERE unique_key=?", (unique_key,))

    return ok


def clear_reset_code(unique_key: str) -> None:
    with db_connect() as con:
        con.execute(
            """
            UPDATE users
            SET reset_code_hash=NULL, reset_code_salt=NULL, reset_expires_at=NULL, reset_tries=0
            WHERE unique_key=?
            """,
            (unique_key,),
        )


def send_reset_code_flow(unique_key: str) -> bool:
    user = get_user(unique_key)
    if not user:
        st.error("No user found with that Unique Key.")
        return False

    username, email, *_ = user
    if not email or "@" not in email:
        st.error("No valid email on this account. Contact admin.")
        return False

    code = f"{secrets.randbelow(10000):04d}"
    set_reset_code(unique_key, code, ttl_seconds=600)

    try:
        send_email(
            to_email=email,
            subject="LinguaNet password reset code",
            body=(
                f"Hello {username},\n\n"
                f"Your 4-digit password reset code is: {code}\n"
                f"It expires in 10 minutes.\n\n"
                f"- LinguaNet"
            ),
        )
        log_event("RESET_CODE_SENT", actor_key=unique_key, target_key=unique_key, details="otp_sent")
        return True
    except Exception as e:
        log_event("RESET_CODE_SEND_FAIL", actor_key=unique_key, target_key=unique_key, details=str(e))
        st.error(f"Email failed: {e}")
        return False


# =========================================================
# 7) Image helpers
# =========================================================
def save_uploaded_images(unique_key: str, files) -> int:
    user_dir = ENROLL_ROOT / unique_key
    user_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for f in files or []:
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            continue
        out_path = user_dir / f"{int(time.time() * 1000)}_upload_{saved}.png"
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


# =========================================================
# 8) Feature/Embedding templates (vector builders)
# =========================================================
@st.cache_resource
def get_embedder():
    return ResNetEmbedder(EmbedConfig(model_name="resnet18", device="cpu"))


def build_global_scaler_for_enrolled() -> StandardScaler:
    feats = []
    for unique_key, _username, _email, _created, _role in list_users():
        if unique_key == ADMIN_KEY:
            continue
        user_dir = ENROLL_ROOT / unique_key
        for p in iter_images(user_dir):
            img = cv2.imread(str(p))
            if img is None:
                continue
            feats.append(extract_features(img))

    scaler = StandardScaler()
    if not feats:
        scaler.fit(np.zeros((2, 10), dtype=np.float32))
        return scaler

    X = np.vstack(feats)
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


# =========================================================
# 9) Auth + session state
# =========================================================
def login(unique_key: str, password: str) -> bool:
    uk = (unique_key or "").strip()
    if not uk:
        return False

    locked, lock_until = is_locked(uk)
    if locked:
        rem = format_lock_remaining(lock_until)
        log_event("LOGIN_BLOCKED_LOCKED", actor_key=uk, target_key=uk, details=f"locked_until={lock_until};remaining={rem}")
        st.session_state["login_lock_msg"] = f"Too many failed attempts. Try again in {rem}."
        return False

    user = get_user(uk)
    if not user:
        log_event("LOGIN_FAIL_NOUSER", actor_key=uk, target_key=uk, details="user_not_found")
        return False

    username, email, salt, pw_hash, _created_at, role = user

    if not verify_password(password, salt, pw_hash):
        record_login_failure(uk)
        log_event("LOGIN_FAIL_BADPASS", actor_key=uk, target_key=uk, details="bad_password")
        return False

    record_login_success(uk)

    st.session_state["logged_in"] = True
    st.session_state["unique_key"] = uk
    st.session_state["username"] = username
    st.session_state["email"] = email
    st.session_state["role"] = role

    log_event("LOGIN_SUCCESS", actor_key=uk, target_key=uk, details=f"role={role}")
    return True


def logout():
    st.session_state["logged_in"] = False
    st.session_state["unique_key"] = ""
    st.session_state["username"] = ""
    st.session_state["email"] = ""
    st.session_state["role"] = ""


def init_states():
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("unique_key", "")
    st.session_state.setdefault("username", "")
    st.session_state.setdefault("email", "")
    st.session_state.setdefault("role", "")

    st.session_state.setdefault("fp_open", False)
    st.session_state.setdefault("fp_stage", None)
    st.session_state.setdefault("fp_userkey", "")
    st.session_state.setdefault("clear_login_pw", False)


# =========================================================
# 10) APP
# =========================================================
st.set_page_config(page_title=f"{APP_NAME} — Demo", layout="wide")
inject_ui_css()

# Left panel logo
try:
    logo_b64 = img_to_base64("assets/linguanet_logo1.png")
    st.markdown(
        f"""
        <div class="left-panel">
            <img src="data:image/png;base64,{logo_b64}" alt="LinguaNet Logo"/>
        </div>
        """,
        unsafe_allow_html=True,
    )
except Exception:
    pass

ensure_admin_account()
init_states()

open_center()

st.markdown(f"## {APP_NAME}")
st.caption(TAGLINE)
st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

if st.session_state["logged_in"]:
    st.info(
        f"Logged in as **{st.session_state['username']}** "
        f"(`{st.session_state['unique_key']}`) — role: **{st.session_state['role']}**"
    )
    if st.button("Logout", key="btn_logout_top"):
        logout()
        st.rerun()


# =========================================================
# Landing: Login / Register (ONLY when not logged in)
# =========================================================
if not st.session_state["logged_in"]:
    st.markdown("### 🔐 Login / ✨ Register")
    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    tab_login, tab_register = st.tabs(["🔐 Login", "✨ Register (Enroll)"])

    # -------------------------
    # LOGIN TAB
    # -------------------------
    with tab_login:
        st.markdown("#### Login Portal")

        # show lock message (if any)
        msg = st.session_state.pop("login_lock_msg", None)
        if msg:
            st.warning(msg)

        # ✅ clear password safely BEFORE widget is created
        if st.session_state.get("clear_login_pw", False):
            st.session_state["login_password"] = ""
            st.session_state["clear_login_pw"] = False

        with st.form("login_form_main", clear_on_submit=False):
            uk = st.text_input("Unique Key", key="login_unique_key")
            pw = st.text_input("Password", type="password", key="login_password")

            b1, b2 = st.columns([1, 1])
            with b1:
                login_clicked = st.form_submit_button("🚀 Login")
            with b2:
                forgot_clicked = st.form_submit_button("🔑 Forgot?")

        if forgot_clicked:
            st.session_state["fp_open"] = True
            st.session_state["fp_stage"] = "request"
            st.session_state["fp_userkey"] = (uk or "").strip()
            st.rerun()

        if login_clicked:
            ok = login((uk or "").strip(), pw)
            if ok:
                st.success("Login successful ✅")
                st.rerun()
            else:
                st.error("Invalid unique key or password.")
                st.session_state["clear_login_pw"] = True
                st.rerun()

        # Forgot UI (hidden until clicked)
        if st.session_state.get("fp_open", False):
            st.markdown("---")
            st.markdown("### 🔑 Forgot Password")

            fp_key = st.text_input(
                "Enter your Unique Key",
                value=st.session_state.get("fp_userkey", ""),
                key="fp_userkey_input",
            ).strip()

            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("Send 4-digit code to email", key="btn_fp_send"):
                    if not fp_key:
                        st.error("Please enter your Unique Key.")
                    else:
                        ok = send_reset_code_flow(fp_key)
                        if ok:
                            st.success("✅ Code sent to your email.")
                            st.session_state["fp_stage"] = "verify"
                            st.session_state["fp_userkey"] = fp_key
                            st.rerun()

            with c2:
                if st.button("Cancel", key="btn_fp_cancel"):
                    st.session_state["fp_open"] = False
                    st.session_state["fp_stage"] = None
                    st.session_state["fp_userkey"] = ""
                    for k in ["fp_userkey_input", "fp_code", "fp_new1", "fp_new2"]:
                        st.session_state.pop(k, None)
                    st.rerun()

            if st.session_state.get("fp_stage") == "verify":
                st.markdown("#### ✅ Enter code + set new password")
                code_in = st.text_input("4-digit code", key="fp_code")
                new_pw1 = st.text_input("New password", type="password", key="fp_new1")
                new_pw2 = st.text_input("Confirm new password", type="password", key="fp_new2")

                if st.button("Reset Password", key="btn_reset_now"):
                    uk2 = (st.session_state.get("fp_userkey") or "").strip()

                    if not uk2:
                        st.error("Missing user key. Try again.")
                    elif not code_in or len(code_in) != 4 or not code_in.isdigit():
                        st.error("Enter a valid 4-digit code.")
                    elif not new_pw1 or len(new_pw1) < 6:
                        st.error("Password must be at least 6 characters.")
                    elif new_pw1 != new_pw2:
                        st.error("Passwords do not match.")
                    else:
                        if verify_reset_code(uk2, code_in):
                            set_user_password(uk2, new_pw1)
                            clear_reset_code(uk2)
                            log_event("PASSWORD_RESET_SUCCESS", actor_key=uk2, target_key=uk2, details="reset_completed")

                            st.success("✅ Password reset successful. Returning to login...")

                            st.session_state["fp_open"] = False
                            st.session_state["fp_stage"] = None
                            st.session_state["fp_userkey"] = ""
                            for k in ["fp_userkey_input", "fp_code", "fp_new1", "fp_new2"]:
                                st.session_state.pop(k, None)

                            logout()
                            st.rerun()
                        else:
                            log_event("PASSWORD_RESET_FAIL", actor_key=uk2, target_key=uk2, details="invalid_or_expired_code")
                            st.error("Invalid/expired code (or too many tries).")

    # -------------------------
    # REGISTER TAB
    # -------------------------
    with tab_register:
        st.markdown("#### Register / Enroll")

        username = st.text_input("Username (display name)", key="enroll_username")
        email = st.text_input("Email (required)", key="enroll_email")
        unique_key = st.text_input("Unique Key (must be unique)", key="enroll_unique_key")
        password = st.text_input("Password", type="password", key="enroll_password")
        password2 = st.text_input("Confirm Password", type="password", key="enroll_password2")

        st.markdown("##### 📁 Upload Enrollment Images")
        files = st.file_uploader(
            "Upload tongue images (2+).",
            key="enroll_images",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            accept_multiple_files=True,
        )

        st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

        if st.button("✨ Create Account", key="btn_enroll"):
            uk = unique_key.strip()
            un = username.strip()
            em = email.strip()

            if not un:
                st.error("Username is required.")
            elif not em or "@" not in em:
                st.error("Valid email is required.")
            elif not uk:
                st.error("Unique Key is required.")
            elif len(uk) < 4:
                st.error("Unique Key must be at least 4 characters.")
            elif uk == ADMIN_KEY:
                st.error("This Unique Key is reserved for admin. Choose another.")
            elif user_exists(uk):
                st.error("This Unique Key already exists. Choose another.")
            elif not password or len(password) < 6:
                st.error("Password must be at least 6 characters.")
            elif password != password2:
                st.error("Passwords do not match.")
            elif not files or len(files) < 2:
                st.error("Provide at least 2 enrollment images.")
            else:
                try:
                    create_user(uk, un, em, password, role="user")
                    saved_upload = save_uploaded_images(uk, files)

                    # Build/store templates now (Phase 2)
                    try:
                        build_and_store_templates_for_user(uk)
                        log_event("TEMPLATES_BUILT", actor_key=uk, target_key=uk, details="deep+handcrafted")
                    except Exception as te:
                        # Keep enrollment successful even if template build fails
                        st.warning(f"User created, but template build failed: {te}")
                        log_event("TEMPLATE_BUILD_FAIL", actor_key=uk, target_key=uk, details=str(te))

                    log_event("USER_ENROLLED", actor_key=uk, target_key=uk, details=f"email={em}")
                    st.success(f"✅ User created. Saved **{saved_upload}** images.")

                    # Enrollment email (optional)
                    try:
                        send_email(
                            to_email=em,
                            subject="LinguaNet enrollment successful",
                            body=(
                                f"Hello {un},\n\n"
                                f"Your LinguaNet account has been enrolled successfully.\n"
                                f"Unique Key: {uk}\n\n"
                                f"- LinguaNet"
                            ),
                        )
                        st.success("📩 Enrollment email sent.")
                    except Exception as e:
                        st.warning(f"User created, but email failed: {e}")

                except Exception as e:
                    st.error(f"Registration failed: {e}")

    close_center()
    st.stop()


# =========================================================
# After login
# =========================================================
role = st.session_state["role"]

if role == "admin":
    st.markdown("#### 📜 Audit Logs (latest)")
    logs = get_audit_logs(limit=200)

    if not logs:
        st.info("No logs yet.")
    else:
        log_rows = []
        for ts, evt, actor, target, details in logs:
            log_rows.append(
                {
                    "Time": ts,
                    "Event": evt,
                    "Actor": actor or "",
                    "Target": target or "",
                    "Details": details or "",
                }
            )
        st.dataframe(log_rows, use_container_width=True, hide_index=True)

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🛡️ Admin Dashboard")
    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    rows = list_users()
    q = st.text_input("🔎 Search users (unique key / username / email)", key="admin_search").strip().lower()

    filtered = []
    for uk, un, em, created, r in rows:
        if q and (q not in uk.lower() and q not in un.lower() and q not in (em or "").lower()):
            continue
        filtered.append((uk, un, em, created, r))

    st.markdown("#### Users")

    h1, h2, h3, h4, h5, h6 = st.columns([1.6, 1.8, 2.2, 2.2, 1.0, 1.4])
    with h1: st.markdown("**Unique Key**")
    with h2: st.markdown("**Username**")
    with h3: st.markdown("**Email**")
    with h4: st.markdown("**Created At**")
    with h5: st.markdown("**Role**")
    with h6: st.markdown("**Actions**")

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    if not filtered:
        st.info("No users match your search.")
    else:
        for uk, un, em, created, r in filtered:
            c1, c2, c3, c4, c5, c6 = st.columns([1.6, 1.8, 2.2, 2.2, 1.0, 1.4])

            with c1:
                st.code(uk, language=None)
            with c2:
                st.write(un)
            with c3:
                st.write(em or "")
            with c4:
                st.write(created)
            with c5:
                st.write(r)

            with c6:
                if uk == ADMIN_KEY:
                    st.caption("Protected")
                else:
                    confirm_key = f"confirm_delete_{uk}"
                    btn_key = f"btn_delete_{uk}"
                    confirm = st.checkbox("Confirm", key=confirm_key)
                    if st.button("🗑️ Delete", key=btn_key, disabled=not confirm):
                        try:
                            delete_user(uk)
                            log_event(
                                "ADMIN_DELETE_USER",
                                actor_key=st.session_state.get("unique_key"),
                                target_key=uk,
                                details="deleted_user",
                            )
                            st.success(f"Deleted user `{uk}` and removed their enrolled images/templates.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")

            st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 🧪 Verify Any User (Admin)")

    user_choices = [row[0] for row in rows if row[0] != ADMIN_KEY and row[4] == "user"]
    if not user_choices:
        st.warning("No normal users enrolled yet.")
        close_center()
        st.stop()

    selected_user = st.selectbox("Select user", user_choices, key="admin_selected_user")
    mode = st.selectbox("Comparison Mode", ["Deep (ResNet18)", "Handcrafted (scaled)"], key="admin_mode")

    default_thr = 0.90 if mode == "Deep (ResNet18)" else 0.25
    threshold = st.slider("Threshold", 0.0, 1.0, float(default_thr), 0.01, key="admin_threshold")

    admin_uploaded = st.file_uploader(
        "Upload probe image (admin)",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        key="admin_probe_upload",
    )

    if st.button("⚡ Verify as Admin", key="btn_admin_verify"):
        img = read_uploaded_file(admin_uploaded)
        if img is None:
            st.error("Provide a probe image.")
        else:
            scaler = build_global_scaler_for_enrolled() if mode == "Handcrafted (scaled)" else None

            template = load_template(selected_user, mode)
            if template is None:
                st.warning("No stored template found. Building now...")
                template = build_template(selected_user, mode, scaler=scaler)
                save_template(selected_user, mode, template)
                log_event("TEMPLATE_AUTO_BUILT", actor_key=st.session_state.get("unique_key"), target_key=selected_user, details=f"mode={mode}")

            probe = extract_probe_vector(img, mode, scaler=scaler)
            score = cosine_similarity(probe, template)
            decision = "ACCEPT ✅" if score >= threshold else "REJECT ❌"

            log_event(
                "ADMIN_VERIFY",
                actor_key=st.session_state.get("unique_key"),
                target_key=selected_user,
                details=f"score={score:.4f};thr={threshold:.2f};mode={mode}",
            )

            st.markdown("##### ✅ Result")
            st.write(f"**Score:** `{score:.4f}`  |  **Threshold:** `{threshold:.2f}`")
            st.write(f"**Decision:** {decision}")

else:
    st.markdown("### ✅ Verify (Logged-in User)")
    st.caption(f"User: {st.session_state['username']} • Key: `{st.session_state['unique_key']}`")
    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    mode = st.selectbox("Comparison Mode", ["Deep (ResNet18)", "Handcrafted (scaled)"], key="user_mode")
    default_thr = 0.90 if mode == "Deep (ResNet18)" else 0.25
    threshold = st.slider("Threshold", 0.0, 1.0, float(default_thr), 0.01, key="user_threshold")

    uploaded = st.file_uploader(
        "Upload probe image (user)",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        key="user_probe_upload",
    )

    if st.button("⚡ Verify", key="btn_user_verify"):
        img = read_uploaded_file(uploaded)
        if img is None:
            st.error("Provide a probe image.")
        else:
            uk = st.session_state["unique_key"]
            scaler = build_global_scaler_for_enrolled() if mode == "Handcrafted (scaled)" else None

            template = load_template(uk, mode)
            if template is None:
                st.warning("No stored template found. Building now...")
                template = build_template(uk, mode, scaler=scaler)
                save_template(uk, mode, template)
                log_event("TEMPLATE_AUTO_BUILT", actor_key=uk, target_key=uk, details=f"mode={mode}")

            probe = extract_probe_vector(img, mode, scaler=scaler)
            score = cosine_similarity(probe, template)
            decision = "ACCEPT ✅" if score >= threshold else "REJECT ❌"

            log_event("USER_VERIFY", actor_key=uk, target_key=uk, details=f"score={score:.4f};thr={threshold:.2f};mode={mode}")
            st.markdown("##### ✅ Result")
            st.write(f"**Score:** `{score:.4f}`  |  **Threshold:** `{threshold:.2f}`")
            st.write(f"**Decision:** {decision}")

close_center()