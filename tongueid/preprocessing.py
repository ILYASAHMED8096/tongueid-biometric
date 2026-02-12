from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    target_size: tuple[int, int] = (256, 256)
    clahe_clip_limit: float = 2.0
    clahe_grid: tuple[int, int] = (8, 8)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def apply_clahe_bgr(img_bgr: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    # CLAHE on L channel in LAB to improve contrast
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit, tileGridSize=cfg.clahe_grid)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def resize_keep_aspect(img_bgr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    th, tw = target_hw
    h, w = img_bgr.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    # pad to target
    out = np.zeros((th, tw, 3), dtype=resized.dtype)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def preprocess_image(img_bgr: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    img = apply_clahe_bgr(img_bgr, cfg)
    img = resize_keep_aspect(img, cfg.target_size)
    return img


def batch_preprocess(input_dir: Path, output_dir: Path, cfg: PreprocessConfig) -> int:
    ensure_dir(output_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in input_dir.rglob("*") if p.suffix.lower() in exts]

    count = 0
    for p in files:
        img = load_image_bgr(p)
        out = preprocess_image(img, cfg)
        out_path = output_dir / f"{p.stem}.png"
        cv2.imwrite(str(out_path), out)
        count += 1
    return count
