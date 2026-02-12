from __future__ import annotations
import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def blur_score(img_bgr: np.ndarray) -> float:
    gray = _to_gray(img_bgr)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def hsv_hist(img_bgr: np.ndarray, bins=(16, 16, 16)) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


def lbp_hist(img_bgr: np.ndarray, radius: int = 2, n_points: int = 16) -> np.ndarray:
    gray = _to_gray(img_bgr)
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def glcm_props(img_bgr: np.ndarray, distances=(1, 2), angles=(0, np.pi/4, np.pi/2)) -> np.ndarray:
    gray = _to_gray(img_bgr)
    gray_q = (gray / 16).astype(np.uint8)  # quantize to 16 levels for stability
    glcm = graycomatrix(gray_q, distances=distances, angles=angles, levels=16, symmetric=True, normed=True)

    props = []
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        v = graycoprops(glcm, prop)
        props.append(v.flatten())
    feat = np.concatenate(props).astype(np.float32)
    return feat


def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    f1 = hsv_hist(img_bgr)
    f2 = lbp_hist(img_bgr)
    f3 = glcm_props(img_bgr)
    f4 = np.array([blur_score(img_bgr)], dtype=np.float32)
    return np.concatenate([f1, f2, f3, f4])
