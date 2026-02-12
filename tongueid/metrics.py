from __future__ import annotations
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def far_frr(scores_genuine: np.ndarray, scores_impostor: np.ndarray, threshold: float) -> tuple[float, float]:
    """
    Accept if score >= threshold.
    FAR = fraction of impostor accepted.
    FRR = fraction of genuine rejected.
    """
    far = float(np.mean(scores_impostor >= threshold))
    frr = float(np.mean(scores_genuine < threshold))
    return far, frr


def find_eer(scores_genuine: np.ndarray, scores_impostor: np.ndarray) -> tuple[float, float, float]:
    """
    Finds threshold where FAR and FRR are closest (approx EER).
    Returns: (eer, threshold, diff)
    """
    all_scores = np.unique(np.concatenate([scores_genuine, scores_impostor]))
    best = (1.0, float(all_scores[0]), 1.0)  # (eer, thr, diff)

    for thr in all_scores:
        far, frr = far_frr(scores_genuine, scores_impostor, float(thr))
        diff = abs(far - frr)
        eer = (far + frr) / 2.0
        if diff < best[2]:
            best = (eer, float(thr), diff)

    return best
    