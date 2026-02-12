from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import cv2

from tongueid.features import extract_features
from tongueid.metrics import cosine_similarity, find_eer


def iter_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in folder.glob("*") if p.suffix.lower() in exts]


def load_user_features(data_root: Path) -> dict[str, np.ndarray]:
    """
    data_root/
      person_01/*.png
      person_02/*.png
    Returns dict label -> feature matrix (n_samples, d)
    """
    users = {}
    for d in sorted(data_root.iterdir()):
        if not d.is_dir():
            continue
        feats = []
        for img_path in iter_images(d):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            feats.append(extract_features(img))
        if feats:
            users[d.name] = np.vstack(feats)
    if not users:
        raise RuntimeError(f"No user folders with images found in {data_root}")
    return users


def template_from_features(X: np.ndarray) -> np.ndarray:
    # average template (simple & effective baseline)
    return X.mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/processed", help="processed dataset root with person_XX folders")
    ap.add_argument("--per_user_probe", type=int, default=5, help="probe samples per user for evaluation")
    args = ap.parse_args()

    data_root = Path(args.data)
    users = load_user_features(data_root)

    rng = np.random.default_rng(42)

    genuine_scores = []
    impostor_scores = []

    labels = sorted(users.keys())

    for label in labels:
        X = users[label]
        n = X.shape[0]
        if n < args.per_user_probe + 2:
            # need enough samples to split into template + probes
            continue

        idx = rng.permutation(n)
        probe_idx = idx[:args.per_user_probe]
        enroll_idx = idx[args.per_user_probe:]

        template = template_from_features(X[enroll_idx])

        # genuine: probes from same user
        for i in probe_idx:
            score = cosine_similarity(X[i], template)
            genuine_scores.append(score)

        # impostor: sample probes from other users against this template
        other_labels = [l for l in labels if l != label]
        rng.shuffle(other_labels)
        for other in other_labels[: min(5, len(other_labels))]:
            Xo = users[other]
            j = int(rng.integers(0, Xo.shape[0]))
            score = cosine_similarity(Xo[j], template)
            impostor_scores.append(score)

    genuine_scores = np.array(genuine_scores, dtype=np.float32)
    impostor_scores = np.array(impostor_scores, dtype=np.float32)

    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        raise RuntimeError("Not enough data to compute verification metrics. Increase samples or users.")

    eer, thr, diff = find_eer(genuine_scores, impostor_scores)

    print(f"Genuine scores:  n={len(genuine_scores)}  mean={genuine_scores.mean():.4f}  std={genuine_scores.std():.4f}")
    print(f"Impostor scores: n={len(impostor_scores)}  mean={impostor_scores.mean():.4f}  std={impostor_scores.std():.4f}")
    print(f"\nApprox EER: {eer*100:.2f}% at threshold={thr:.4f} (|FAR-FRR|={diff:.4f})")


if __name__ == "__main__":
    main()
