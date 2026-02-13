from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler

from tongueid.features import extract_features
from tongueid.metrics import cosine_similarity, find_eer


def iter_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in folder.glob("*") if p.suffix.lower() in exts]


def load_user_features(data_root: Path) -> dict[str, np.ndarray]:
    users = {}
    for d in sorted(data_root.iterdir()):
        if not d.is_dir() or not d.name.startswith("person_"):
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
        raise RuntimeError(f"No users found under: {data_root}")
    return users


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/processed")
    ap.add_argument("--per_user_probe", type=int, default=5)
    args = ap.parse_args()

    data_root = Path(args.data)
    users = load_user_features(data_root)
    labels = sorted(users.keys())
    rng = np.random.default_rng(42)

    # Build enroll/probe splits first
    splits = {}
    enroll_all = []
    for label in labels:
        X = users[label]
        n = X.shape[0]
        if n < args.per_user_probe + 5:
            continue
        idx = rng.permutation(n)
        probe_idx = idx[:args.per_user_probe]
        enroll_idx = idx[args.per_user_probe:]
        splits[label] = (enroll_idx, probe_idx)
        enroll_all.append(X[enroll_idx])

    if not splits:
        raise RuntimeError("Not enough data per user. Increase SAMPLES_PER_USER or reduce per_user_probe.")

    enroll_all = np.vstack(enroll_all)

    # ✅ Fit a global scaler using ALL enrollment data (this is the main fix)
    scaler = StandardScaler()
    scaler.fit(enroll_all)

    genuine_scores = []
    impostor_scores = []

    for label in sorted(splits.keys()):
        X = users[label]
        enroll_idx, probe_idx = splits[label]

        # Transform and build template in standardized space
        X_enroll = scaler.transform(X[enroll_idx])
        template = X_enroll.mean(axis=0)
        template = l2norm(template)

        # Genuine probes
        for i in probe_idx:
            x_probe = scaler.transform(X[i:i+1]).squeeze(0)
            x_probe = l2norm(x_probe)
            genuine_scores.append(cosine_similarity(x_probe, template))

        # Impostor probes (from other users)
        other = [l for l in splits.keys() if l != label]
        rng.shuffle(other)
        for l2 in other[:min(10, len(other))]:
            X2 = users[l2]
            j = int(rng.integers(0, X2.shape[0]))
            x_imp = scaler.transform(X2[j:j+1]).squeeze(0)
            x_imp = l2norm(x_imp)
            impostor_scores.append(cosine_similarity(x_imp, template))

    genuine_scores = np.array(genuine_scores, dtype=np.float32)
    impostor_scores = np.array(impostor_scores, dtype=np.float32)

    eer, thr, diff = find_eer(genuine_scores, impostor_scores)

    print(f"Genuine scores:  n={len(genuine_scores)}  mean={genuine_scores.mean():.4f}  std={genuine_scores.std():.4f}")
    print(f"Impostor scores: n={len(impostor_scores)}  mean={impostor_scores.mean():.4f}  std={impostor_scores.std():.4f}")
    print(f"\nApprox EER: {eer*100:.2f}% at threshold={thr:.4f} (|FAR-FRR|={diff:.4f})")


if __name__ == "__main__":
    main()
