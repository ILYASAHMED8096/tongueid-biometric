from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler

from tongueid.features import extract_features
from tongueid.metrics import cosine_similarity, find_eer

DATA_ROOT = Path("data/processed")
OUT_PNG = Path("reports/figures/verification_roc_scaled.png")


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
        raise RuntimeError(f"No users found in {data_root}")
    return users


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-12)


def main():
    users = load_user_features(DATA_ROOT)
    labels = sorted(users.keys())
    rng = np.random.default_rng(42)

    # Build splits + collect enrollment for scaler
    splits = {}
    enroll_all = []
    for label in labels:
        X = users[label]
        if X.shape[0] < 10:
            continue
        idx = rng.permutation(X.shape[0])
        probe_idx = idx[:5]
        enroll_idx = idx[5:]
        splits[label] = (enroll_idx, probe_idx)
        enroll_all.append(X[enroll_idx])

    if not splits:
        raise RuntimeError("Not enough users/samples to compute ROC.")

    enroll_all = np.vstack(enroll_all)
    scaler = StandardScaler()
    scaler.fit(enroll_all)

    genuine_scores = []
    impostor_scores = []

    for label in sorted(splits.keys()):
        X = users[label]
        enroll_idx, probe_idx = splits[label]

        X_enroll = scaler.transform(X[enroll_idx])
        template = l2norm(X_enroll.mean(axis=0))

        # genuine
        for i in probe_idx:
            x_probe = l2norm(scaler.transform(X[i:i+1]).squeeze(0))
            genuine_scores.append(cosine_similarity(x_probe, template))

        # impostor
        other = [l for l in splits.keys() if l != label]
        rng.shuffle(other)
        for l2 in other[:min(10, len(other))]:
            X2 = users[l2]
            j = int(rng.integers(0, X2.shape[0]))
            x_imp = l2norm(scaler.transform(X2[j:j+1]).squeeze(0))
            impostor_scores.append(cosine_similarity(x_imp, template))

    genuine_scores = np.array(genuine_scores, dtype=np.float32)
    impostor_scores = np.array(impostor_scores, dtype=np.float32)

    eer, thr, _ = find_eer(genuine_scores, impostor_scores)

    y_true = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(impostor_scores)])
    y_score = np.concatenate([genuine_scores, impostor_scores])

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.xlabel("False Positive Rate (FAR)")
    plt.ylabel("True Positive Rate (1 - FRR)")
    plt.title(f"Handcrafted ROC (scaled) (EER={eer*100:.2f}% @ thr={thr:.3f})")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")

    print(f"Saved scaled handcrafted ROC -> {OUT_PNG.resolve()}")
    print(f"EER (handcrafted scaled): {eer*100:.2f}%  threshold={thr:.3f}")


if __name__ == "__main__":
    main()
