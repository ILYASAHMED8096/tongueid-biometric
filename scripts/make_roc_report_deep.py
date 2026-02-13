from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from tongueid.metrics import cosine_similarity, find_eer
from tongueid.embeddings import ResNetEmbedder, EmbedConfig

DATA_ROOT = Path("data/processed")
OUT_PNG = Path("reports/figures/verification_roc_deep.png")


def iter_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in folder.glob("*") if p.suffix.lower() in exts]


def load_user_embeddings(data_root: Path, embedder: ResNetEmbedder) -> dict[str, np.ndarray]:
    users = {}
    for d in sorted(data_root.iterdir()):
        if not d.is_dir() or not d.name.startswith("person_"):
            continue

        embs = []
        for img_path in iter_images(d):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            embs.append(embedder.embed(img))

        if embs:
            users[d.name] = np.vstack(embs)

    if not users:
        raise RuntimeError(f"No users found in {data_root}")
    return users


def main():
    cfg = EmbedConfig(model_name="resnet18", device="cpu")
    embedder = ResNetEmbedder(cfg)

    users = load_user_embeddings(DATA_ROOT, embedder)
    labels = sorted(users.keys())
    rng = np.random.default_rng(42)

    genuine_scores = []
    impostor_scores = []

    for label in labels:
        X = users[label]
        if X.shape[0] < 10:
            continue

        idx = rng.permutation(X.shape[0])
        probe_idx = idx[:5]
        enroll_idx = idx[5:]

        template = X[enroll_idx].mean(axis=0)
        template = template / (np.linalg.norm(template) + 1e-12)

        # genuine
        for i in probe_idx:
            genuine_scores.append(cosine_similarity(X[i], template))

        # impostor
        other = [l for l in labels if l != label]
        rng.shuffle(other)
        for l2 in other[:min(10, len(other))]:
            X2 = users[l2]
            j = int(rng.integers(0, X2.shape[0]))
            impostor_scores.append(cosine_similarity(X2[j], template))

    genuine_scores = np.array(genuine_scores, dtype=np.float32)
    impostor_scores = np.array(impostor_scores, dtype=np.float32)

    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        raise RuntimeError("Not enough scores. Increase users or samples per user.")

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
    plt.title(f"Deep Embeddings ROC (EER={eer*100:.2f}% @ thr={thr:.3f})")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved deep ROC plot -> {OUT_PNG.resolve()}")
    print(f"EER (deep): {eer*100:.2f}%  threshold={thr:.3f}")


if __name__ == "__main__":
    main()
