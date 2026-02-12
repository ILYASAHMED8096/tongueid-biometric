from __future__ import annotations
from pathlib import Path
import argparse
import joblib
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from tongueid.features import extract_features


def iter_images(data_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for p in data_dir.rglob("*"):
        if p.suffix.lower() in exts:
            yield p


def load_dataset(root: Path):
    """
    Expected structure:
      data/processed/<person_id>/*.png
    """
    X, y = [], []
    for person_dir in root.iterdir():
        if not person_dir.is_dir():
            continue
        label = person_dir.name
        for img_path in iter_images(person_dir):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            feat = extract_features(img)
            X.append(feat)
            y.append(label)
    if not X:
        raise RuntimeError(f"No images found under: {root}")
    return np.vstack(X), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/processed", help="Processed data folder with per-person subfolders")
    ap.add_argument("--out", type=str, default="models/svm.joblib", help="Output model path")
    args = ap.parse_args()

    data_root = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(data_root)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nReport:\n", classification_report(y_test, y_pred))

    joblib.dump(model, out_path)
    print(f"\nSaved model -> {out_path}")


if __name__ == "__main__":
    main()
