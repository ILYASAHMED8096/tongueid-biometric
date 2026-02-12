from pathlib import Path
import cv2
import numpy as np
import random

SRC_DIR = Path("data/processed/biohit_roi")
OUT_ROOT = Path("data/processed")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

NUM_USERS = 30          # number of pseudo people/classes
SAMPLES_PER_USER = 20   # augmented samples per user

def augment(img):
    out = img.copy()

    # random brightness/contrast
    alpha = random.uniform(0.9, 1.1)  # contrast
    beta  = random.randint(-15, 15)   # brightness
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    # small rotation
    h, w = out.shape[:2]
    angle = random.uniform(-8, 8)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # slight blur sometimes
    if random.random() < 0.3:
        out = cv2.GaussianBlur(out, (3, 3), 0)

    return out

def main():
    imgs = list(SRC_DIR.glob("*.png"))
    if len(imgs) < NUM_USERS:
        raise SystemExit(f"Need at least {NUM_USERS} ROI images in {SRC_DIR}, found {len(imgs)}")

    chosen = imgs[:NUM_USERS]

    for i, p in enumerate(chosen, start=1):
        user_dir = OUT_ROOT / f"person_{i:02d}"
        user_dir.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(p))
        if img is None:
            continue

        # save one “original”
        cv2.imwrite(str(user_dir / "00.png"), img)

        for k in range(1, SAMPLES_PER_USER):
            aug = augment(img)
            cv2.imwrite(str(user_dir / f"{k:02d}.png"), aug)

    print(f"Created pseudo-ID dataset in {OUT_ROOT} (person_XX folders).")

if __name__ == "__main__":
    main()
