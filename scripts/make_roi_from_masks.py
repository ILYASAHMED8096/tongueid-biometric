from pathlib import Path
import cv2
import numpy as np

IMG_DIR = Path("data/seg/images")
MSK_DIR = Path("data/seg/masks")
OUT_DIR = Path("data/processed/biohit_roi")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def find_mask_for_image(img_path: Path) -> Path | None:
    # Common matching patterns: same stem, different ext
    candidates = list(MSK_DIR.glob(img_path.stem + ".*"))
    return candidates[0] if candidates else None

def crop_by_mask(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # ensure binary mask
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0 or len(ys) == 0:
        return img_bgr  # fallback if mask empty

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    # small padding
    pad = 10
    h, w = img_bgr.shape[:2]
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(w - 1, x1 + pad); y1 = min(h - 1, y1 + pad)

    roi = img_bgr[y0:y1+1, x0:x1+1]
    return roi

def main():
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = [p for p in IMG_DIR.iterdir() if p.suffix.lower() in exts]

    if not imgs:
        raise SystemExit(f"No images found in {IMG_DIR} (did you move them there?)")

    saved = 0
    for img_path in imgs:
        mask_path = find_mask_for_image(img_path)
        if mask_path is None:
            print(f"[skip] no mask for {img_path.name}")
            continue

        img = cv2.imread(str(img_path))
        msk = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if img is None or msk is None:
            print(f"[skip] unreadable {img_path.name}")
            continue

        roi = crop_by_mask(img, msk)
        out_path = OUT_DIR / f"{img_path.stem}.png"
        cv2.imwrite(str(out_path), roi)
        saved += 1

    print(f"Saved {saved} ROI images to {OUT_DIR}")

if __name__ == "__main__":
    main()
