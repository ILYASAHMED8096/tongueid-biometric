# TongueID Biometric (Python)

TongueID is a **tongue-print biometric prototype** built in Python. It demonstrates an end-to-end pipeline using a **public tongue segmentation dataset workflow**:

- Tongue ROI extraction using segmentation masks (crop to the tongue region)
- Feature engineering (color + texture + quality)
- ML baseline training (SVM identification)
- Biometric verification evaluation (cosine similarity + FAR/FRR + EER)
- ROC curve reporting (handcrafted + deep embeddings)
- Streamlit demo for interactive verification

> ⚠️ This repository does **not** store or publish biometric images.  
> Put datasets locally under `data/` (gitignored).

---

## Tech Stack
- Python 3.10+
- OpenCV, NumPy
- scikit-image (LBP, GLCM)
- scikit-learn (SVM, ROC/AUC)
- Streamlit (demo)
- PyTorch + TorchVision (ResNet deep embeddings baseline)

---

## Project Structure

tongueid-biometric/
tongueid/ # core pipeline (features, training, verification, deep embeddings)
scripts/ # dataset preparation + ROC scripts
app/ # Streamlit demo
notebooks/ # experiments (optional)
reports/
figures/ # ROC plots
data/ # ignored (datasets go here)
models/ # ignored (trained models saved here)
README.md
requirements.txt
.gitignore
pyproject.toml


---

## Setup (Windows PowerShell)

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
If you are adding deep embeddings:

pip install torch torchvision
pip freeze > requirements.txt
Dataset Workflow (Public-Only)
Most public tongue datasets provide segmentation masks but do not provide person-identity labels for true biometric identification.

This project supports:

ROI extraction from (image + mask)

A pseudo-enrollment dataset for public-only demonstration:

Choose N ROI images

Generate multiple augmented “captures” per ROI image

Treat each ROI source image as a pseudo “person” class

Note: Pseudo-enrollment can inflate performance versus real multi-session biometric data.

How to Run
A) Place dataset locally (ignored by git)
Recommended local layout:

data/seg/images/   # tongue images
data/seg/masks/    # segmentation masks (binary masks)
B) Make ROI crops (mask → tongue ROI)
python scripts/make_roi_from_masks.py
Output:

data/processed/biohit_roi/
C) Create pseudo-ID dataset (public-only biometric demo)
python scripts/make_pseudo_id_dataset.py
Output (example):

data/processed/person_01/
data/processed/person_02/
...
D) Train SVM baseline (Identification)
python -m tongueid.train --data data/processed --out models/svm.joblib
This prints:

Confusion matrix

Precision/Recall/F1 per class
Model saved locally to models/ (gitignored).

Verification (Biometric Evaluation)
Handcrafted verification (engineered features)
python -m tongueid.verify --data data/processed
Latest run (pseudo-ID dataset):

Genuine mean ≈ 0.9982

Impostor mean ≈ 0.9977

Approx EER ≈ 41.94% @ threshold ≈ 0.9999

Interpretation:

Handcrafted features are not separating classes well under this setup (genuine/impostor scores overlap).
Handcrafted (scaled): EER ≈ 8.67% @ thr ≈ 0.239

Deep embeddings: EER ≈ 0.00% @ thr ≈ 0.965 (pseudo-ID setup)
ROC Reports
Handcrafted ROC plot
python scripts/make_roc_report.py
Output:

reports/figures/verification_roc.png

Deep embeddings ROC plot (ResNet18)
python scripts/make_roc_report_deep.py
Output:

reports/figures/verification_roc_deep.png

Example deep baseline result (pseudo-ID setup):

AUC = 1.000

EER = 0.00% (can be inflated due to pseudo-enrollment)

Streamlit Demo (Enroll + Verify)
Run:

streamlit run app/streamlit_app.py
The app:

selects a claimed identity (person_XX)

computes cosine similarity between probe and user template

accepts/rejects based on threshold

Privacy & Ethics
This repo does NOT include biometric images.

Keep datasets under data/ locally (gitignored).

Use public datasets according to their licensing terms.

Roadmap / Next Improvements
Add a stricter evaluation split: enroll/ vs probe/ folders per user

Add stronger augmentations to reduce “easy” pseudo-ID leakage

Add Streamlit toggle: Handcrafted vs Deep embeddings

Add PCA + whitening for embeddings and compare EER