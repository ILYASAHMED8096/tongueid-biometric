# TongueID Biometric (Python)

TongueID is a Python-based biometric prototype using **tongue-print** imagery.  
It builds an end-to-end pipeline for **tongue ROI extraction**, **feature engineering**, and **ML-based classification**, with a public dataset workflow.

> ⚠️ This repository does **not** include biometric image data.  
> Place datasets locally under `data/` (ignored by git).

---

## What This Project Does

### 1) Tongue ROI Extraction (Segmentation → Crop)
- Uses tongue segmentation masks (public dataset) to crop a clean **Region of Interest (ROI)**.
- Saves ROI images for downstream modeling.

### 2) Feature Engineering
Extracts handcrafted features from ROI:
- **Color**: HSV histogram
- **Texture**: LBP (Local Binary Patterns), GLCM properties
- **Quality**: blur score (Laplacian variance)

### 3) ML Baseline (Identification Prototype)
- Trains an **SVM** classifier on engineered features.
- Prints confusion matrix + classification report.
- Saves the trained model locally (ignored by git).

---

## Dataset Notes (Public-Only Workflow)

Many public tongue datasets provide **segmentation masks** but not person identity labels.
To demonstrate an identification/verification-style pipeline without private images, this project supports a **pseudo-enrollment** setup:

- Create ROI images from (image + mask)
- Generate multiple augmented “captures” per ROI image
- Treat each ROI source image as a pseudo “person” class

---

## Project Structure

ongueid-biometric/
tongueid/ # core pipeline (features + training)
scripts/ # dataset preparation scripts
notebooks/ # experiments (optional)
app/ # Streamlit app (optional)
data/ # ignored (datasets go here)
models/ # ignored (trained models saved here)
reports/ # metrics / notes (optional)
README.md
requirements.txt
.gitignore

## Setup

### 1) Create venv & install
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\activate
pip install -r requirements.txt
## Setup

### 1) Create venv & install
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\activate
pip install -r requirements.txt
) Create pseudo-ID dataset (public-only biometric demo)
python scripts/make_pseudo_id_dataset.py


This writes:

data/processed/person_01/
data/processed/person_02/
...

D) Train SVM baseline
python -m tongueid.train --data data/processed --out models/svm.joblib
Results (Baseline)

After training, the script prints:

Confusion matrix

Precision / Recall / F1 per class

Next planned additions: ROC/AUC + FAR/FRR + EER (verification-style evaluation)

Privacy & Ethics

No biometric images are stored in this repo.

Keep all datasets in data/ locally (gitignored).

Use public datasets according to their licensing terms.
Roadmap

Add verification mode (cosine similarity + threshold)

Add ROC/AUC + FAR/FRR + EER metrics

Add Streamlit enroll/verify demo

✅ Save `README.md`.

---

# 2) Check your `.gitignore` (super important before pushing)
Open `.gitignore` and ensure it includes at least:

✅ Save `README.md`.

---

# 2) Check your `.gitignore` (super important before pushing)
Open `.gitignore` and ensure it includes at least:
data/
models/
.venv/
pycache/
.ipynb_checkpoints/
*.joblib
*.pkl

If any are missing, add them and save.

---

# 3) Now push to GitHub (safe way)

Open VS Code terminal in your project folder and run:

### Step 1 — See what will be committed
```powershell
git status
