# TongueID Biometric (Python)

TongueID is a tongue-print biometric prototype that performs:
- Tongue ROI extraction (preprocessing + optional segmentation)
- Feature engineering (texture/color/shape)
- Identification (1:N) and Verification (1:1)
- Biometric evaluation (ROC/AUC, FAR/FRR, EER)
- Optional Streamlit demo for enroll/verify

## Tech Stack
Python, OpenCV, scikit-image, scikit-learn, Streamlit (optional)

## Project Structure
- `tongueid/` core pipeline code
- `notebooks/` experiments & EDA
- `app/` Streamlit demo
- `data/` (ignored) local tongue images only
- `models/` (ignored) saved models

## Setup
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
