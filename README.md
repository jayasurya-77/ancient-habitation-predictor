# Ancient Human Habitation Predictor

A **Streamlit** web application that combines two pretrained models:

- **CatBoost binary classifier** (`catboost_model.cbm`) – predicts whether a location hosted ancient human settlement based on geographic and environmental features.
- **TensorFlow/Keras MobileNetV2 image classifier** (`pollen_model.keras`) – identifies the dominant pollen class from a user‑uploaded image.

Both models were trained on **synthetic data** for research prototyping. Their outputs are **not calibrated** to real‑world probabilities and should be interpreted alongside archaeological evidence.

## Repository structure
```
ancient-habitation-webapp/
├─ app.py                 # Streamlit entry point (multi‑page layout)
├─ requirements.txt       # Pinned package versions
├─ pollen_class_mapping.json
├─ catboost_model.cbm     # Binary CatBoost model (Git LFS recommended)
├─ pollen_model.keras     # TensorFlow/Keras model (Git LFS recommended)
├─ README.md              # This file
├─ .gitignore            # Ignored files
└─ setup.sh               # Optional local setup script
```

## Local development
```bash
# 1️⃣ Clone the repository (private GitHub repo required)
git clone <your‑repo‑url>
cd ancient-habitation-webapp

# 2️⃣ Install dependencies (Python 3.12+ recommended)
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3️⃣ Run the app
streamlit run app.py
```
The app will be available at `http://localhost:8501`.

## Deployment on Streamlit Community Cloud (private)
1. Push the repository to a **private** GitHub repo (enable Git LFS for the `.cbm` and `.keras` files).
2. In Streamlit Community Cloud, click **New app**, select the repo and branch, and set the **Main file path** to `app.py`.
3. The platform will automatically install the packages from `requirements.txt` and launch the app.
4. Ensure the repository remains **private** – the app will be accessible only to collaborators you grant access to.

## Important notes
- **No retraining** – the model files are used exactly as provided.
- **Synthetic training data** – the CatBoost model's class probabilities are illustrative only.
- **Model paths** – all paths are relative to `app.py`, so the app works on any host without external dependencies.
- **Git LFS** – because the model binaries exceed GitHub's normal file‑size limits, store them via Git Large‑File Storage.

Feel free to open an issue if you encounter any problems!
