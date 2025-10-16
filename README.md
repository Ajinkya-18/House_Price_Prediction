
# 🏠 California Housing Price Prediction

This repository contains the code, notebooks, data pointers and trained artifacts used to build regression models that predict median house values for California districts (based on the common California Housing dataset).

This updated README documents the repository structure, implementation details (data processing, feature engineering, model training), how to reproduce results locally, and how to load the produced artifacts for inference.

---

## Key highlights

- Problem: Regression — predict median house value for a geographic block in California.
- Data: California Housing dataset (Kaggle copy or the original from scikit-learn / StatLib).
- Models explored: Linear Regression, k-NN Regressor, Random Forest Regressor, HistGradientBoosting Regressor (best performer in experiments).
- Pipeline: data preprocessing -> feature engineering -> scaling -> model selection/tuning -> evaluation -> model export.

---

## Dataset

- Source / citation: [California Housing Prices (Kaggle)](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
- Notes: The dataset is a Kaggle copy of the original California housing dataset (1990 census). Use this link for citation and download.

---

## Repository layout

Top-level files and folders you should expect in this project (present in this workspace):

- `environment.yml` / `conda-environment.yml` / `requirements.txt` - dependency manifests to create a reproducible environment.
- `data/` - place the dataset CSV (commonly named `housing.csv`) here. The repo does not include large raw datasets by default.
- `notebooks/` - contains the exploratory notebook (e.g., `HousePricePrediction.ipynb`) used to iterate on EDA and modeling.
- `models/` - serialized model artifacts and preprocessor objects (scalers, encoders). Files present in this repository include:

- `HistGradientBoostingRegressor.joblib`
- `RandomForestRegressor.joblib`
- `hgbr_best_model.joblib`
- `RFECV_fitted.joblib` (feature selector)
- `RobustScaler_fitted.joblib` (preprocessor / scaler)
- `app/` - optional demo/deployment app (if present) for quick inference or UI.
- `src/` - source scripts and small modules used to run preprocessing, training and inference in a reproducible way.
- `reports/` - generated reports, figures and experiment notes.

If a file or folder is not present locally, check the notebook or `src/` scripts which reference them to find the expected names and locations.

---

## Implementation details

Below are the main implementation components and the expected behavior (as implemented in the notebooks and scripts):

### Data ingestion

- Load the CSV from `data/housing.csv` (or another path referenced in the notebook).
- Basic checks: missing values, data types, head/tail, and summary statistics.

### Data preprocessing

- Handle missing values (common pattern: fill median values for numerical columns or drop rows depending on strategy).
- Categorical encoding: `ocean_proximity` (or similar) is label-encoded or one-hot encoded depending on pipeline.
- Split data into train / test (e.g., 80/20 or using stratified sampling by discretized `median_income`).

### Feature engineering

- New features created to improve signal:
  - `rooms_per_household` = total_rooms / households
  - `bedrooms_per_room` = total_bedrooms / total_rooms
  - `population_per_household` = population / households

- Feature scaling: numerical features are scaled (examples: `QuantileTransformer`, `StandardScaler` or `RobustScaler`) as part of a scikit-learn `ColumnTransformer` within a pipeline to avoid data leakage.

### Modeling and hyperparameter search

- Models trained and compared include:
  - Linear Regression
  - k-NN Regressor
  - Random Forest Regressor
  - HistGradientBoosting Regressor (best performer in experiments)

- Hyperparameter tuning was performed via `GridSearchCV` or `RandomizedSearchCV` inside a pipeline. Cross-validation folds (e.g., 5-fold) and scoring metric R2 (or negative MSE) were used to select the best model.

### Evaluation

- Primary metrics reported: R2 score and RMSE. Visual checks include scatter plots of predicted vs actual values and residual diagnostics.

### Model artifacts

- Trained models and preprocessors are saved to `models/`. Current artifact names in this repository include:
  - `HistGradientBoostingRegressor.joblib`
  - `RandomForestRegressor.joblib`
  - `hgbr_best_model.joblib`
  - `RFECV_fitted.joblib` (used to select features before scaling)
  - `RobustScaler_fitted.joblib` (used to scale features prior to model input)

When saving artifacts, the project stores both the fitted preprocessing objects and the model so inference code can apply the same transforms. The `src/utils.py` exposes `save_model()` and `load_model()` helpers that use joblib.

Note: `src/training_pipeline.py` currently runs training and prints validation scores. There is a commented `save_model(...)` call in that script — to persist a model from training, either uncomment that line or call `save_model()` from your own wrapper.

---

## How to reproduce (recommended minimal steps)

### Step 1 — Create the environment (Conda example)

```powershell
conda env create -f environment.yml; conda activate house-price-env
```

If you don't use conda, install the Python dependencies directly:

```powershell
pip install -r requirements.txt
```

### Step 2 — Place the dataset

Download the California Housing dataset (Kaggle copy or original) and place the CSV at `data/housing.csv`.

### Step 3 — Run the notebook for an interactive workflow

Open and run the notebook from `notebooks/HousePricePrediction.ipynb` in Jupyter or JupyterLab. The notebook follows these stages:

- data loading and basic EDA
- preprocessing and feature engineering
- cross-validation experiments and hyperparameter tuning
- final model training, evaluation and export

### Step 4 — Or run scripts from `src/` (if present)

If there are command-line scripts in `src/` for training or inference, they typically accept arguments for the data path and output model path. Example (adjust to actual script names):

```powershell
python src/train.py --data data/housing.csv --out models/trained_model.pkl
python src/inference.py --model models/trained_model.pkl --input data/sample_rows.csv
```

Check `src/` to find the exact script names and supported CLI flags.

---

## Loading a saved model for inference (example)

Below is a minimal Python snippet showing how to load a model & preprocessor saved with joblib/pickle and run inference on pandas DataFrame rows.

```python
import joblib
import pandas as pd

# adjust path to your artifact
artifact = "models/trained_model.pkl"
pipeline = joblib.load(artifact)

# `pipeline` expected to include preprocessing + model
sample = pd.read_csv('data/housing_sample.csv')
preds = pipeline.predict(sample)
print(preds[:10])
```

Replace `trained_model.pkl` and sample input path with actual filenames from the `models/` folder.

---

## Notes, assumptions and troubleshooting

- This README is written to reflect the typical layout and implementation used across the project. Some filenames (for scripts and artifacts) are generic; if your checkout contains different names, inspect `notebooks/` and `src/` to find the exact locations.
- If training fails due to missing packages, re-run the environment install step and verify the Python version matches the one used for development (Python 3.8–3.11 recommended).
- If your dataset uses different column names, update the feature-engineering code (rooms/household calculations) accordingly.

---

## Results summary

- In experiments recorded in the notebook, the HistGradientBoosting Regressor produced the best validation R2 and lowest RMSE after feature engineering and proper scaling.

For precise numbers and plots, open `notebooks/HousePricePrediction.ipynb` which contains the training logs, cross-validation results and evaluation visualizations.

---
## Streamlit App Link
- [House Price Predictor](https://housepriceprediction-eyw2ifj2trmnwqeepr45qc.streamlit.app/)

---
## License & Author

Author: Ajinkya Tamhankar

License: See the LICENSE file in the repository root (if present).

