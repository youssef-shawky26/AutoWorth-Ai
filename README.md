# 🚗 AutoWorth AI — Used Car Price & Deal Advisor

A machine learning project that predicts the fair market price of a used car and tells you whether a listing is a good deal, a fair deal, or overpriced. The full pipeline — from raw brand-by-brand CSVs to a deployed Streamlit app — lives in this repo.

## 📓 Project Structure

```
.
├── Car-Price.ipynb    # End-to-end ML notebook: data → EDA → cleaning → modeling → final pipeline
├── app_v3.py           # Streamlit app (dark "carbon & crimson" theme)
├── model.pkl            # Trained pipeline exported from the notebook (preprocessing + Random Forest)
├── requirements.txt
└── README.md
```

## ✨ What the App Does

- Takes a car's **brand, model, year, mileage, transmission, fuel type, engine size, MPG, and tax band**
- Predicts a **fair market price** using a trained Random Forest pipeline
- Compares that estimate to the **seller's asking price** and returns a verdict:
  - 🟢 **Good Deal** — asking price is below the fair-price estimate
  - 🟡 **Fair Deal** — asking price is up to 10% above fair value
  - 🔴 **Overpriced** — asking price is more than 10% above fair value
- Brand → Model dropdown is cascading, built from the real training data, so invalid combinations can't be selected

## 🧪 The ML Pipeline (`Car-Price.ipynb`)

### 1. Get the Data
Eleven separate UK used-car CSVs (Audi, BMW, Mercedes, Ford, Hyundai, Skoda, Toyota, VW, Vauxhall, plus the Ford Focus and Mercedes C-Class datasets) are each tagged with a `Make` column and concatenated into a single DataFrame — **108,540 rows × 10 columns**.

### 2. EDA & Cleaning
- Dropped duplicate rows → **106,267 rows**
- `tax` and `mpg` were missing only for the Focus and C-Class subsets. Rather than dropping them, missing values were **imputed using grouped medians**:
  - `tax` ← median grouped by `fuelType`, `transmission`, `year`
  - `mpg` ← median grouped by `engineSize`, `year`
  - The handful of rows that still had no group match were dropped
- Outlier check on `year` caught a single corrupted row (`year = 2060`), which was removed
- `mpg` outliers (> 400) were inspected but intentionally kept, since they reflect real high-efficiency/hybrid vehicles rather than data errors

### 3. Skewness & Distributions
`price`, `mileage`, and `mpg` were right-skewed, so a `PowerTransformer` was fit on each for visualization and for training several candidate models on a more Gaussian-shaped target.

### 4. Visualization
Average price by manufacturer, price by transmission/fuel type, and manufacturer market share were plotted to sanity-check the data before modeling.

### 5. Train / Validation / Test Split
70% / 15% / 15% split (`train_test_split`, `random_state=42`).

### 6. Encoding & Scaling Experiments
Several encoding strategies were tried before settling on the final combination:
- **`model`** (100+ distinct values) → **Target Encoding**, since one-hot would have created an enormous sparse matrix
- **`Make`, `fuelType`, `transmission`** → tried Label Encoding and Binary Encoding, but the final pipeline uses **One-Hot Encoding** (`drop='first'`)
- All numeric features → **Robust Scaler**, chosen over standard scaling because of the outliers noted above

### 7. Model Comparison
Five regressors were trained and compared on validation R² / RMSE / MAE:

| Model | Notes |
|---|---|
| Linear Regression | Baseline |
| Decision Tree | Prone to overfitting |
| **Random Forest** | **Best performer** |
| Gradient Boosting | Competitive, slower to tune |
| XGBoost | Competitive |

Random Forest was carried forward for hyperparameter tuning.

### 8. Hyperparameter Tuning
`GridSearchCV` / `RandomizedSearchCV` over `n_estimators`, `max_depth`, `max_features`, `min_samples_split`, and `min_samples_leaf` converged on:

```python
RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    max_features=0.5,
    min_samples_split=5,
    min_samples_leaf=1,
    oob_score=True,
    random_state=42,
)
```

Learning curves and feature-importance plots confirmed the model generalizes well without overfitting, with `year`, `mileage`, and `model` (target-encoded) as the strongest predictors.

### 9. Final Pipeline
For deployment, everything is wrapped into a single `sklearn.Pipeline` so a raw, un-encoded row can be predicted directly:

```python
ColumnTransformer([
    ('target_encoding', TargetEncoder(cols=['model']), ['model']),
    ('ohe_encoding',    OneHotEncoder(drop='first', handle_unknown='ignore'), ['Make', 'fuelType', 'transmission']),
    ('RobustScaler',    RobustScaler(), ['year', 'mileage', 'tax', 'mpg', 'engineSize']),
])
→ RandomForestRegressor(...)
```

The pipeline predicts price directly in £ (the target's `PowerTransformer` used during model comparison is inverted before this final fit, so no inverse-transform is needed downstream). The fitted pipeline is exported with `joblib.dump(final_Pipeline, 'model.pkl')`.

### Final Model Performance

| Split | R² | RMSE | MAE |
|---|---|---|---|
| Train | 0.9683 | 1,743.6 | 1,151.8 |
| Validation | 0.9590 | 1,962.5 | 1,282.4 |
| **Test** | **0.9567** | **2,035.0** | **1,282.5** |

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/autoworth-ai.git
cd autoworth-ai
pip install -r requirements.txt
```

**Requirements:** Python 3.10+

## ▶️ Usage

```bash
streamlit run app_v3.py
```

Open the local URL Streamlit prints, fill in the car's details and the seller's asking price, and click **Predict Market Price**.

To explore or re-run the full modeling process, open `Car-Price.ipynb` in Jupyter — note it expects the original per-brand CSVs in a local `Data/` folder.

## 🚧 Possible Improvements

- Persist the target-price `PowerTransformer` alongside the pipeline so it can be reused for future experiments without retraining from scratch
- Add SHAP-based explanations for individual predictions
- Expand beyond the 11 UK brands/models currently supported
- Deploy to Streamlit Community Cloud or Hugging Face Spaces

## 📄 License

Released for educational purposes — feel free to fork and adapt it.

## 🙋 Author

Youssef Mohamed — Machine Learning Final Project
