# Credit Score Classification Pipeline

An end-to-end, locally runnable machine learning pipeline that classifies customers into **Poor**, **Standard**, or **Good** credit score categories. It covers data ingestion, preprocessing, multi-model training with cross-validation, experiment tracking with MLflow, an automated deployment approval gate, and a Streamlit web app for inference.

**Live Demo:** [Open the Streamlit app](https://creditscoreprediction-nadine.streamlit.app/)

## Features

- **Modular pipeline**: ingestion → preprocessing → training → evaluation, each as its own class
- **Leakage-safe preprocessing**: outlier bounds, medians, and loan vocabularies are fitted on the training split only
- **3 models compared**: Random Forest, XGBoost, CatBoost, all with class-imbalance handling
- **Experiment tracking**: parameters, CV metrics, and test metrics logged to MLflow (SQLite backend)
- **Model Registry**: the best model is registered as `CreditScoreBestModel` with test metrics attached as tags
- **Approval gate**: the model is approved for deployment only if it passes all metric thresholds
- **Streamlit app**: interactive UI that returns the predicted class and class probabilities

## Project Structure

```
.
├── data_B.csv              # Raw dataset
├── data_ingestion.py       # Step 1: load & validate raw data
├── preprocessing.py        # Step 2: cleaning, feature engineering, train/test split
├── train.py                # Step 3: training + stratified 5-fold CV + MLflow logging
├── evaluation.py           # Step 4: test-set evaluation & best model selection
├── pipeline.py             # Orchestrator: runs all steps, registers model, approval check
├── app_streamlit.py        # Streamlit inference app
├── exploration.ipynb       # Exploratory data analysis
├── requirements.txt
├── artifacts/              # best_model.pkl (generated)
├── ingested/               # ingested data copy (generated)
└── mlflow.db / mlruns/     # MLflow tracking data (generated)
```

## Pipeline Overview

| Step | Module | What it does |
|------|--------|--------------|
| 1. Ingestion | `data_ingestion.py` | Reads the raw CSV, checks it is not empty and contains the `Credit_Score` target, then saves a copy to `ingested/` |
| 2. Preprocessing | `preprocessing.py` | Cleans noisy values, engineers features, imputes, and performs a stratified 80/20 train/test split |
| 3. Training | `train.py` | Trains 3 models inside a sklearn `Pipeline` with stratified 5-fold CV and logs everything to MLflow |
| 4. Evaluation | `evaluation.py` | Evaluates each model on the held-out test set and selects the best one by CV macro-F1 |
| 5. Registration & Approval | `pipeline.py` | Saves the best model, registers it in MLflow, and runs the threshold checks |

### Preprocessing Highlights

- Strips junk characters from numeric columns (e.g. `5_` → `5`) and converts them to numbers
- Replaces placeholder noise values (`_______`, `_`, `!@9#%8`) with missing values
- Converts `Credit_History_Age` (e.g. "3 Years and 2 Months") to months
- Splits `Payment_Behaviour` into `Spent_Level` and `Payment_Value`
- Parses `Type_of_Loan` into per-loan-type frequency features
- Caps extreme outliers (Q3 + 3×IQR) and caps `Total_EMI_per_month` at the 99th percentile
- Imputes income-related columns using the median per `Occupation`
- Final transformer: median imputation + `RobustScaler` (numeric), one-hot (nominal), ordinal encoding (ordered categories)
- Drops identifiers and non-informative columns (`ID`, `Customer_ID`, `Name`, `SSN`, `Month`, `Num_of_Loan`)

### Models

| Model | Imbalance handling |
|-------|--------------------|
| Random Forest | `class_weight='balanced'` |
| XGBoost | balanced `sample_weight` |
| CatBoost | `auto_class_weights='Balanced'` |

## Results

Evaluated on the held-out 20% test set (target encoding: Poor = 0, Standard = 1, Good = 2).

| Model | CV F1 (macro) | Test Accuracy | Test F1 (macro) | Recall (Poor) | Precision (Good) |
|-------|:-------------:|:-------------:|:---------------:|:-------------:|:----------------:|
| **Random Forest** ⭐ | **0.7147** | **0.7450** | **0.7316** | **0.7472** | **0.6085** |
| XGBoost | 0.6971 | 0.7208 | 0.7131 | 0.7729 | 0.5660 |
| CatBoost | 0.6722 | 0.6842 | 0.6803 | 0.7840 | 0.5282 |

### Deployment Approval Gate

The best model must meet **all** thresholds to be approved:

| Metric | Threshold | Random Forest | Status |
|--------|:---------:|:-------------:|:------:|
| Macro F1 | ≥ 0.65 | 0.7316 | ✅ Pass |
| Recall (Poor) | ≥ 0.70 | 0.7472 | ✅ Pass |
| Precision (Good) | ≥ 0.58 | 0.6085 | ✅ Pass |

> Recall on **Poor** is prioritized to catch risky customers, while precision on **Good** limits the chance of wrongly approving them.

## Getting Started

### Prerequisites

- Python 3.11+

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Run the Pipeline

```bash
python pipeline.py
```

This will ingest the data, train and evaluate all three models, save the best one to `artifacts/best_model.pkl`, register it in the MLflow Model Registry, and print the approval decision.

### Explore Experiments with MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://localhost:5000 to compare runs, metrics, and registered model versions.

### Launch the Streamlit App

```bash
streamlit run app_streamlit.py
```

Fill in the applicant, credit behavior, and loan portfolio tabs, then click **Predict Credit Score** to see the predicted class, confidence, and class probabilities.

## Tech Stack

- **Data & ML**: pandas, NumPy, scikit-learn, XGBoost, CatBoost
- **Experiment tracking**: MLflow
- **Serialization**: joblib
- **App & visualization**: Streamlit, Plotly

## Notes

- All random seeds are fixed (`random_state=42`) for reproducibility.
- `pipeline.py` expects the raw dataset at `data_B.csv` in the project root.
- The Streamlit app requires `artifacts/best_model.pkl`, so run the pipeline first.
