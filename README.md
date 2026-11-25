# Stroke Risk Factors & Clinical Triage App 🩺

**Project #49** — Introduction to Data Science (`25A12104`)  
**Student:** Vaddi Varshitha (`S230617`) — AIML-2A  
**Course Instructor:** Satish C  
**Submission:** November 2026

---

## Problem Statement

> *Impute BMI and smoking attributes, test hypertension/glucose risk via Chi-Square & Logistic Regression, and deploy a Streamlit stroke risk screening app.*

## Dataset

| Field | Detail |
| :--- | :--- |
| **Source** | [Kaggle — Stroke Prediction Dataset (fedesoriano)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) |
| **File** | `healthcare-dataset-stroke-data.csv` |
| **Dimensions** | 5,110 rows × 12 columns |
| **Target** | `stroke` (binary) — 249 positive (4.87%) vs. 4,861 negative (95.13%) |

> Place `healthcare-dataset-stroke-data.csv` in `data/raw/` after downloading. Raw data files are gitignored.

See [data/README.md](data/README.md) for the full 12-column Data Dictionary with clinical annotations.

## Key Findings

- **Age** is the dominant independent risk gradient: OR = 2.02 per 10 years (p < 1e-38).
- **Missing BMI** is a powerful acute-admission proxy: OR = 3.70 (p < 1e-9) — stroke rate is 19.9% when BMI is unrecorded vs. 4.3% when present.
- **Hypertension** (OR = 1.39, p = 0.049) and **average blood glucose** (OR = 1.04 per 10 mg/dL, p = 0.002) are independently significant after multivariable adjustment.
- **`class_weight='balanced'`** outperformed SMOTE for imbalance handling (CV ROC-AUC: 0.849 vs. 0.842).
- **Logistic Regression** (CV ROC-AUC: 0.852, Test ROC-AUC: 0.831) is competitive with Random Forest (0.833) and XGBoost (0.841), while providing full log-odds explainability.
- Calibrated clinical screening at threshold ~0.54 achieves **80% Recall (Sensitivity)** with **75% Specificity** on the held-out test set.

## Repository Architecture

```text
ds-stroke-risk-triage-app/
├── data/
│   ├── README.md                               # Data dictionary & download steps
│   ├── raw/                                     # healthcare-dataset-stroke-data.csv (GITIGNORED)
│   └── processed/                               # Cleaned EDA CSVs (GITIGNORED)
├── notebooks/
│   ├── 01_data_audit_and_cleaning.ipynb
│   ├── 02_eda_and_visualizations.ipynb
│   ├── 03_statistical_hypothesis_testing.ipynb
│   └── 04_modeling_and_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py                                # Paths, seeds, clinical thresholds
│   ├── data_cleaning.py                         # Audit + stratified imputation
│   ├── statistical_tests.py                     # Chi-Square, Mann-Whitney, Logit ORs
│   ├── feature_engineering.py                   # Custom transformers & ColumnTransformer
│   ├── train_models.py                          # 5-model benchmark, tuning & serialization
│   └── app.py                                   # Interactive Streamlit Triage App
├── models/
│   ├── logistic_regression_pipeline.joblib
│   ├── best_model_pipeline.joblib
│   └── model_metrics.json
├── docs/
│   ├── figures/                                 # 300-DPI charts (Figures 1–10)
│   ├── generate_report.py                       # ReportLab PDF builder
│   ├── generate_ppt.py                          # python-pptx slide builder
│   ├── Project_Report_S230617_Vaddi_Varshitha.pdf
│   ├── Presentation_S230617_Vaddi_Varshitha.pptx
│   └── VIVA_PREP.md                             # 25+ Viva defense Q&A
├── .gitignore
├── requirements.txt
├── DEVELOPMENT_STRATEGY.md
├── DEVELOPMENT_INSTRUCTIONS.md
└── README.md                                    # ← You are here
```

## Tech Stack

| Category | Libraries |
| :--- | :--- |
| Data & Statistics | pandas, NumPy, SciPy, Statsmodels |
| Machine Learning | scikit-learn, imbalanced-learn, XGBoost |
| Visualization | Matplotlib, Seaborn, Plotly |
| Web Application | Streamlit |
| Academic Deliverables | ReportLab (PDF), python-pptx (PPTX) |
| Version Control | Git / GitHub (feature branches + PR merges) |

## Setup & Run

```bash
# 1. Clone the repository
git clone https://github.com/RajuGudem26/ds-stroke-risk-triage-app.git
cd ds-stroke-risk-triage-app

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows

# 3. Install dependencies (macOS XGBoost needs: brew install libomp)
pip install -r requirements.txt

# 4. Download dataset from Kaggle
# Place healthcare-dataset-stroke-data.csv in data/raw/

# 5. Run data cleaning & EDA
python -m src.data_cleaning
jupyter nbconvert --execute --inplace notebooks/01_data_audit_and_cleaning.ipynb

# 6. Run statistical tests
python -m src.statistical_tests

# 7. Train models (5-fold CV + hyperparameter tuning)
python -m src.train_models

# 8. Launch the Streamlit Triage App
streamlit run src/app.py

# 9. Generate academic deliverables
PYTHONPATH=. python docs/generate_report.py
PYTHONPATH=. python docs/generate_ppt.py
```

## Git Branching Workflow

| Branch | Modules | Status |
| :--- | :--- | :---: |
| `setup-and-cleaning` | 0, 1 | ✅ Merged |
| `eda-and-stats` | 2, 3 | ✅ Merged |
| `feature-eng-and-modeling` | 4, 5 | ✅ Merged |
| `streamlit-triage-app` | 6 | ✅ Merged |
| `docs-and-presentation` | 7 | ✅ Merged |
| `main` | 8 (Final Audit) | ✅ Complete |

## License

This project is an academic capstone exercise and is provided for educational purposes.

---

*Built with Antigravity • November 2026*
