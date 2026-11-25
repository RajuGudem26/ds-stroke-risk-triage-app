# Detailed Development Instructions for Antigravity
**Project #49:** Stroke Risk Factors & Clinical Triage App  
**Student:** Vaddi Varshitha (`S230617`) | **Course:** Introduction to Data Science (`25A12104`)  

> **MANDATORY DIRECTIVE FOR ANTIGRAVITY:**  
> You MUST read this file alongside `DEVELOPMENT_STRATEGY.md` before executing any task. Follow the exact folder architecture, technical methods, statistical rigor, code quality standards, and Post-Module Verification & Self-Healing Protocol defined below.

---

## 1. Repository Architecture & File Manifest

Maintain this exact directory structure:

```text
ds-stroke-risk-triage-app/
├── data/
│   ├── README.md                               # Dataset source link, download steps, & data dictionary
│   ├── raw/                                    # Contains healthcare-dataset-stroke-data.csv (GITIGNORED)
│   └── processed/                              # Contains cleaned EDA/modeling CSVs (GITIGNORED)
├── notebooks/
│   ├── 01_data_audit_and_cleaning.ipynb        # Module 1: Audit, BMI & smoking imputation
│   ├── 02_eda_and_visualizations.ipynb         # Module 2: Univariate, bivariate & multivariate EDA
│   ├── 03_statistical_hypothesis_testing.ipynb # Module 3: Chi-Square, Mann-Whitney/t-test, Logit ORs
│   └── 04_modeling_and_evaluation.ipynb        # Module 4 & 5: Feature eng, 5-model benchmark, error analysis
├── src/
│   ├── __init__.py
│   ├── config.py                               # Paths, random seed (42), clinical thresholds
│   ├── data_cleaning.py                        # Reusable cleaning & imputation helpers
│   ├── statistical_tests.py                    # Chi-Square, Cramér's V, Mann-Whitney, Statsmodels Logit
│   ├── feature_engineering.py                  # Custom transformers & ColumnTransformer pipeline
│   ├── train_models.py                         # End-to-end training, CV, threshold tuning & saving
│   └── app.py                                  # Interactive Streamlit Clinical Triage App
├── models/
│   ├── logistic_regression_pipeline.joblib     # Primary interpretable model pipeline
│   ├── best_model_pipeline.joblib              # Best overall tuned model pipeline
│   └── model_metrics.json                      # Serialized metrics, thresholds & feature metadata
├── docs/
│   ├── figures/                                # 300-DPI charts, tables & Streamlit app screenshots
│   ├── generate_report.py                      # Script to compile formal PDF report (ReportLab/WeasyPrint)
│   ├── generate_ppt.py                         # Script to build 10-12 slide PPTX (python-pptx)
│   ├── Project_Report_S230617_Vaddi_Varshitha.pdf
│   ├── Presentation_S230617_Vaddi_Varshitha.pptx
│   └── VIVA_PREP.md                            # Analytical reasoning & Viva defense guide
├── .gitignore
├── requirements.txt
├── DEVELOPMENT_STRATEGY.md
├── DEVELOPMENT_INSTRUCTIONS.md
└── README.md
```

---

## 2. Exact `.gitignore` & `requirements.txt`

### `.gitignore` (Strictly Enforced per Course Git Guide)
```gitignore
# Environments & Secrets
venv/
.venv/
.env
__pycache__/
*.pyc
.ipynb_checkpoints/

# Data Files (Do NOT push raw/large datasets to GitHub)
*.csv
*.xlsx
*.zip
data/raw/*
data/processed/*
!data/README.md

# OS / IDE artifacts
.DS_Store
Thumbs.db
.vscode/
```

### `requirements.txt`
Include pinned or compatible versions of:
- `pandas`, `numpy`, `scipy`, `statsmodels`
- `scikit-learn`, `imbalanced-learn`, `xgboost`
- `matplotlib`, `seaborn`, `plotly`
- `streamlit`, `joblib`, `jupyter`, `nbconvert`, `ipykernel`
- `python-pptx`, `reportlab` (or `fpdf2` / `weasyprint` for PDF report generation)

---

## 3. Module-by-Module Technical Execution Specifications

### Module 0: GitHub Account, Git Setup & Dataset Acquisition
1. **GitHub Account & Repo Verification:**
   - Prompt the user to confirm they have created their GitHub account (e.g., `vaddi-varshitha-ds`) and created a Public repo named `ds-stroke-risk-triage-app`.
   - Verify local Git config via `git config --global --list`. If unset, run:
     ```bash
     git config --global user.name "Vaddi Varshitha"
     git config --global user.email "<user-github-email>"
     ```
2. **Dataset Check:**
   - Check if `data/raw/healthcare-dataset-stroke-data.csv` exists.
   - If missing, pause and instruct the user to download it from `https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset` (or use Kaggle CLI if configured) and place it in `data/raw/healthcare-dataset-stroke-data.csv`.
   - Write `data/README.md` with the full 12-column Data Dictionary (feature name, type, unit, clinical meaning, known quality issues).

---

### Module 1: Data Auditing & Clinical Imputation (`setup-and-cleaning`)
**Dataset Known Characteristics to Address Explicitly:**
- `id`: Drop (non-predictive identifier).
- `gender`: 1 row has `"Other"`. Drop this single row (0.02% of data) or document why dropping avoids zero-variance dummy issues in Chi-Square/Logit.
- `bmi`: ~201 missing values (~3.93%) and extreme values (> 60, up to 97.6).
  - **Imputation Strategy:** Compare simple median vs. **Age-Band & Gender-Stratified Median Imputation** (and KNN Imputer in modeling pipeline). Add binary flag `bmi_was_missing` (important clinically because missing BMI in hospital records often correlates with emergency/bedridden admissions—check if missing BMI correlates with stroke!).
  - **Outlier Handling:** Analyze BMI > 60 using IQR/clinical bounds. Do NOT blindly delete stroke cases; instead cap/winsorize extreme implausible values (e.g., BMI > 65) or evaluate sensitivity with vs. without capping and document findings.
- `smoking_status`: ~1,544 `"Unknown"` values (~30.2%).
  - **Domain-Aware Strategy:** Inspect age distribution of `"Unknown"`. Almost all pediatric patients (`age < 12`) are marked `"Unknown"`.
  - Rule 1: Create `smoking_was_unknown` indicator flag (preserves clinical missingness signal).
  - Rule 2: Impute `"Unknown"` for `age < 12` as `"never smoked"` (clinically valid).
  - Rule 3: For `age >= 12`, compare retaining `"Unknown"` as a valid category vs. predictive imputation (e.g., Random Forest/mode conditioned on age/gender) and select the approach that yields better CV ROC-AUC/interpretability without introducing bias. Document the exact rationale!

---

### Module 2: Exploratory Data Analysis (`eda-and-stats`)
Generate and save at **300 DPI** inside `docs/figures/` with clear clinical captions:
1. `01_target_imbalance.png`: Stroke class distribution (95.13% vs. 4.87%) with exact counts & percentages.
2. `02_numeric_distributions.png`: Histograms + KDE for `age`, `avg_glucose_level` (highlight bimodal distribution!), and `bmi`.
3. `03_stroke_rate_by_risk_factors.png`: Multi-panel bar chart showing Stroke Rate (%) across `hypertension`, `heart_disease`, `smoking_status`, `ever_married`, and `work_type`.
4. `04_age_glucose_bmi_by_stroke.png`: Violin/Box plots comparing `age`, `avg_glucose_level`, and `bmi` for Stroke=0 vs. Stroke=1.
5. `05_correlation_heatmap.png`: Spearman/Pearson correlation matrix of numeric & binary features.
6. `06_hypertension_glucose_interaction.png`: Multivariate plot showing how elevated glucose + hypertension compounds stroke prevalence across age groups.

---

### Module 3: Statistical Hypothesis Testing (`eda-and-stats`)
Implement `src/statistical_tests.py` and `notebooks/03_statistical_hypothesis_testing.ipynb`:
1. **Chi-Square Tests of Independence ($\alpha = 0.05$):**
   - Test `stroke` against: `hypertension`, `heart_disease`, `glucose_risk_tier` (`<100`, `100-125`, `126-199`, `>=200`), `bmi_category` (WHO: Underweight, Normal, Overweight, Obese), `smoking_status`, `ever_married`, `work_type`, `Residence_type`, and `gender`.
   - Report for each: $\chi^2$ statistic, degrees of freedom, $p$-value, **Cramér's V** effect size, and decision (`Reject H0` / `Fail to Reject H0`).
2. **Continuous Feature Tests:**
   - Check normality (`scipy.stats.shapiro` / `normaltest`) on `age`, `avg_glucose_level`, `bmi`.
   - Since clinical variables are non-normal/skewed, run **Mann-Whitney U test** (primary) alongside **Welch's t-test** (secondary check) comparing Stroke vs. Non-Stroke groups. Report medians, IQRs, means, and $p$-values.
3. **Epidemiological Logistic Regression (`statsmodels.api.Logit`):**
   - Fit multivariable Logistic Regression on standardized/clean clinical predictors.
   - Compute and export a clean **Odds Ratio Table**: Feature, Coefficient ($\beta$), **Odds Ratio ($e^\beta$)**, 95% CI (`[Lower, Upper]`), and $p$-value.
   - Explicitly interpret the Odds Ratios for `hypertension` and `avg_glucose_level` in plain clinical English (directly answering the Problem Statement).
   - Plot `07_odds_ratios_forest_plot.png` (Forest plot of Odds Ratios with 95% CIs).

---

### Module 4: Feature Engineering & Leakage-Free Pipeline (`feature-eng-and-modeling`)
Implement `src/feature_engineering.py`:
1. **Domain-Enriched Clinical Features:**
   - `bmi_category`: WHO thresholds (`<18.5` Underweight, `18.5-24.9` Normal, `25-29.9` Overweight, `>=30` Obese).
   - `glucose_risk_tier`: ADA clinical thresholds (`<100` Normal, `100-125` Prediabetes, `126-199` Hyperglycemia, `>=200` Severe Hyperglycemia).
   - `age_group`: `<18`, `18-44`, `45-64`, `>=65`.
   - `cardiometabolic_risk_count`: Sum of (`hypertension == 1`) + (`heart_disease == 1`) + (`avg_glucose_level >= 140`) + (`bmi >= 30`).
   - `age_x_glucose`: Interaction feature (`age * avg_glucose_level / 100.0`).
2. **Leakage-Free Preprocessing (`ColumnTransformer`):**
   - Split data FIRST: `train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)`.
   - Fit all imputers, scalers (`StandardScaler`), and encoders (`OneHotEncoder(handle_unknown='ignore', drop='first')`) **strictly on `X_train`**.
   - Wrap SMOTE inside `imblearn.pipeline.Pipeline` during Cross-Validation so synthetic minority samples NEVER bleed into validation folds.

---

### Module 5: Modeling, Tuning & Evaluation (`feature-eng-and-modeling`)
Implement `src/train_models.py` and `notebooks/04_modeling_and_evaluation.ipynb`:
1. **Model Benchmark (5-Fold Stratified CV):**
   - **Logistic Regression** (L2 regularized, tuned `C`, `class_weight='balanced'` & SMOTE comparison) — *Required primary model*.
   - **Random Forest Classifier** (tuned `max_depth`, `min_samples_leaf`, `class_weight='balanced'`).
   - **XGBoost Classifier** (tuned `scale_pos_weight`, `max_depth`, `learning_rate`).
   - **K-Nearest Neighbors (KNN)** (with SMOTE + scaled features, tuned `n_neighbors`).
   - **Gaussian Naive Bayes** (fast clinical baseline).
2. **Evaluation Metrics (Imbalance-First Discipline):**
   - Never rely on Accuracy alone (a dummy all-zero classifier gets 95.1% accuracy!).
   - Primary metrics: **ROC-AUC**, **PR-AUC (Average Precision)**, **Recall (Sensitivity)**, **F1-Score**, **Precision**, and **Specificity**.
3. **Clinical Triage Threshold Calibration:**
   - Compute optimal decision thresholds on validation predictions (e.g., Youden's J statistic / target Recall $\ge 0.80$).
   - Define 3 Clinical Triage Risk Bands based on predicted probability:
     - **Low Risk:** `< low_thresh` (e.g., `< 0.25`)
     - **Moderate Risk:** `low_thresh` to `high_thresh` (e.g., `0.25 – 0.55`)
     - **High Risk (Priority Triage):** `>= high_thresh` (e.g., `>= 0.55`)
4. **Save Plots & Artifacts:**
   - `08_model_comparison_roc_pr.png`: Combined ROC and Precision-Recall curves for all 5 models.
   - `09_confusion_matrix_best_model.png`: Confusion matrix at calibrated triage threshold.
   - `10_feature_importance.png`: Top risk drivers from Logistic Regression & best tree model.
   - Save `models/logistic_regression_pipeline.joblib`, `models/best_model_pipeline.joblib`, and `models/model_metrics.json`.

---

### Module 6: Streamlit Stroke Risk Screening App (`streamlit-triage-app`)
Implement `src/app.py`:
- **Page Config:** `st.set_page_config(page_title="Stroke Risk Clinical Triage App", page_icon="🩺", layout="wide")`.
- **Sidebar:** Student & Project Badge (`Project #49 | Vaddi Varshitha | S230617`), Model Selector (`Logistic Regression (Interpretable)` vs. `Best Ensemble Model`), Clinical Sensitivity Threshold Slider, and Medical Disclaimer.
- **Tab 1 — Live Patient Triage Screener:**
  - Clean 2-column clinical input form with sensible defaults and tooltips (Age, Gender, Hypertension, Heart Disease, Avg Glucose Level mg/dL, BMI kg/m², Smoking Status, Ever Married, Work Type, Residence Type).
  - Real-time Plotly Gauge Chart showing Stroke Risk Probability (%).
  - Color-coded Triage Tier Card (Green = Low Risk, Amber = Moderate Risk, Red = High Risk — Immediate Clinical Follow-Up Recommended).
  - Personalized Risk Factor Summary highlighting which of the patient's inputs exceed WHO/ADA/AHA thresholds.
- **Tab 2 — Prediction Explainability:**
  - Horizontal bar chart showing local feature contributions (standardized feature × Logistic Regression coefficient) for the current patient so clinicians/evaluators understand *why* the score was generated.
- **Tab 3 — EDA & Statistical Evidence:**
  - Interactive Plotly charts + Chi-Square & Odds Ratio tables loaded directly from project outputs.
- **Tab 4 — Model Performance & Batch Screening:**
  - Model comparison leaderboard table, ROC/PR curve display, and optional CSV batch triage uploader (with downloadable scored results).

---

### Module 7: Documentation (PDF), Presentation (PPT) & Viva Prep (`docs-and-presentation`)
1. **Official Project Report (`docs/Project_Report_S230617_Vaddi_Varshitha.pdf`):**
   - Build via `docs/generate_report.py` ensuring **exact compliance** with course formatting rules:
     - Font: Times New Roman or Arial (Body 12 pt, Headings 14–16 pt Bold).
     - Line Spacing: 1.5 spacing, justified body text.
     - All tables, charts, and Streamlit screenshots numbered and captioned (e.g., *Figure 1: Target Class Distribution*).
   - Required 10 Sections:
     1. Title Page (Project Title, Vaddi Varshitha, S230617, AIML-2A / Introduction to Data Science 25A12104, Instructor: Satish C)
     2. Abstract (150–200 words)
     3. Problem Statement & Business/Domain Context
     4. Objectives & Research Questions
     5. Dataset Description & Data Dictionary
     6. Tech Stack & Libraries
     7. Methodology & Data Science Pipeline
     8. Results, Visualizations & Comparative Analysis (with Streamlit screenshots)
     9. Conclusion, Limitations & Future Scope
     10. References (Kaggle dataset, WHO/ADA/AHA clinical guidelines, Scikit-learn, Streamlit, Statsmodels)
2. **Presentation (`docs/Presentation_S230617_Vaddi_Varshitha.pptx`):**
   - Build via `docs/generate_ppt.py` with **10–12 clean, visual slides**:
     - Slide 1: Title Slide
     - Slide 2: Problem Statement & Research Questions
     - Slide 3: Dataset Overview, Data Dictionary & Imputation Strategy (BMI & Smoking)
     - Slide 4–5: EDA Visual Insights & Statistical Tests (Chi-Square + Odds Ratios)
     - Slide 6–7: Feature Engineering, Imbalance Handling & Modeling Rationale
     - Slide 8–9: Model Comparison Metrics & Streamlit Triage App Live Demo Screenshots
     - Slide 10: Key Takeaways, Clinical Limitations & Future Scope
3. **Viva Defense Guide (`docs/VIVA_PREP.md`):**
   - Cover 20+ deep analytical questions (e.g., *"Why not use Accuracy?"*, *"Why Mann-Whitney U instead of only t-test?"*, *"Why fit SMOTE inside CV folds?"*, *"How did you handle Unknown smoking status?"*, *"What does an Odds Ratio of X for hypertension mean?"*).

---

## 4. Mandatory Post-Module Verification & Self-Healing Protocol

After finishing **every single module**, Antigravity MUST run this 4-step verification loop BEFORE updating checklists or moving to the next module:

1. **Execution Test:**
   - Run every newly created/modified `.py` script and execute `.ipynb` notebooks via `jupyter nbconvert --to notebook --execute --inplace <notebook>` (or equivalent runner).
   - For Module 6, start Streamlit headlessly / run smoke tests on all app helper functions and verify zero import or runtime errors.
2. **Output & Sanity Audit:**
   - Verify expected artifacts exist and are non-empty (plots in `docs/figures/`, models in `models/`, PDF/PPTX in `docs/`).
   - Sanity-check data shapes, zero unexpected `NaN`/`inf` values, valid probability bounds (`[0, 1]`), and realistic metric ranges.
3. **Error Diagnosis & Auto-Fix:**
   - If ANY error, traceback, deprecation crash, or broken path occurs:
     - Diagnose the root cause immediately.
     - Patch the code cleanly (no hacks or silent `try/except: pass` suppression).
     - Re-run the full verification test until **100% green**.
4. **Git & Checklist Update:**
   - Run `git status` and confirm NO `.csv`, `.xlsx`, `.zip`, or `venv/` files are staged.
   - Commit with a clear, descriptive message (e.g., `git commit -m "Module 3: Completed Chi-Square, Mann-Whitney U, and Statsmodels Logit Odds Ratios"`).
   - Update completed `[x]` items in `DEVELOPMENT_STRATEGY.md`.
   - Output a concise **Module Completion & Verification Report** summarizing:
     - Files created/updated
     - Verification commands run & pass status
     - Any errors encountered & how they were resolved
     - Git branch/commit/PR status
     - Readiness to proceed to the next module.