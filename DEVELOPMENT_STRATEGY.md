# Project Development Strategy & Master Checklist
**Project #49:** Stroke Risk Factors & Clinical Triage App  
**Student Name:** Vaddi Varshitha  
**ID Number:** S230617  
**Course:** Introduction to Data Science (25A12104) — AIML-2A  
**Course Instructor:** Satish C  
**Project Helpers:** Pardha Saradhi (8328675049) | Rajesh (7032479565) | Srikanth (7995991365)  
**Final Submission Deadline:** 30th November 2026  

---

## 1. Core Problem Statement & Evaluation Alignment

### Assigned Problem Statement
> *"Impute BMI and smoking attributes, test hypertension/glucose risk via Chi-Square & Logistic Regression, and deploy a Streamlit stroke risk screening app."*

### Primary Dataset
- **Name:** Stroke Prediction Dataset (Kaggle — `fedesoriano`)
- **Link:** `https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset`
- **Target File:** `healthcare-dataset-stroke-data.csv` (5,110 rows, 12 columns; severe class imbalance ~4.87% positive stroke cases)
- **Dataset Enrichment (Encouraged by Rubric):** Incorporate WHO BMI clinical categories, ADA Blood Glucose clinical thresholds, and AHA Hypertension/Cardiometabolic risk indicators to enrich domain features and clinical interpretability.

### Grading Rubric Mapping (100%)
| Evaluation Criterion | Weight | How This Strategy Satisfies It |
| :--- | :---: | :--- |
| **Analytical Reasoning & Viva Clarity** | **30%** | Every modeling/imputation choice is statistically justified, documented inline, and summarized in `docs/VIVA_PREP.md`. |
| **Data Preprocessing, EDA & Feature Engineering** | **25%** | Rigorous missing-value imputation (`bmi`, `smoking_status`), outlier handling, univariate/bivariate/multivariate plots, Chi-Square & Mann-Whitney/t-tests. |
| **Modeling, Metrics & Working Output** | **20%** | Multi-model benchmark (Logistic Regression + 4 models), Stratified K-Fold CV, imbalance-aware metrics (Recall, F1, ROC-AUC, PR-AUC), and an interactive Streamlit Triage App. |
| **Documentation & PPT Quality** | **15%** | 10-section formal PDF report (Times New Roman/Arial 12pt, 1.5 spacing, justified) + crisp 10–12 slide presentation in `docs/`. |
| **GitHub Discipline & Dataset Exploration** | **10%** | Clean repo structure (`data/`, `notebooks/`, `src/`, `docs/`), feature branches merged via PRs, consistent descriptive commits, strict `.gitignore`. |

---

## 2. Antigravity Governance Rules (STRICT)

1. **Dual-File Compliance:** Antigravity MUST read both `DEVELOPMENT_STRATEGY.md` (this file) and `DEVELOPMENT_INSTRUCTIONS.md` before starting any module.
2. **Strict Sequential Execution:** Work on **one module at a time** (Module 0 -> Module 8). Do NOT skip ahead or merge incomplete work.
3. **Mandatory Post-Module Review & Self-Healing Gate:** After completing every module, Antigravity MUST:
   - Execute all automated checks, scripts, or notebooks for that module.
   - Verify outputs, file paths, statistical sanity, and zero runtime/syntax/lint errors.
   - If any error, warning, or discrepancy is detected, **diagnose and fix it immediately** before marking the module complete.
   - Update the checklist checkboxes (`[ ]` -> `[x]`) in this file and output a concise **Module Status Report** before asking permission or proceeding to the next module.
4. **Git Branching Discipline:** Every major phase MUST be developed on its own feature branch, committed incrementally with descriptive messages, pushed to GitHub, and merged into `main` via a Pull Request. **Never commit raw `.csv`, `.xlsx`, `.zip`, `venv/`, or `.env` files.**
5. **No Blind Black-Box Code:** All code must be cleanly modularized, well-commented, and accompanied by plain-English analytical explanations so the student can defend every line in the Viva.

---

## 3. Fast-Track Schedule (Nov 25 – Nov 30, 2026)

| Date | Modules | Git Feature Branch | Deliverable Milestone |
| :--- | :--- | :--- | :--- |
| **Day 1 (Nov 25)** | Module 0 & Module 1 | `main` -> `setup-and-cleaning` | GitHub account/repo live, environment ready, dataset downloaded, cleaning & imputation complete. |
| **Day 2 (Nov 26)** | Module 2 & Module 3 | `eda-and-stats` | Full EDA notebook, publication-ready charts, Chi-Square, Mann-Whitney/t-tests & Logistic Regression Odds Ratios. |
| **Day 3 (Nov 27)** | Module 4 & Module 5 | `feature-eng-and-modeling` | Feature engineering pipeline, imbalance handling, 5-model benchmark, tuned & saved best models. |
| **Day 4 (Nov 28)** | Module 6 | `streamlit-triage-app` | Full interactive Streamlit Clinical Triage & Risk Screening App tested end-to-end. |
| **Day 5 (Nov 29)** | Module 7 & Module 8 | `docs-and-presentation` | Official PDF Report, 10–12 Slide PPT, Viva Prep Guide, Final README & Repo Audit. |
| **Day 6 (Nov 30)** | Final Buffer & Submit | `main` | Submit GitHub link, PDF report, and PPT before the **30th November 2026** deadline. |

---

## 4. Module-by-Module Breakdown & Master Checklists

### Module 0: GitHub Account, Local Git & Repository Initialization
**Goal:** Establish a compliant GitHub account, local Git identity, repository (`ds-stroke-risk-triage-app`), folder hierarchy, Python virtual environment, and dataset placement.
- [x] **0.1 (User Action):** Create GitHub account at `https://github.com` using college/professional email (recommended username: `vaddi-varshitha-ds`). *(Verified existing remote `RajuGudem26`)*
- [x] **0.2 (User Action):** Create a new **Public** repository named `ds-stroke-risk-triage-app` with a `README.md`.
- [x] **0.3:** Verify local Git installation and configure `user.name` ("Vaddi Varshitha") and `user.email`.
- [x] **0.4:** Clone repository locally and create mandatory folder structure (`data/raw/`, `data/processed/`, `notebooks/`, `src/`, `models/`, `docs/figures/`).
- [x] **0.5:** Create strict `.gitignore` (excluding `venv/`, `.env`, `__pycache__/`, `.ipynb_checkpoints/`, `*.csv`, `*.xlsx`, `*.zip`).
- [x] **0.6:** Create Python virtual environment (`venv`) and install dependencies via `requirements.txt`.
- [x] **0.7:** Place `healthcare-dataset-stroke-data.csv` inside `data/raw/` (downloaded from Kaggle) and document download instructions + schema in `data/README.md`.
- [x] **0.8 (Review Gate):** Verify `git status` shows NO ignored `.csv` or `venv/` files, commit initial scaffold to `main`, and push.

---

### Module 1: Data Ingestion, Auditing & Clinical Imputation Pipeline
**Branch:** `setup-and-cleaning`  
**Goal:** Audit raw data quality and build a reproducible preprocessing script (`src/data_cleaning.py`) + notebook (`notebooks/01_data_audit_and_cleaning.ipynb`) focusing on **BMI** and **Smoking Status** imputation.
- [x] **1.1:** Create and switch to branch `setup-and-cleaning`.
- [x] **1.2:** Load `healthcare-dataset-stroke-data.csv` and log shape, data types, duplicates, missing values, and class balance (`stroke` distribution).
- [x] **1.3:** Handle rare `gender == "Other"` record (1 row) cleanly with documented justification.
- [x] **1.4:** Audit `bmi` missing values (~201 NaNs) and extreme outliers (e.g., BMI > 60). Implement **Age- & Gender-Stratified Median / KNN Imputation** (fit on training split later to prevent leakage; exploratory clean copy for EDA) and create `bmi_was_missing` indicator.
- [x] **1.5:** Audit `smoking_status == "Unknown"` (~1,544 rows). Analyze its strong correlation with `age` (children < 12). Implement documented domain-aware handling: impute children `< 12` as `"never smoked"` and evaluate predictive/mode imputation vs. preserving `"Unknown"` with a `smoking_was_unknown` flag.
- [x] **1.6:** Save cleaned EDA dataset to `data/processed/stroke_cleaned_eda.csv` (gitignored) and unit-test `src/data_cleaning.py`.
- [x] **1.7 (Review Gate):** Verify zero unexpected NaNs, valid data types, clean notebook execution, commit, push, and merge PR into `main`.

---

### Module 2: Exploratory Data Analysis (EDA) & Clinical Visualizations
**Branch:** `eda-and-stats`  
**Goal:** Uncover stroke risk patterns through univariate, bivariate, and multivariate visual storytelling (`notebooks/02_eda_and_visualizations.ipynb`) and save high-DPI charts to `docs/figures/`.
- [x] **2.1:** Checkout `main`, pull latest, and create branch `eda-and-stats`.
- [x] **2.2:** Univariate analysis: distributions of `age`, `avg_glucose_level`, `bmi`, and count plots of all categorical variables (`stroke`, `hypertension`, `heart_disease`, `smoking_status`, `work_type`, `ever_married`, `Residence_type`).
- [x] **2.3:** Bivariate analysis against `stroke`: stacked bar charts (stroke rate by hypertension, heart disease, smoking status, marital status, work type) and violin/box/KDE plots (`age`, `avg_glucose_level`, `bmi` by stroke outcome).
- [x] **2.4:** Multivariate analysis: correlation heatmap, Age vs. Avg Glucose scatter colored by Stroke, and interaction plot of Hypertension × Glucose Level on Stroke Risk.
- [x] **2.5:** Export all key figures at 300 DPI with clear titles, axis labels, and legends into `docs/figures/` for the Report and PPT.
- [x] **2.6 (Review Gate):** Verify every chart is legible, saved in `docs/figures/`, and accompanied by written clinical takeaways in Markdown cells.

---

### Module 3: Statistical Hypothesis Testing & Epidemiological Risk Analysis
**Branch:** `eda-and-stats` (continued)  
**Goal:** Statistically validate hypertension, glucose, and lifestyle stroke risk factors using **Chi-Square tests**, **Normality + Mann-Whitney U / Welch's t-tests**, and **Statsmodels Logistic Regression Odds Ratios** (`src/statistical_tests.py` & `notebooks/03_statistical_hypothesis_testing.ipynb`).
- [x] **3.1:** Define clinical reference bins for testing: WHO BMI categories & ADA Glucose risk categories (`Normal <100`, `Prediabetes 100-125`, `High/Diabetic >=126` or clinical `>=200` high-risk tier).
- [x] **3.2:** Run **Chi-Square Tests of Independence** (with Cramér's V effect size) for `hypertension`, `heart_disease`, `glucose_category`, `smoking_status`, `ever_married`, `work_type`, and `Residence_type` vs. `stroke`.
- [x] **3.3:** Run Shapiro-Wilk / D'Agostino normality tests on `age`, `avg_glucose_level`, and `bmi`, followed by appropriate group comparison tests (**Mann-Whitney U** and **Welch's t-test**) between Stroke vs. Non-Stroke patients.
- [x] **3.4:** Fit an interpretable **Statsmodels Logit (Logistic Regression)** model to compute **Adjusted Odds Ratios (OR)**, 95% Confidence Intervals, and p-values for hypertension, glucose, age, heart disease, BMI, and smoking.
- [x] **3.5:** Export formatted statistical summary tables (`docs/figures/chi_square_results.csv` or markdown/png tables) for the Project Report.
- [x] **3.6 (Review Gate):** Verify all p-values, Odds Ratios, and conclusions are mathematically sound and error-free; commit, push, and merge PR `eda-and-stats` into `main`.

---

### Module 4: Feature Engineering & Imbalance-Aware Preprocessing Pipeline
**Branch:** `feature-eng-and-modeling`  
**Goal:** Build a leakage-free Scikit-Learn `ColumnTransformer` + feature engineering module (`src/feature_engineering.py`) ready for both model training and live Streamlit inference.
- [ ] **4.1:** Checkout `main`, pull latest, and create branch `feature-eng-and-modeling`.
- [ ] **4.2:** Engineer clinically meaningful features: `bmi_category` (WHO), `glucose_risk_tier` (ADA), `age_group`, `cardiometabolic_risk_score` (`hypertension + heart_disease + (avg_glucose_level >= 140)`), and interaction terms (`age_x_glucose`, `hypertension_x_age`).
- [ ] **4.3:** Build a unified Scikit-Learn preprocessing pipeline:
  - Numeric features: Iterative/Median Imputer + `StandardScaler`
  - Categorical features: Mode/Domain Imputer + `OneHotEncoder(handle_unknown='ignore')`
- [ ] **4.4:** Perform **Stratified Train-Test Split** (80/20, `random_state=42`, `stratify=y`) BEFORE any scaling, imputation fitting, or resampling to guarantee zero data leakage.
- [ ] **4.5:** Configure class-imbalance strategies to compare: `class_weight='balanced'` vs. **SMOTE** (applied strictly inside cross-validation folds using `imblearn.pipeline.Pipeline`).
- [ ] **4.6 (Review Gate):** Unit-test pipeline fit/transform shapes, verify zero NaNs in transformed matrices, and confirm no target leakage.

---

### Module 5: Predictive Modeling, Hyperparameter Tuning & Evaluation
**Branch:** `feature-eng-and-modeling` (continued)  
**Goal:** Train, tune, and compare multiple classification models (`src/train_models.py` & `notebooks/04_modeling_and_evaluation.ipynb`), prioritizing **clinical recall (sensitivity)** alongside **ROC-AUC**, **PR-AUC**, and **F1-Score**.
- [ ] **5.1:** Train and benchmark at least **5 models** using **5-Fold Stratified Cross-Validation**:
  1. **Logistic Regression** (Mandatory primary interpretable model from Problem Statement)
  2. **Random Forest Classifier**
  3. **Gradient Boosting / XGBoost Classifier**
  4. **K-Nearest Neighbors (KNN)**
  5. **Gaussian Naive Bayes**
- [ ] **5.2:** Compare imbalance handling (`class_weight='balanced'` vs. `SMOTE`) and evaluate on held-out Test Set using: **Recall (Sensitivity)**, **Precision**, **F1-Score**, **ROC-AUC**, **PR-AUC**, and **Specificity**.
- [ ] **5.3:** Perform **Threshold Optimization** (e.g., tuning decision threshold from 0.50 to clinical triage thresholds for High/Moderate/Low risk tiers) to maximize Stroke Recall while controlling false alarms.
- [ ] **5.4:** Perform Error Analysis (inspect False Negatives vs. False Positives) and generate comparative ROC Curve, Precision-Recall Curve, Confusion Matrix, and Feature Importance plots in `docs/figures/`.
- [ ] **5.5:** Serialize the best interpretable model (`models/logistic_regression_pipeline.joblib`), best ensemble model (`models/best_model_pipeline.joblib`), and metadata/metrics JSON (`models/model_metrics.json`).
- [ ] **5.6 (Review Gate):** Run an end-to-end inference test script on sample patient profiles, verify saved artifacts load cleanly, commit, push, and merge PR `feature-eng-and-modeling` into `main`.

---

### Module 6: Interactive Streamlit Clinical Triage & Risk Screening App
**Branch:** `streamlit-triage-app`  
**Goal:** Build and verify a polished, clinical-grade Streamlit web application (`src/app.py`) for real-time stroke risk screening, explainability, and visual analytics.
- [ ] **6.1:** Checkout `main`, pull latest, and create branch `streamlit-triage-app`.
- [ ] **6.2:** Design a clean multi-tab layout with medical-grade styling:
  - **Tab 1: Clinical Triage Screener** — Interactive patient input form (Age, Gender, Hypertension, Heart Disease, Avg Glucose, BMI, Smoking Status, Marital Status, Work Type, Residence), model selector (Logistic Regression vs. Best Ensemble), real-time Risk Probability Gauge, Triage Tier Badge (`LOW` / `MODERATE` / `HIGH RISK`), and personalized actionable recommendations.
  - **Tab 2: Patient Risk Explainability** — Dynamic breakdown of which patient factors pushed risk up or down (local feature contributions / Odds Ratios).
  - **Tab 3: EDA & Statistical Evidence** — Interactive Plotly charts and Chi-Square / Odds Ratio summary tables.
  - **Tab 4: Model Benchmarks & Threshold Lab** — Interactive threshold slider showing live trade-off between Sensitivity (Recall) and False Positive Rate, plus model comparison table & ROC/PR curves.
- [ ] **6.3:** Add a prominent **Clinical Decision-Support Disclaimer** clarifying this app is for educational screening support only, not a medical diagnosis.
- [ ] **6.4:** Verify app launches cleanly via `streamlit run src/app.py` with zero exceptions, warnings, or broken layouts, and capture high-resolution screenshots for `docs/figures/`.
- [ ] **6.5 (Review Gate):** Test edge-case patient inputs (infant, elderly high-glucose hypertensive smoker, missing/default values), commit, push, and merge PR `streamlit-triage-app` into `main`.

---

### Module 7: Official Project Report (PDF), Presentation (PPT) & Viva Prep
**Branch:** `docs-and-presentation`  
**Goal:** Produce all required academic deliverables strictly adhering to the course documentation and presentation guidelines.
- [ ] **7.1:** Checkout `main`, pull latest, and create branch `docs-and-presentation`.
- [ ] **7.2:** Generate the complete 10-section **Official Project Report** (`docs/Project_Report_S230617_Vaddi_Varshitha.pdf`) formatted in Times New Roman / Arial (12 pt body, 14–16 pt Bold headings, 1.5 line spacing, justified alignment, numbered & captioned figures/tables).
- [ ] **7.3:** Generate the **10–12 Slide Presentation** (`docs/Presentation_S230617_Vaddi_Varshitha.pptx`) following the exact Slide 1–10 rubric structure.
- [ ] **7.4:** Create `docs/VIVA_PREP.md` containing 20+ likely Viva questions and clear, first-person analytical answers covering every pipeline choice.
- [ ] **7.5 (Review Gate):** Verify PDF opens cleanly with all figures embedded, PPT has 10–12 slides with zero text overflow, commit, push, and merge PR `docs-and-presentation` into `main`.

---

### Module 8: Final Repository Polish, README & Pre-Submission Audit
**Branch:** `main` (via final PR `final-audit`)  
**Goal:** Conduct a complete quality audit of the GitHub repository against every rule in the course handbook before submission.
- [ ] **8.1:** Write a comprehensive root `README.md` (Project Title, Student Info, Problem Statement, Kaggle Dataset Link & Schema, Tech Stack, Key Findings, Architecture, and Exact Setup/Run Commands).
- [ ] **8.2:** Run full repository audit:
  - Confirm `git log --oneline` shows steady, descriptive commits across feature branches and PR merges.
  - Confirm NO raw `.csv`, `.zip`, or `venv/` files exist in Git history or remote repo.
  - Confirm all notebooks run cleanly top-to-bottom with clear outputs.
  - Confirm `streamlit run src/app.py` launches cleanly in a fresh check.
- [ ] **8.3 (Final Review Gate):** Present the Final Readiness Sign-Off checklist for submission before **30th November 2026**.