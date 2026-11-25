# Dataset Documentation — Stroke Prediction Dataset

## 1. Source & Acquisition
- **Dataset Name:** Stroke Prediction Dataset (Kaggle — `fedesoriano`)
- **Source URL:** https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
- **Expected Raw File Path:** `data/raw/healthcare-dataset-stroke-data.csv`
- **Expected Dimensions:** 5,110 rows × 12 columns (~4.87% positive `stroke = 1` class imbalance)

### Manual Download Instructions
1. Visit https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset (sign in to Kaggle if prompted).
2. Click **Download** to download the archive (`archive.zip`).
3. Extract `healthcare-dataset-stroke-data.csv` and place it at:
   `data/raw/healthcare-dataset-stroke-data.csv`
4. **Do NOT commit** `.csv` or `.zip` files to Git; `data/raw/` and `data/processed/` are excluded via `.gitignore`.

---

## 2. Data Dictionary (12 Columns)

| # | Column Name | Data Type | Units / Allowed Values | Clinical Meaning | Known Data Quality Issues |
| :-: | :--- | :--- | :--- | :--- | :--- |
| 1 | `id` | Integer | Unique positive integer | Patient record identifier | Non-predictive identifier; drop prior to modeling. |
| 2 | `gender` | Categorical | `Male`, `Female`, `Other` | Biological sex / recorded gender | 1 record with `"Other"` (~0.02%); handle explicitly to prevent zero-variance dummy issues in statistical tests. |
| 3 | `age` | Float | Years (`0.08` – `82.0`) | Patient age in years | Fractional values `< 1.0` represent infants; verify pediatric records alongside `smoking_status`. |
| 4 | `hypertension` | Binary Int | `0` (No), `1` (Yes) | History of high blood pressure (AHA risk factor) | Imbalanced (~9.7% positive); check interaction with age and glucose. |
| 5 | `heart_disease` | Binary Int | `0` (No), `1` (Yes) | History of cardiac disease | Imbalanced (~5.4% positive); strong cardiovascular comorbidity. |
| 6 | `ever_married` | Categorical | `No`, `Yes` | Marital history | Strongly confounded with `age`; interpret alongside age in multivariable models. |
| 7 | `work_type` | Categorical | `Private`, `Self-employed`, `Govt_job`, `children`, `Never_worked` | Employment / lifestyle category | `"children"` category aligns with pediatric patients. |
| 8 | `Residence_type` | Categorical | `Urban`, `Rural` | Patient residential setting | Approximately 50/50 split; check statistical significance vs. stroke. |
| 9 | `avg_glucose_level` | Float | mg/dL (`~55.12` – `271.74`) | Average blood glucose concentration | Right-skewed / bimodal distribution (second peak near diabetic range `>= 126` mg/dL, ADA threshold). |
| 10 | `bmi` | Float | kg/m² (`~10.3` – `97.6`) | Body Mass Index | ~201 missing values (~3.93%) and extreme outliers (`> 60`); missingness may correlate with acute/emergency admissions. |
| 11 | `smoking_status` | Categorical | `formerly smoked`, `never smoked`, `smokes`, `Unknown` | Patient smoking history | ~1,544 `"Unknown"` records (~30.2%); heavily concentrated among children (`age < 12`). |
| 12 | `stroke` | Binary Int | `0` (No Stroke), `1` (Stroke) | **Target variable** — clinical stroke event | Severe class imbalance: ~249 positive (`4.87%`) vs. ~4,861 negative (`95.13%`). |

---

## 3. Clinical Reference Frameworks Used for Enrichment
- **WHO BMI Categories:** Underweight (`< 18.5`), Normal (`18.5–24.9`), Overweight (`25.0–29.9`), Obese (`>= 30.0`).
- **ADA Blood Glucose Thresholds:** Normal (`< 100` mg/dL), Prediabetes (`100–125` mg/dL), Hyperglycemia / Diabetic Range (`126–199` mg/dL), Severe Hyperglycemia (`>= 200` mg/dL).
- **AHA Cardiometabolic Indicators:** Hypertension (`hypertension == 1`), Heart Disease (`heart_disease == 1`), and elevated metabolic burden.

