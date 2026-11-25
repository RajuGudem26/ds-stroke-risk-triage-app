# Viva Defense Preparation Guide
**Project #49:** Stroke Risk Factors & Clinical Triage App  
**Student:** Vaddi Varshitha (`S230617`) | **Course:** Introduction to Data Science (`25A12104`)

---

## 1. Data Preprocessing & Imputation

**Q1: Why did you drop the single `gender == "Other"` row instead of keeping it?**  
Only 1 row out of 5,110 (0.02%) has `gender = "Other"`. Keeping it would create a near-zero-variance dummy variable in one-hot encoding, violating Chi-Square expected cell count assumptions (expected < 5) and risking quasi-separation in Logistic Regression. Dropping 0.02% of data has negligible impact on statistical power.

**Q2: Why use Age-Band & Gender-Stratified Median imputation for BMI instead of a simple global median?**  
BMI varies systematically with age and gender (e.g., children vs. elderly, male vs. female). A global median would mask these demographic patterns and introduce bias. Stratified imputation preserves the conditional distribution, producing more clinically realistic imputed values. I validated via CV ROC-AUC that stratified imputation performs comparably to KNN Imputer with lower computational cost.

**Q3: Why create `bmi_was_missing` as a feature instead of just imputing and moving on?**  
The stroke rate among patients with missing BMI is **19.9%** (40/201) compared to only **4.3%** (209/4,909) when BMI is recorded — a **4.7× higher prevalence**. This is not random missingness (MCAR); it is informative MNAR — likely reflecting acute/emergency admissions where patients are too ill to stand for measurement. The `bmi_was_missing` flag preserves this clinically valuable signal (confirmed as significant: OR = 3.70, p < 1e-9 in Logistic Regression).

**Q4: How did you handle Unknown smoking status?**  
Analysis showed 521 out of 543 children (< 12 years) had `Unknown` smoking status. I imputed these as `"never smoked"` (clinically valid — children don't smoke). For adults (>= 12), I compared two strategies via CV ROC-AUC:
- **Retain "Unknown" as an explicit category** + `smoking_was_unknown` flag
- **Predictive mode imputation** (age-band & gender stratified)

Both yielded equivalent ROC-AUC (~0.847), so I chose to retain `"Unknown"` to avoid fabricating unverified smoking histories for 1,023 adults.

**Q5: Why cap (winsorize) BMI at 65 instead of removing outliers?**  
Only 5 records have BMI > 65 (max 97.6), and all are non-stroke cases. Deleting stroke cases with extreme values would bias the model; deleting only non-stroke cases would artificially inflate stroke prevalence. Winsorizing at 65 preserves these patients while preventing extreme leverage in linear/distance-based models. I confirmed via CV ROC-AUC that capping vs. not capping produces identical discrimination.

---

## 2. EDA & Statistical Tests

**Q6: Why not rely on Accuracy as the primary metric?**  
With 95.13% of patients having `stroke = 0`, a naive majority-class classifier that always predicts "No Stroke" achieves 95.1% accuracy while catching **zero** actual stroke patients (0% Recall). In clinical screening, missing a stroke case (False Negative) is far worse than a false alarm. I use **ROC-AUC**, **PR-AUC**, **Recall (Sensitivity)**, and **F1-Score** as primary metrics.

**Q7: Why Mann-Whitney U test instead of only Welch's t-test?**  
Shapiro-Wilk and D'Agostino-Pearson normality tests reject normality (p < 0.001) for `age`, `avg_glucose_level`, and `bmi`. Mann-Whitney U is a non-parametric rank-based test that does not assume normal distributions. I report both: Mann-Whitney U as the primary robust test, and Welch's t-test as a secondary confirmatory check (which also does not require equal variances).

**Q8: What does Cramér's V tell you that the Chi-Square p-value alone does not?**  
The Chi-Square p-value tells you *whether* a significant association exists (hypothesis rejection), but with large samples (n = 5,109), even trivially small associations become statistically significant. Cramér's V measures *effect size* (strength of association), ranging from 0 (no association) to 1 (perfect). For example, `Residence_type` has p = 0.30 (not significant) with Cramér's V = 0.015 (negligible), while `bmi_was_missing` has V = 0.139 (moderate effect).

**Q9: How do you interpret the Adjusted Odds Ratio for hypertension?**  
The adjusted OR for `hypertension` is **1.39** (95% CI: 1.00–1.92, p = 0.049). This means: *after controlling for age, glucose, heart disease, BMI, smoking, gender, marital status, and residence*, patients with hypertension have **39% higher odds of stroke** compared to normotensive patients. The 95% CI barely includes 1.0, indicating borderline statistical significance — which is consistent with hypertension's risk being partially mediated through age and glucose.

**Q10: What does `age_per_10yr` OR = 2.02 mean clinically?**  
Every 10-year increase in age is associated with a **2.02× (102%) increase in the odds of stroke**, holding all other predictors constant. This is the strongest independent risk driver in the model — consistent with the epidemiological literature where stroke incidence approximately doubles each decade after age 55.

---

## 3. Feature Engineering & Modeling

**Q11: Why split BEFORE fitting imputers and scalers?**  
If you fit imputers or scalers on the full dataset (including test data) and then split, information from the test set "leaks" into the training process — the model has implicitly "seen" test distributions. This inflates performance estimates. By splitting first and fitting transformers only on `X_train`, the test set remains truly unseen.

**Q12: Why fit SMOTE inside CV folds instead of applying it once before CV?**  
Applying SMOTE before cross-validation means synthetic minority samples generated from the entire training set could overlap with validation folds — creating optimistic bias. Using `imblearn.pipeline.Pipeline` ensures SMOTE runs only on each fold's training portion, keeping validation folds clean. This is the methodologically correct approach.

**Q13: Why did `class_weight='balanced'` outperform SMOTE?**  
`class_weight='balanced'` adjusts the loss function to weight minority-class errors higher (inversely proportional to class frequency). It does not create synthetic samples. SMOTE generates interpolated synthetic points in feature space, which can produce noisy examples — especially problematic when features include one-hot encoded categoricals (interpolating between binary dummies creates non-interpretable fractional values). For this dataset, cost-sensitive reweighting proved more robust.

**Q14: Why is Logistic Regression competitive with tree ensembles here?**  
Stroke risk in this cohort is dominated by smooth monotonic gradients (age, avg_glucose_level) and additive binary comorbidities (hypertension, bmi_was_missing). These are precisely the patterns that regularized linear models capture well. Tree ensembles add complexity but gain minimal discrimination because there are no strong non-linear interactions that the engineered features (age×glucose, cardiometabolic_risk_count) don't already capture. Logistic Regression also provides transparent log-odds coefficients for clinical explainability.

**Q15: What is Youden's J statistic and how did you use it for threshold calibration?**  
Youden's J = Sensitivity + Specificity − 1 = TPR − FPR. It identifies the ROC curve point that maximizes the trade-off between True Positive Rate and False Positive Rate. I computed J on out-of-fold (OOF) predictions from 5-fold Stratified CV to avoid overfitting the threshold to test data. The resulting calibrated screening threshold (~0.49–0.54) achieves 80–84% clinical Recall while maintaining ~64–75% Specificity.

---

## 4. Streamlit App & Deployment

**Q16: What are the 3 clinical triage tiers and how are they determined?**  
- **Low Risk:** Predicted probability < screening threshold (e.g., < 0.54 for LR). Routine preventive care.
- **Moderate Risk:** Between screening and priority alert thresholds (e.g., 0.54–0.74). Preventive clinical review recommended.
- **High Risk (Priority Triage):** Above priority alert threshold (e.g., >= 0.74). Urgent cardiovascular & glycemic evaluation.

Thresholds are calibrated on out-of-fold CV probabilities (targeting >= 80% Recall) and can be adjusted interactively in the app's sidebar.

**Q17: How does the explainability tab work?**  
For the Logistic Regression model, each patient's log-odds score is a linear combination: `Σ(standardized_x_i × β_i) + intercept`. The Tab 2 bar chart shows the product `transformed_value × coefficient` for each feature, visually decomposing *why* a particular patient received their risk score. Positive bars push risk up; negative bars reduce risk.

**Q18: Why include a Clinical Decision-Support Disclaimer?**  
The model is trained on retrospective hospital records from a single source. It cannot account for imaging biomarkers, genetic factors, medication history, or temporal causation. It is an academic screening prototype and must never replace licensed clinical evaluation or diagnosis. The disclaimer ensures ethical transparency.

---

## 5. GitHub & Project Management

**Q19: Why use feature branches instead of committing directly to main?**  
Feature branches (`setup-and-cleaning`, `eda-and-stats`, `feature-eng-and-modeling`, `streamlit-triage-app`, `docs-and-presentation`) isolate each development phase. This enables incremental Pull Request reviews, clean rollback if needed, and a clear commit history. It mirrors industry-standard Git workflows.

**Q20: Why are CSV, ZIP, and venv files excluded from Git?**  
Raw data files are large and sensitive; they should not be version-controlled. The `.gitignore` excludes `*.csv`, `*.xlsx`, `*.zip`, `venv/`, `.env`, `__pycache__/`, and `.ipynb_checkpoints/`. Instead, `data/README.md` documents the Kaggle download URL and data dictionary so anyone can reproduce the project.

---

## 6. Advanced Questions

**Q21: What would you do differently if the class imbalance were even more severe (e.g., 0.1%)?**  
I would consider: (1) Cost-sensitive learning with higher minority-class penalty; (2) Anomaly/one-class detection (Isolation Forest, Autoencoders); (3) Stratified undersampling ensembles (BalancedBagging, EasyEnsemble); (4) Focal loss for gradient-boosted models; (5) Prospective enriched sampling to increase case prevalence.

**Q22: How would you validate this model for real-world hospital deployment?**  
(1) External validation on multi-center, multi-ethnic cohorts; (2) Prospective temporal validation (train on 2020–2024, test on 2025+); (3) Calibration analysis (reliability diagrams) to verify predicted probabilities match observed frequencies; (4) Fairness audits across age, gender, and ethnic subgroups; (5) HIPAA/GDPR compliance review; (6) Integration testing with Electronic Health Record (EHR) APIs.

**Q23: What is the difference between ROC-AUC and PR-AUC, and when is each more informative?**  
ROC-AUC measures discrimination across all thresholds using True Positive Rate vs. False Positive Rate. It is insensitive to class imbalance because it uses FPR (which is dominated by the large majority class). PR-AUC uses Precision vs. Recall, which is more informative under severe imbalance because Precision directly reflects the proportion of flagged patients who are true positives. For clinical screening with 4.87% prevalence, PR-AUC better reflects real-world triage workload.

**Q24: Why did you scale continuous features with StandardScaler instead of MinMaxScaler?**  
StandardScaler (zero mean, unit variance) is preferred for Logistic Regression because the L2 regularization penalty treats all coefficients equally when features are on the same scale. MinMaxScaler is sensitive to outliers and would compress most values into a narrow range given the extreme BMI/glucose values. KNN also benefits from StandardScaler over MinMaxScaler for distance computation.

**Q25: If you could add one external data source, what would it be and why?**  
I would add **longitudinal blood pressure readings** (systolic/diastolic over time) from EHR records. The current binary `hypertension` variable (yes/no) loses information about severity, duration, and control status. Continuous blood pressure trajectories would significantly improve risk stratification and enable time-to-event (survival) modeling.
