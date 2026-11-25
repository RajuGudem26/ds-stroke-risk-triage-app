"""Generate formal 10-section PDF project report using ReportLab."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.config import DOCS_DIR, FIGURES_DIR, METRICS_PATH

OUTPUT_PATH = DOCS_DIR / "Project_Report_S230617_Vaddi_Varshitha.pdf"
STATS_JSON = FIGURES_DIR / "statistical_summary.json"


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("TitlePage", parent=ss["Title"], fontSize=22, leading=28, alignment=TA_CENTER, spaceAfter=6, fontName="Times-Bold"))
    ss.add(ParagraphStyle("SubTitle", parent=ss["Normal"], fontSize=13, leading=17, alignment=TA_CENTER, spaceAfter=4, fontName="Times-Roman"))
    ss.add(ParagraphStyle("SectionHead", parent=ss["Heading1"], fontSize=15, leading=20, spaceAfter=8, spaceBefore=14, fontName="Times-Bold", textColor=colors.HexColor("#1a365d")))
    ss.add(ParagraphStyle("BodyJ", parent=ss["Normal"], fontSize=12, leading=18, alignment=TA_JUSTIFY, fontName="Times-Roman"))
    ss.add(ParagraphStyle("Caption", parent=ss["Normal"], fontSize=10, leading=14, alignment=TA_CENTER, fontName="Times-Italic", spaceAfter=10))
    ss.add(ParagraphStyle("BulletJ", parent=ss["BodyJ"], leftIndent=20, bulletIndent=10, bulletFontName="Times-Bold"))
    return ss


def _add_figure(elements, fig_name: str, caption: str, width=14 * cm, ss=None):
    p = FIGURES_DIR / fig_name
    if p.exists():
        elements.append(Image(str(p), width=width, height=width * 0.55))
        elements.append(Paragraph(caption, ss["Caption"]))
    else:
        elements.append(Paragraph(f"[Figure not found: {fig_name}]", ss["Caption"]))


def _make_table(headers, rows, col_widths=None):
    data = [headers] + rows
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#edf2f7")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def generate_report():
    ss = _styles()
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    stats = json.loads(STATS_JSON.read_text()) if STATS_JSON.exists() else {}

    doc = SimpleDocTemplate(str(OUTPUT_PATH), pagesize=A4,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            leftMargin=2.5 * cm, rightMargin=2.5 * cm)
    story = []

    # --- Section 1: Title Page ---
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("Stroke Risk Factors &<br/>Clinical Triage App", ss["TitlePage"]))
    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph("Project #49 — Introduction to Data Science (25A12104)", ss["SubTitle"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("<b>Student:</b> Vaddi Varshitha (S230617) — AIML-2A", ss["SubTitle"]))
    story.append(Paragraph("<b>Course Instructor:</b> Satish C", ss["SubTitle"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("November 2026", ss["SubTitle"]))
    story.append(PageBreak())

    # --- Section 2: Abstract ---
    story.append(Paragraph("1. Abstract", ss["SectionHead"]))
    story.append(Paragraph(
        "This project investigates stroke risk factors using the Kaggle Stroke Prediction Dataset "
        "(5,110 patients, 4.87% stroke prevalence). Missing BMI values were imputed via age-band and "
        "gender-stratified medians with an informative missingness indicator, while pediatric Unknown "
        "smoking status was clinically resolved. Chi-Square tests identified hypertension, heart disease, "
        "glucose risk tier, and BMI missingness as significantly associated with stroke (p < 0.001). "
        "A multivariable Statsmodels Logistic Regression confirmed age (OR = 2.02 per 10 years), "
        "BMI missingness (OR = 3.70), glucose (OR = 1.04 per 10 mg/dL), and hypertension (OR = 1.39) "
        "as independently significant adjusted predictors. Five classification models were benchmarked "
        "using 5-fold Stratified Cross-Validation; Logistic Regression achieved a test ROC-AUC of 0.83 "
        "with 80–84% clinical Recall at a calibrated screening threshold. An interactive Streamlit "
        "Clinical Triage App provides real-time risk scoring, local explainability, and batch screening "
        "for educational clinical decision support.",
        ss["BodyJ"],
    ))
    story.append(Spacer(1, 0.3 * cm))

    # --- Section 3: Problem Statement ---
    story.append(Paragraph("2. Problem Statement & Domain Context", ss["SectionHead"]))
    story.append(Paragraph(
        "Stroke is a leading cause of death and long-term disability worldwide. Early identification of "
        "at-risk individuals through screening of modifiable risk factors (hypertension, hyperglycemia, "
        "smoking, obesity) can enable timely clinical intervention. The assigned problem statement requires: "
        "(1) imputing missing BMI and smoking attributes, (2) validating hypertension and glucose as stroke "
        "risk factors via Chi-Square tests and Logistic Regression, and (3) deploying an interactive "
        "Streamlit stroke risk screening application.",
        ss["BodyJ"],
    ))
    story.append(Spacer(1, 0.3 * cm))

    # --- Section 4: Objectives ---
    story.append(Paragraph("3. Objectives & Research Questions", ss["SectionHead"]))
    objectives = [
        "Audit data quality and impute missing BMI (3.93%) and Unknown smoking status (30.2%) using clinically defensible strategies.",
        "Visualize univariate, bivariate, and multivariate stroke risk patterns through publication-quality EDA.",
        "Statistically validate categorical and continuous stroke risk factors using Chi-Square, Mann-Whitney U, Welch's t-test, and Statsmodels Logistic Regression Odds Ratios.",
        "Engineer WHO/ADA/AHA clinical features and build a leakage-free preprocessing pipeline.",
        "Benchmark 5 classification models with imbalance-aware metrics (ROC-AUC, PR-AUC, Recall, F1) and calibrate clinical triage thresholds.",
        "Deploy a 4-tab interactive Streamlit Clinical Triage & Risk Screening App with real-time explainability.",
    ]
    for i, obj in enumerate(objectives, 1):
        story.append(Paragraph(f"{i}. {obj}", ss["BulletJ"]))
    story.append(Spacer(1, 0.3 * cm))

    # --- Section 5: Dataset Description ---
    story.append(Paragraph("4. Dataset Description & Data Dictionary", ss["SectionHead"]))
    story.append(Paragraph(
        "The Stroke Prediction Dataset (Kaggle, fedesoriano) contains 5,110 patient records with 12 columns. "
        "Target variable <b>stroke</b> exhibits severe class imbalance: 249 positive cases (4.87%) vs. 4,861 negative (95.13%). "
        "Key quality issues include 201 missing BMI values (with informative MNAR pattern — 19.9% stroke rate when BMI is missing "
        "vs. 4.3% when present) and 1,544 Unknown smoking status entries (95.9% concentrated among children < 12 years).",
        ss["BodyJ"],
    ))
    _add_figure(story, "01_target_imbalance.png", "Figure 1: Target Class Distribution (Severe 95:5 Imbalance)", ss=ss)

    # --- Section 6: Tech Stack ---
    story.append(Paragraph("5. Tech Stack & Libraries", ss["SectionHead"]))
    story.append(Paragraph(
        "Python 3.13 • pandas • NumPy • SciPy • Statsmodels (Logistic Regression Odds Ratios) • "
        "scikit-learn (pipelines, CV, metrics) • imbalanced-learn (SMOTE) • XGBoost • "
        "Matplotlib/Seaborn (300-DPI static figures) • Plotly (interactive app charts) • "
        "Streamlit (web application) • ReportLab (PDF generation) • python-pptx (PPTX slides) • "
        "Git/GitHub (version control with feature branches and PR merges).",
        ss["BodyJ"],
    ))
    story.append(Spacer(1, 0.3 * cm))

    # --- Section 7: Methodology & Pipeline ---
    story.append(Paragraph("6. Methodology & Data Science Pipeline", ss["SectionHead"]))
    story.append(Paragraph(
        "<b>Data Cleaning:</b> Dropped the single 'Other' gender row (0.02%); created bmi_was_missing and "
        "smoking_was_unknown indicator flags; imputed pediatric Unknown smoking as 'never smoked'; applied "
        "age-band & gender-stratified median BMI imputation; winsorized implausible BMI > 65.",
        ss["BodyJ"],
    ))
    story.append(Paragraph(
        "<b>Feature Engineering:</b> WHO BMI categories, ADA glucose risk tiers, age groups, "
        "cardiometabolic risk count, age×glucose and hypertension×age interaction terms.",
        ss["BodyJ"],
    ))
    story.append(Paragraph(
        "<b>Leakage-Free Pipeline:</b> Stratified 80/20 train-test split (random_state=42, stratify=y) BEFORE "
        "fitting imputers, scalers (StandardScaler), and encoders (OneHotEncoder). SMOTE applied strictly "
        "inside CV folds via imblearn.pipeline.Pipeline.",
        ss["BodyJ"],
    ))
    _add_figure(story, "02_numeric_distributions.png", "Figure 2: Numeric Feature Distributions with Clinical Thresholds", ss=ss)
    _add_figure(story, "03_stroke_rate_by_risk_factors.png", "Figure 3: Bivariate Stroke Prevalence Across Risk Factors", ss=ss)
    story.append(PageBreak())

    # --- Section 8: Results ---
    story.append(Paragraph("7. Results, Visualizations & Comparative Analysis", ss["SectionHead"]))

    # Chi-Square table
    if stats.get("chi_square_results"):
        story.append(Paragraph("<b>Table 1: Chi-Square Tests of Independence (α = 0.05)</b>", ss["BodyJ"]))
        chi_headers = ["Feature", "χ²", "df", "p-value", "Cramér's V", "Decision"]
        chi_rows = []
        for r in stats["chi_square_results"]:
            chi_rows.append([
                r["feature"], f'{r["chi2"]:.2f}', str(r["dof"]),
                f'{r["p_value"]:.2e}', f'{r["cramers_v"]:.4f}', r["decision"][:10],
            ])
        story.append(_make_table(chi_headers, chi_rows, col_widths=[2.8*cm, 1.5*cm, 0.8*cm, 2.0*cm, 1.8*cm, 2.5*cm]))
        story.append(Spacer(1, 0.3 * cm))

    _add_figure(story, "07_odds_ratios_forest_plot.png", "Figure 7: Adjusted Odds Ratios Forest Plot (Statsmodels Logit, 95% CI)", ss=ss)

    # Model leaderboard table
    if metrics.get("leaderboard"):
        story.append(Paragraph("<b>Table 2: 5-Model Benchmark Leaderboard</b>", ss["BodyJ"]))
        lb_headers = ["Model", "CV ROC", "Test ROC", "Test PR", "Thresh", "Recall", "F1"]
        lb_rows = []
        for r in metrics["leaderboard"]:
            cal = r["test_calibrated"]
            lb_rows.append([
                r["model"][:22], f'{r["cv_roc_auc"]:.4f}', f'{r["test_roc_auc"]:.4f}',
                f'{r["test_pr_auc"]:.4f}', f'{r["calibrated_screen_thresh"]:.2f}',
                f'{cal["recall_sensitivity"]:.3f}', f'{cal["f1_score"]:.3f}',
            ])
        story.append(_make_table(lb_headers, lb_rows, col_widths=[3.5*cm, 1.5*cm, 1.5*cm, 1.5*cm, 1.3*cm, 1.3*cm, 1.3*cm]))
        story.append(Spacer(1, 0.3 * cm))

    _add_figure(story, "08_model_comparison_roc_pr.png", "Figure 8: ROC & Precision-Recall Curve Comparison (Test Set)", ss=ss)
    _add_figure(story, "09_confusion_matrix_best_model.png", "Figure 9: Confusion Matrices at Calibrated Screening Thresholds", ss=ss)
    _add_figure(story, "10_feature_importance.png", "Figure 10: Feature Importance — Logistic Regression vs. Random Forest", ss=ss)
    story.append(PageBreak())

    # --- Section 9: Conclusion ---
    story.append(Paragraph("8. Conclusion, Limitations & Future Scope", ss["SectionHead"]))
    story.append(Paragraph(
        "<b>Key Findings:</b> Age is the dominant stroke risk gradient (OR = 2.02 per 10 years). "
        "Hypertension (OR = 1.39) and elevated blood glucose (OR = 1.04 per 10 mg/dL) are independently "
        "significant after adjusting for confounders. Missing BMI is a powerful acute-admission proxy "
        "(OR = 3.70). Class-weight balancing outperformed SMOTE for imbalanced stroke prediction.",
        ss["BodyJ"],
    ))
    story.append(Paragraph(
        "<b>Limitations:</b> Retrospective single-source hospital data limits external generalizability; "
        "severe 95:5 class imbalance constrains precision at high-recall thresholds; temporal/causal "
        "inference is not possible from cross-sectional records; the model is not a medical diagnostic "
        "and must not replace licensed clinical evaluation.",
        ss["BodyJ"],
    ))
    story.append(Paragraph(
        "<b>Future Scope:</b> Multi-center prospective validation; integration of imaging biomarkers "
        "(CT/MRI); longitudinal risk trajectories; Explainability via SHAP values; deployment as a "
        "HIPAA-compliant clinical decision-support tool with EHR integration.",
        ss["BodyJ"],
    ))
    story.append(Spacer(1, 0.3 * cm))

    # --- Section 10: References ---
    story.append(Paragraph("9. References", ss["SectionHead"]))
    refs = [
        "fedesoriano (2021). Stroke Prediction Dataset. Kaggle. https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset",
        "World Health Organization (WHO). BMI Classification. https://www.who.int/data/gho/data/themes/topics/topic-details/GHO/body-mass-index",
        "American Diabetes Association (ADA). Standards of Care in Diabetes — Classification and Diagnosis.",
        "American Heart Association (AHA). Heart Disease and Stroke Statistics.",
        "Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, pp. 2825–2830.",
        "Seabold & Perktold (2010). Statsmodels: Econometric and Statistical Modeling with Python.",
        "Streamlit Inc. Streamlit Documentation. https://docs.streamlit.io",
    ]
    for i, ref in enumerate(refs, 1):
        story.append(Paragraph(f"[{i}] {ref}", ss["BodyJ"]))

    doc.build(story)
    print(f"Report saved to: {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    generate_report()
