"""Generate 10-12 slide PPTX presentation using python-pptx."""

from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from src.config import DOCS_DIR, FIGURES_DIR, METRICS_PATH

OUTPUT_PATH = DOCS_DIR / "Presentation_S230617_Vaddi_Varshitha.pptx"
STATS_JSON = FIGURES_DIR / "statistical_summary.json"


def _add_text_slide(prs, title: str, bullets: list[str], layout_idx: int = 1):
    slide = prs.slides.add_slide(prs.slide_layouts[layout_idx])
    slide.shapes.title.text = title
    tf = slide.placeholders[1].text_frame
    tf.clear()
    for i, bullet in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = bullet
        p.font.size = Pt(16)
        p.font.name = "Arial"
        p.space_after = Pt(6)
    return slide


def _add_figure_slide(prs, title: str, fig_name: str, caption: str = "", layout_idx: int = 5):
    fig_path = FIGURES_DIR / fig_name
    slide = prs.slides.add_slide(prs.slide_layouts[layout_idx])
    slide.shapes.title.text = title
    if fig_path.exists():
        slide.shapes.add_picture(str(fig_path), Inches(0.4), Inches(1.6), width=Inches(9.2))
    if caption:
        txBox = slide.shapes.add_textbox(Inches(0.5), Inches(6.8), Inches(9.0), Inches(0.5))
        tf = txBox.text_frame
        p = tf.paragraphs[0]
        p.text = caption
        p.font.size = Pt(11)
        p.font.italic = True
        p.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)
        p.alignment = PP_ALIGN.CENTER
    return slide


def generate_ppt():
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    stats = json.loads(STATS_JSON.read_text()) if STATS_JSON.exists() else {}
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Slide 1: Title
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "Stroke Risk Factors &\nClinical Triage App"
    subtitle = slide.placeholders[1]
    subtitle.text = (
        "Project #49 • Introduction to Data Science (25A12104)\n"
        "Vaddi Varshitha (S230617) • AIML-2A\n"
        "Instructor: Satish C • November 2026"
    )

    # Slide 2: Problem Statement
    _add_text_slide(prs, "Problem Statement & Research Questions", [
        "Impute BMI and smoking attributes in a 5,110-patient stroke dataset",
        "Validate hypertension & glucose as stroke risk factors via Chi-Square & Logistic Regression",
        "Deploy an interactive Streamlit clinical stroke risk screening app",
        "Severe class imbalance: only 4.87% positive stroke cases (249 / 5,110)",
        "Clinical relevance: early screening enables timely preventive intervention",
    ])

    # Slide 3: Dataset & Imputation
    _add_text_slide(prs, "Dataset Overview & Clinical Imputation Strategy", [
        "Kaggle Stroke Prediction Dataset: 5,110 rows × 12 columns",
        "BMI missing (201 rows, 3.93%) — MNAR: 19.9% stroke rate when missing vs. 4.3%",
        "→ Age-band & gender-stratified median imputation + bmi_was_missing indicator flag",
        "Smoking Unknown (1,544 rows, 30.2%) — 95.9% are children < 12 years old",
        "→ Pediatric < 12: imputed 'never smoked'; Adult >= 12: retained 'Unknown' category",
        "BMI > 65 winsorized (5 implausible records); 'Other' gender row dropped (0.02%)",
    ])

    # Slide 4: EDA Insights
    _add_figure_slide(prs, "EDA Visual Insights: Risk Factor Distributions",
                      "03_stroke_rate_by_risk_factors.png",
                      "Figure 3: Hypertension (13.3%), Heart Disease (17.0%), and Missing BMI (19.9%) show highest unadjusted stroke prevalence.")

    # Slide 5: Statistical Tests
    _add_figure_slide(prs, "Statistical Validation: Chi-Square & Adjusted Odds Ratios",
                      "07_odds_ratios_forest_plot.png",
                      "Figure 7: Age (OR=2.02/10yr), BMI Missingness (OR=3.70), Glucose (OR=1.04/10mg), Hypertension (OR=1.39) are significant adjusted predictors.")

    # Slide 6: Feature Engineering
    _add_text_slide(prs, "Feature Engineering & Leakage-Free Pipeline", [
        "Clinically enriched features: WHO BMI categories, ADA glucose risk tiers, age groups",
        "Interaction terms: cardiometabolic_risk_count, age×glucose, hypertension×age",
        "Leakage-free: Stratified 80/20 split BEFORE fitting imputers, scalers, encoders",
        "SMOTE applied strictly inside CV folds via imblearn.pipeline.Pipeline",
        "class_weight='balanced' outperformed SMOTE in CV ROC-AUC (0.849 vs. 0.842)",
    ])

    # Slide 7: Modeling Rationale
    _add_text_slide(prs, "5-Model Benchmark & Imbalance-Aware Evaluation", [
        "Logistic Regression (primary interpretable model, mandatory per problem statement)",
        "Random Forest • XGBoost • K-Nearest Neighbors (SMOTE) • Gaussian Naive Bayes",
        "5-Fold Stratified Cross-Validation with GridSearchCV hyperparameter tuning",
        "Imbalance-first metrics: ROC-AUC, PR-AUC, Recall (Sensitivity), F1-Score, Specificity",
        "Accuracy is misleading (95.1% for a majority-class dummy classifier)",
    ])

    # Slide 8: Model Comparison Results
    _add_figure_slide(prs, "Model Comparison: ROC & Precision-Recall Curves",
                      "08_model_comparison_roc_pr.png",
                      "Figure 8: XGBoost leads Test ROC-AUC (0.841); LR (0.831) is competitive with full explainability.")

    # Slide 9: Streamlit App
    _add_text_slide(prs, "Streamlit Clinical Triage & Risk Screening App", [
        "Tab 1: Live patient intake form → real-time Plotly risk gauge + triage tier badge",
        "Tab 2: Local LR explainability (transformed_x × β) + cohort Odds Ratio table",
        "Tab 3: Interactive EDA + Chi-Square + Mann-Whitney summary tables",
        "Tab 4: Model leaderboard, interactive ROC/PR curves, threshold trade-off lab",
        "Batch CSV upload with downloadable scored triage results",
        "Clinical disclaimer: educational screening prototype, not a medical diagnosis",
    ])

    # Slide 10: Key Takeaways & Future Scope
    _add_text_slide(prs, "Key Takeaways, Limitations & Future Scope", [
        "Age is the dominant stroke risk gradient (OR = 2.02 per 10 years of aging)",
        "Missing BMI is a powerful acute-admission proxy (OR = 3.70, p < 1e-9)",
        "Hypertension & hyperglycemia are independently significant after adjustment",
        "Limitations: single-source retrospective data, severe 95:5 imbalance, no temporal causation",
        "Future: multi-center validation, SHAP explainability, EHR integration, prospective deployment",
    ])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUTPUT_PATH))
    print(f"Presentation saved to: {OUTPUT_PATH} ({len(prs.slides)} slides)")
    return OUTPUT_PATH


if __name__ == "__main__":
    generate_ppt()

