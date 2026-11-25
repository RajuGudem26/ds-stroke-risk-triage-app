"""Interactive Streamlit Clinical Triage & Stroke Risk Screening Application."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BEST_MODEL_PATH,
    CLEANED_EDA_PATH,
    FIGURES_DIR,
    LR_MODEL_PATH,
    METRICS_PATH,
)

# ---------------------------------------------------------------------------
# Pure Helper Functions (Testable Independently of UI)
# ---------------------------------------------------------------------------

def classify_triage_tier(
    prob: float,
    low_thresh: float,
    high_thresh: float,
) -> dict[str, str]:
    """Assign color-coded clinical triage tier from predicted stroke probability."""
    if prob >= high_thresh:
        return {
            "tier": "HIGH RISK",
            "color": "#c53030",
            "bg": "#fff5f5",
            "action": (
                "Priority clinical follow-up recommended: schedule urgent physician "
                "cardiovascular & glycemic evaluation."
            ),
        }
    if prob >= low_thresh:
        return {
            "tier": "MODERATE RISK",
            "color": "#b7791f",
            "bg": "#fffff0",
            "action": (
                "Preventive clinical review recommended: monitor blood pressure, "
                "fasting glucose/HbA1c, and lifestyle risk factors."
            ),
        }
    return {
        "tier": "LOW RISK",
        "color": "#2f855a",
        "bg": "#f0fff4",
        "action": (
            "Routine preventive care: maintain healthy blood pressure, glycemic "
            "control, balanced BMI, and tobacco cessation."
        ),
    }


def evaluate_clinical_flags(patient: dict[str, Any]) -> list[dict[str, str]]:
    """Evaluate WHO, ADA, and AHA risk thresholds for personalized patient guidance."""
    flags: list[dict[str, str]] = []
    age = float(patient["age"])
    glucose = float(patient["avg_glucose_level"])
    bmi = patient.get("bmi")

    if age >= 65:
        flags.append({"factor": "Age >= 65 yrs", "status": "Elevated", "detail": f"Age {age:.0f} yrs carries substantially elevated baseline stroke incidence."})
    elif age >= 45:
        flags.append({"factor": "Age 45–64 yrs", "status": "Moderate", "detail": f"Age {age:.0f} yrs enters mid-life vascular screening window."})

    if int(patient["hypertension"]) == 1:
        flags.append({"factor": "AHA Hypertension", "status": "Elevated", "detail": "Hypertension increases adjusted stroke odds (~1.39× in cohort Logit)."})

    if int(patient["heart_disease"]) == 1:
        flags.append({"factor": "Cardiac Comorbidity", "status": "Elevated", "detail": "Existing heart disease amplifies thromboembolic & ischemic risk."})

    if glucose >= 200:
        flags.append({"factor": "ADA Severe Hyperglycemia", "status": "Elevated", "detail": f"Avg glucose {glucose:.1f} mg/dL (>=200 mg/dL diabetic high-risk tier)."})
    elif glucose >= 126:
        flags.append({"factor": "ADA Hyperglycemia", "status": "Elevated", "detail": f"Avg glucose {glucose:.1f} mg/dL (>=126 mg/dL diabetic threshold)."})
    elif glucose >= 100:
        flags.append({"factor": "ADA Prediabetes", "status": "Moderate", "detail": f"Avg glucose {glucose:.1f} mg/dL (100–125 mg/dL impaired fasting range)."})

    if bmi is None or pd.isna(bmi):
        flags.append({"factor": "Unmeasured BMI", "status": "Alert", "detail": "Missing BMI in clinical records correlates with acute/non-ambulatory presentation."})
    elif float(bmi) >= 30.0:
        flags.append({"factor": "WHO Obesity", "status": "Moderate", "detail": f"BMI {float(bmi):.1f} kg/m² (>=30.0 WHO Obese category)."})
    elif float(bmi) >= 25.0:
        flags.append({"factor": "WHO Overweight", "status": "Watch", "detail": f"BMI {float(bmi):.1f} kg/m² (25.0–29.9 WHO Overweight range)."})

    if patient["smoking_status"] == "smokes":
        flags.append({"factor": "Active Smoking", "status": "Elevated", "detail": "Active tobacco use accelerates endothelial injury and stroke risk."})
    elif patient["smoking_status"] == "formerly smoked":
        flags.append({"factor": "Former Smoker", "status": "Watch", "detail": "Residual vascular risk persists post-cessation; continued monitoring advised."})

    return flags


def build_patient_dataframe(patient: dict[str, Any]) -> pd.DataFrame:
    """Convert a patient input dictionary into a single-row DataFrame matching pipeline input."""
    return pd.DataFrame([patient])


def compute_local_lr_contributions(
    lr_pipeline: Any,
    patient_df: pd.DataFrame,
    top_n: int = 12,
) -> pd.DataFrame:
    """Compute local linear log-odds contributions (transformed_x * beta) for a single patient."""
    cleaned = lr_pipeline.named_steps["cleaner"].transform(patient_df)
    engineered = lr_pipeline.named_steps["features"].transform(cleaned)
    transformed = lr_pipeline.named_steps["preprocessor"].transform(engineered)[0]
    feat_names = list(lr_pipeline.named_steps["preprocessor"].get_feature_names_out())
    coefs = lr_pipeline.named_steps["classifier"].coef_[0]

    contrib = transformed * coefs
    df_contrib = pd.DataFrame(
        {
            "feature": feat_names,
            "transformed_value": transformed,
            "coefficient": coefs,
            "log_odds_contribution": contrib,
            "abs_contribution": np.abs(contrib),
        }
    )
    # Show active or continuous features first
    active = df_contrib[df_contrib["abs_contribution"] > 1e-9].sort_values(
        "abs_contribution", ascending=False
    )
    if len(active) < top_n:
        active = df_contrib.sort_values("abs_contribution", ascending=False)
    return active.head(top_n).reset_index(drop=True)


@st.cache_resource
def load_models() -> tuple[Any, Any]:
    """Load serialized Logistic Regression and Best Ensemble pipelines."""
    lr_pipe = joblib.load(LR_MODEL_PATH)
    ens_pipe = joblib.load(BEST_MODEL_PATH)
    return lr_pipe, ens_pipe


@st.cache_data
def load_metadata() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load model metrics JSON and statistical summary JSON."""
    metrics = json.loads(METRICS_PATH.read_text())
    stats_path = FIGURES_DIR / "statistical_summary.json"
    stats_data = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    return metrics, stats_data


@st.cache_data
def load_eda_dataframe() -> pd.DataFrame | None:
    """Load cleaned EDA dataset if present locally (optional for gitignored environments)."""
    if CLEANED_EDA_PATH.exists():
        return pd.read_csv(CLEANED_EDA_PATH)
    return None


def make_gauge_chart(prob_pct: float, tier_color: str, low_pct: float, high_pct: float) -> go.Figure:
    """Create Plotly risk probability gauge chart."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            number={"suffix": "%", "font": {"size": 36, "color": tier_color}},
            title={"text": "Estimated Stroke Risk Score", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": tier_color, "thickness": 0.30},
                "bgcolor": "white",
                "steps": [
                    {"range": [0, low_pct], "color": "#c6f6d5"},
                    {"range": [low_pct, high_pct], "color": "#fefcbf"},
                    {"range": [high_pct, 100], "color": "#fed7d7"},
                ],
                "threshold": {
                    "line": {"color": "#1a202c", "width": 3},
                    "thickness": 0.8,
                    "value": low_pct,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=45, b=10))
    return fig


def main() -> None:
    """Render the Streamlit Clinical Triage & Stroke Risk Screening application."""
    st.set_page_config(
        page_title="Stroke Risk Clinical Triage App",
        page_icon="🩺",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .triage-card {
            padding: 1.1rem 1.3rem;
            border-radius: 0.65rem;
            border-left: 6px solid;
            margin-bottom: 1rem;
        }
        .disclaimer-box {
            background-color: #edf2f7;
            border-left: 4px solid #2b6cb0;
            padding: 0.75rem 1rem;
            border-radius: 0.4rem;
            font-size: 0.88rem;
            color: #2d3748;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    lr_pipe, ens_pipe = load_models()
    metrics, stats_data = load_metadata()
    eda_df = load_eda_dataframe()

    best_ens_name = metrics.get("best_ensemble_model", "Random Forest")
    model_options = {
        "Logistic Regression (Interpretable)": ("Logistic Regression", lr_pipe),
        f"Best Ensemble Model ({best_ens_name})": (best_ens_name, ens_pipe),
    }

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.markdown("### 🩺 Stroke Clinical Triage")
        st.info("**Project #49**\nVaddi Varshitha (`S230617`)\nIDS (`25A12104`) — AIML-2A")

        selected_label = st.selectbox("Select Screening Model", list(model_options.keys()))
        model_key, active_pipe = model_options[selected_label]

        calib = metrics["triage_thresholds"].get(
            model_key, {"low_risk_max": 0.50, "high_risk_min": 0.70}
        )
        default_low = float(calib["low_risk_max"])
        default_high = float(calib["high_risk_min"])

        low_thresh = st.slider(
            "Screening Sensitivity Cutoff (Low -> Moderate)",
            min_value=0.15,
            max_value=0.70,
            value=default_low,
            step=0.01,
            help="Lowering this threshold catches more stroke cases (higher Recall) at the cost of more follow-up alerts.",
        )
        high_thresh = st.slider(
            "Priority Alert Cutoff (Moderate -> High Risk)",
            min_value=min(low_thresh + 0.05, 0.85),
            max_value=0.90,
            value=max(default_high, min(low_thresh + 0.15, 0.90)),
            step=0.01,
        )

        st.markdown(
            """
            <div class="disclaimer-box">
            <strong>Clinical Decision-Support Disclaimer:</strong><br>
            This tool is an academic Data Science screening prototype built on retrospective hospital records. It is <strong>not a medical diagnosis</strong> and must not replace licensed clinical evaluation.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.title("Stroke Risk Factors & Clinical Triage Screening App")
    st.caption(
        "Imbalance-aware stroke screening powered by age/gender-stratified imputation, "
        "WHO/ADA/AHA clinical features, and calibrated triage thresholds."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "1. Clinical Triage Screener",
            "2. Patient Risk Explainability",
            "3. EDA & Statistical Evidence",
            "4. Model Benchmarks & Threshold Lab",
        ]
    )

    # ---------------- Tab 1: Live Screener ----------------
    with tab1:
        st.subheader("Patient Clinical Intake Form")
        col_form, col_out = st.columns([1.15, 1.0], gap="large")

        with col_form:
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age (years)", min_value=0.1, max_value=100.0, value=64.0, step=1.0)
                gender = st.selectbox("Gender", ["Female", "Male", "Other"])
                hypertension = st.selectbox(
                    "Hypertension History",
                    options=[0, 1],
                    index=1,
                    format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
                )
                heart_disease = st.selectbox(
                    "Heart Disease History",
                    options=[0, 1],
                    index=0,
                    format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
                )
                ever_married = st.selectbox("Ever Married", ["Yes", "No"])

            with c2:
                avg_glucose_level = st.number_input(
                    "Average Glucose Level (mg/dL)",
                    min_value=40.0,
                    max_value=350.0,
                    value=148.0,
                    step=1.0,
                    help="ADA thresholds: <100 Normal, 100–125 Prediabetes, >=126 Diabetic range.",
                )
                bmi_unknown = st.checkbox("BMI unmeasured / unavailable at intake", value=False)
                bmi_val = st.number_input(
                    "BMI (kg/m²)",
                    min_value=10.0,
                    max_value=70.0,
                    value=29.5,
                    step=0.5,
                    disabled=bmi_unknown,
                )
                smoking_status = st.selectbox(
                    "Smoking Status",
                    ["never smoked", "formerly smoked", "smokes", "Unknown"],
                )
                work_type = st.selectbox(
                    "Work Type",
                    ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
                )
                residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])

        patient_dict = {
            "gender": gender,
            "age": float(age),
            "hypertension": int(hypertension),
            "heart_disease": int(heart_disease),
            "ever_married": ever_married,
            "work_type": work_type,
            "Residence_type": residence_type,
            "avg_glucose_level": float(avg_glucose_level),
            "bmi": np.nan if bmi_unknown else float(bmi_val),
            "smoking_status": smoking_status,
        }
        patient_df = build_patient_dataframe(patient_dict)
        prob = float(active_pipe.predict_proba(patient_df)[0, 1])
        tier = classify_triage_tier(prob, low_thresh, high_thresh)
        clinical_flags = evaluate_clinical_flags(patient_dict)

        with col_out:
            fig_gauge = make_gauge_chart(
                prob * 100.0, tier["color"], low_thresh * 100.0, high_thresh * 100.0
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown(
                f"""
                <div class="triage-card" style="background-color:{tier['bg']}; border-color:{tier['color']};">
                    <h3 style="margin:0; color:{tier['color']};">{tier['tier']} ({prob:.1%} Risk Score)</h3>
                    <p style="margin:0.4rem 0 0 0; color:#1a202c;">{tier['action']}</p>
                    <small style="color:#4a5568;">Model: {selected_label} | Cutoffs: Low &lt; {low_thresh:.0%}, High &ge; {high_thresh:.0%}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### Personalized WHO / ADA / AHA Risk Indicators")
            if clinical_flags:
                for fl in clinical_flags:
                    st.markdown(f"- **{fl['factor']} ({fl['status']}):** {fl['detail']}")
            else:
                st.success("No elevated WHO/ADA/AHA clinical thresholds triggered for this profile.")

    # ---------------- Tab 2: Explainability ----------------
    with tab2:
        st.subheader("Local Patient Risk Explainability")
        st.write(
            "Local log-odds contributions computed from the interpretable Logistic Regression pipeline "
            "(`standardized feature value × learned β coefficient`). Positive red bars push estimated risk up; "
            "negative blue bars reduce estimated risk relative to the cohort average."
        )

        contrib_df = compute_local_lr_contributions(lr_pipe, patient_df, top_n=12)
        lr_prob = float(lr_pipe.predict_proba(patient_df)[0, 1])
        st.metric("Interpretable Logistic Regression Probability", f"{lr_prob:.1%}")

        plot_df = contrib_df.iloc[::-1].copy()
        plot_df["direction"] = np.where(
            plot_df["log_odds_contribution"] >= 0, "Increases Risk", "Decreases Risk"
        )
        fig_exp = px.bar(
            plot_df,
            x="log_odds_contribution",
            y="feature",
            color="direction",
            orientation="h",
            color_discrete_map={"Increases Risk": "#c53030", "Decreases Risk": "#2b6cb0"},
            title="Top Local Risk Drivers for Current Patient",
            labels={"log_odds_contribution": "Log-Odds Contribution", "feature": "Engineered Feature"},
        )
        fig_exp.add_vline(x=0, line_dash="dash", line_color="#1a202c")
        fig_exp.update_layout(height=430, legend_title_text="Effect Direction")
        st.plotly_chart(fig_exp, use_container_width=True)

        if stats_data.get("logit_odds_ratios"):
            st.markdown("#### Cohort-Wide Adjusted Odds Ratios (Statsmodels Logit, n = 5,109)")
            or_table = pd.DataFrame(stats_data["logit_odds_ratios"])
            st.dataframe(
                or_table[["predictor", "odds_ratio", "ci_lower_95", "ci_upper_95", "p_value", "significant"]]
                .round({"odds_ratio": 3, "ci_lower_95": 3, "ci_upper_95": 3, "p_value": 5}),
                use_container_width=True,
                hide_index=True,
            )

    # ---------------- Tab 3: EDA & Statistical Evidence ----------------
    with tab3:
        st.subheader("Exploratory Data Analysis & Hypothesis Testing Evidence")
        if eda_df is not None:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Cleaned Cohort Size", f"{len(eda_df):,}")
            m2.metric("Stroke Cases", f"{int(eda_df['stroke'].sum())} ({eda_df['stroke'].mean():.2%})")
            m3.metric("Hypertension Stroke Rate", f"{eda_df.loc[eda_df['hypertension'] == 1, 'stroke'].mean():.1%}")
            m4.metric("Missing-BMI Stroke Rate", f"{eda_df.loc[eda_df['bmi_was_missing'] == 1, 'stroke'].mean():.1%}")

            c_left, c_right = st.columns(2)
            with c_left:
                rate_df = (
                    eda_df.assign(
                        htn=eda_df["hypertension"].map({0: "No HTN", 1: "Hypertension"}),
                        gluc=np.where(eda_df["avg_glucose_level"] >= 126, "Glucose >=126", "Glucose <126"),
                    )
                    .groupby(["htn", "gluc"])["stroke"]
                    .mean()
                    .mul(100)
                    .reset_index()
                )
                fig_htn = px.bar(
                    rate_df,
                    x="htn",
                    y="stroke",
                    color="gluc",
                    barmode="group",
                    title="Stroke Rate (%) by Hypertension × ADA Hyperglycemia",
                    labels={"htn": "Hypertension Status", "stroke": "Stroke Prevalence (%)", "gluc": "Glucose Tier"},
                    color_discrete_sequence=["#4299e1", "#c53030"],
                )
                st.plotly_chart(fig_htn, use_container_width=True)

            with c_right:
                fig_age = px.box(
                    eda_df.assign(outcome=eda_df["stroke"].map({0: "No Stroke", 1: "Stroke"})),
                    x="outcome",
                    y="age",
                    color="outcome",
                    title="Age Distribution by Stroke Outcome (Medians: 43 vs. 71 yrs)",
                    labels={"outcome": "", "age": "Age (Years)"},
                    color_discrete_map={"No Stroke": "#4299e1", "Stroke": "#c53030"},
                )
                st.plotly_chart(fig_age, use_container_width=True)
        else:
            st.info("Showing saved 300-DPI EDA figures (cleaned CSV is gitignored locally).")
            fig_a, fig_b = st.columns(2)
            fig_a.image(str(FIGURES_DIR / "03_stroke_rate_by_risk_factors.png"), use_container_width=True)
            fig_b.image(str(FIGURES_DIR / "06_hypertension_glucose_interaction.png"), use_container_width=True)

        st.markdown("#### Chi-Square Tests of Independence (α = 0.05)")
        if stats_data.get("chi_square_results"):
            st.dataframe(
                pd.DataFrame(stats_data["chi_square_results"]).round(
                    {"chi2": 2, "p_value": 6, "cramers_v": 4, "min_expected_cell": 2}
                ),
                use_container_width=True,
                hide_index=True,
            )

        if stats_data.get("continuous_group_tests"):
            st.markdown("#### Continuous Feature Tests (Mann-Whitney U & Welch's t-test)")
            st.dataframe(
                pd.DataFrame(stats_data["continuous_group_tests"])[
                    [
                        "feature",
                        "non_stroke_median",
                        "stroke_median",
                        "mann_whitney_p",
                        "welch_p",
                        "decision",
                    ]
                ].round({"non_stroke_median": 2, "stroke_median": 2, "mann_whitney_p": 6, "welch_p": 6}),
                use_container_width=True,
                hide_index=True,
            )

    # ---------------- Tab 4: Model Benchmarks & Threshold Lab ----------------
    with tab4:
        st.subheader("5-Model Benchmark Leaderboard (Stratified 5-Fold CV & Test Set)")
        lb_rows = []
        for row in metrics["leaderboard"]:
            cal = row["test_calibrated"]
            lb_rows.append(
                {
                    "Model": row["model"],
                    "CV ROC-AUC": row["cv_roc_auc"],
                    "CV PR-AUC": row["cv_pr_auc"],
                    "Test ROC-AUC": row["test_roc_auc"],
                    "Test PR-AUC": row["test_pr_auc"],
                    "Calibrated Thresh": row["calibrated_screen_thresh"],
                    "Recall (Sens.)": cal["recall_sensitivity"],
                    "Specificity": cal["specificity"],
                    "Precision": cal["precision"],
                    "F1-Score": cal["f1_score"],
                }
            )
        st.dataframe(
            pd.DataFrame(lb_rows).sort_values("CV ROC-AUC", ascending=False).round(4),
            use_container_width=True,
            hide_index=True,
        )

        # Interactive ROC & PR Plotly Curves
        c_roc, c_pr = st.columns(2)
        fig_roc = go.Figure()
        fig_pr = go.Figure()
        for mname, cdata in metrics["curve_data"].items():
            fig_roc.add_trace(go.Scatter(x=cdata["fpr"], y=cdata["tpr"], mode="lines", name=mname))
            fig_pr.add_trace(
                go.Scatter(x=cdata["pr_recall"], y=cdata["pr_precision"], mode="lines", name=mname)
            )
        fig_roc.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random (0.5)", line=dict(dash="dash", color="black"))
        )
        fig_roc.update_layout(title="Test-Set ROC Curves", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate (Recall)", height=380)
        fig_pr.update_layout(title="Test-Set Precision-Recall Curves", xaxis_title="Recall", yaxis_title="Precision", height=380)
        c_roc.plotly_chart(fig_roc, use_container_width=True)
        c_pr.plotly_chart(fig_pr, use_container_width=True)

        # Threshold Sensitivity Lab
        st.markdown("#### Interactive Clinical Threshold Trade-Off Lab")
        sweep_data = metrics["threshold_sweeps"].get(model_key) or metrics["threshold_sweeps"]["Logistic Regression"]
        sweep_df = pd.DataFrame(sweep_data)
        closest_row = sweep_df.iloc[(sweep_df["threshold"] - low_thresh).abs().argsort()[:1]].iloc[0]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Active Threshold", f"{closest_row['threshold']:.2f}")
        k2.metric("Test Recall (Sensitivity)", f"{closest_row['recall_sensitivity']:.1%}")
        k3.metric("Test Specificity", f"{closest_row['specificity']:.1%}")
        k4.metric("False Negatives / Positives", f"{int(closest_row['fn'])} FN / {int(closest_row['fp'])} FP")

        fig_sweep = go.Figure()
        fig_sweep.add_trace(go.Scatter(x=sweep_df["threshold"], y=sweep_df["recall_sensitivity"], mode="lines+markers", name="Sensitivity (Recall)", line=dict(color="#c53030")))
        fig_sweep.add_trace(go.Scatter(x=sweep_df["threshold"], y=sweep_df["specificity"], mode="lines+markers", name="Specificity", line=dict(color="#2b6cb0")))
        fig_sweep.add_trace(go.Scatter(x=sweep_df["threshold"], y=sweep_df["f1_score"], mode="lines+markers", name="F1-Score", line=dict(color="#2f855a")))
        fig_sweep.add_vline(x=low_thresh, line_dash="dash", line_color="#1a202c")
        fig_sweep.update_layout(
            title=f"Threshold Trade-Off Curve ({model_key})",
            xaxis_title="Decision Threshold",
            yaxis_title="Score",
            height=340,
        )
        st.plotly_chart(fig_sweep, use_container_width=True)

        # Batch CSV Screening Uploader
        st.markdown("#### Batch Patient CSV Triage Uploader")
        uploaded_csv = st.file_uploader("Upload patient CSV (matching raw input columns)", type=["csv"])
        if uploaded_csv is not None:
            batch_df = pd.read_csv(uploaded_csv)
            batch_probs = active_pipe.predict_proba(batch_df)[:, 1]
            scored_df = batch_df.copy()
            scored_df["stroke_risk_probability"] = np.round(batch_probs, 4)
            scored_df["triage_tier"] = [
                classify_triage_tier(float(p), low_thresh, high_thresh)["tier"] for p in batch_probs
            ]
            st.dataframe(scored_df.head(20), use_container_width=True)
            st.download_button(
                "Download Scored Triage CSV",
                data=scored_df.to_csv(index=False).encode("utf-8"),
                file_name="stroke_triage_scored_results.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()

