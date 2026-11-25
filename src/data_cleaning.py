"""Reproducible data auditing, missing-value imputation, and clinical cleaning utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import (
    AGE_BINS,
    AGE_LABELS,
    BMI_CAP,
    CLEANED_EDA_PATH,
    DATA_RAW_PATH,
    PEDIATRIC_SMOKING_AGE_CUTOFF,
)


def load_raw_data(path: Path | str = DATA_RAW_PATH) -> pd.DataFrame:
    """Load the raw Kaggle stroke dataset from CSV."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {csv_path}. "
            "Download healthcare-dataset-stroke-data.csv into data/raw/."
        )
    return pd.read_csv(csv_path)


def audit_raw_data(df: pd.DataFrame) -> dict[str, Any]:
    """Compute a structured quality audit of the raw stroke dataset."""
    bmi_missing = df["bmi"].isna()
    stroke_rate_bmi_present = float(df.loc[~bmi_missing, "stroke"].mean())
    stroke_rate_bmi_missing = float(df.loc[bmi_missing, "stroke"].mean())

    unknown_mask = df["smoking_status"] == "Unknown"
    pediatric_mask = df["age"] < PEDIATRIC_SMOKING_AGE_CUTOFF

    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_ids": int(df["id"].nunique()) if "id" in df.columns else None,
        "missing_counts": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
        "gender_counts": df["gender"].value_counts().to_dict(),
        "stroke_counts": df["stroke"].value_counts().to_dict(),
        "stroke_rate": float(df["stroke"].mean()),
        "bmi_outliers_gt_60": int((df["bmi"] > 60).sum()),
        "bmi_max": float(df["bmi"].max()),
        "stroke_rate_when_bmi_present": stroke_rate_bmi_present,
        "stroke_rate_when_bmi_missing": stroke_rate_bmi_missing,
        "smoking_unknown_total": int(unknown_mask.sum()),
        "smoking_unknown_pediatric_lt12": int((unknown_mask & pediatric_mask).sum()),
        "pediatric_lt12_total": int(pediatric_mask.sum()),
        "smoking_unknown_adult_ge12": int((unknown_mask & ~pediatric_mask).sum()),
    }


def _assign_age_band(age_series: pd.Series) -> pd.Series:
    """Bin continuous age into clinical age bands for stratified imputation."""
    return pd.cut(
        age_series,
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=False,
        include_lowest=True,
    ).astype(str)


class ClinicalDataCleaner(BaseEstimator, TransformerMixin):
    """Leakage-free clinical cleaner and imputer for the Stroke Prediction dataset.

    Steps applied:
    1. Drop non-predictive `id` column if present.
    2. Handle rare `gender == 'Other'` (map to modal gender during inference;
       filter rows beforehand for training/EDA via `drop_other_gender_rows`).
    3. Create missingness indicator flags `bmi_was_missing` and `smoking_was_unknown`.
    4. Domain-aware smoking imputation: children `< 12` with `'Unknown'` become
       `'never smoked'`; adult `'Unknown'` (`>= 12`) can be retained (`'keep'`)
       or imputed using training-fit age-band & gender modes (`'mode'`).
    5. Age-band & gender-stratified median BMI imputation (fit only on training rows)
       plus upper winsorization at `bmi_cap`.
    """

    def __init__(
        self,
        bmi_cap: float | None = BMI_CAP,
        pediatric_age_cutoff: float = PEDIATRIC_SMOKING_AGE_CUTOFF,
        adult_unknown_strategy: str = "keep",
        map_other_gender_to: str = "Female",
    ) -> None:
        self.bmi_cap = bmi_cap
        self.pediatric_age_cutoff = pediatric_age_cutoff
        self.adult_unknown_strategy = adult_unknown_strategy
        self.map_other_gender_to = map_other_gender_to

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ClinicalDataCleaner":
        """Learn stratified BMI medians and optional adult smoking modes from training data."""
        df = X.copy()
        if "gender" in df.columns:
            df["gender"] = df["gender"].replace({"Other": self.map_other_gender_to})

        df["_age_band"] = _assign_age_band(df["age"])

        # Stratified BMI medians by (age_band, gender), with age_band & global fallbacks
        self.global_bmi_median_ = float(df["bmi"].median())
        self.age_bmi_medians_ = df.groupby("_age_band")["bmi"].median().to_dict()
        self.stratified_bmi_medians_ = (
            df.groupby(["_age_band", "gender"])["bmi"].median().to_dict()
        )

        # Optional age-band & gender modal smoking status for adults >= cutoff
        known_adults = df[
            (df["age"] >= self.pediatric_age_cutoff)
            & (df["smoking_status"] != "Unknown")
        ]
        if not known_adults.empty:
            self.global_adult_smoking_mode_ = str(
                known_adults["smoking_status"].mode().iloc[0]
            )
            self.stratified_smoking_modes_ = (
                known_adults.groupby(["_age_band", "gender"])["smoking_status"]
                .agg(lambda s: s.mode().iloc[0])
                .to_dict()
            )
        else:
            self.global_adult_smoking_mode_ = "never smoked"
            self.stratified_smoking_modes_ = {}

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply learned clinical cleaning rules without modifying the input frame."""
        df = X.copy()
        if "id" in df.columns:
            df = df.drop(columns=["id"])

        if "gender" in df.columns:
            df["gender"] = df["gender"].replace({"Other": self.map_other_gender_to})

        # Preserve informative missingness signals
        df["bmi_was_missing"] = df["bmi"].isna().astype(int)
        df["smoking_was_unknown"] = (df["smoking_status"] == "Unknown").astype(int)

        age_bands = _assign_age_band(df["age"])

        # Domain rule 1: pediatric Unknown (<12) -> never smoked
        pediatric_unknown = (df["smoking_status"] == "Unknown") & (
            df["age"] < self.pediatric_age_cutoff
        )
        df.loc[pediatric_unknown, "smoking_status"] = "never smoked"

        # Domain rule 2: adult Unknown (>=12)
        if self.adult_unknown_strategy == "mode":
            adult_unknown = (df["smoking_status"] == "Unknown") & (
                df["age"] >= self.pediatric_age_cutoff
            )
            if adult_unknown.any():
                keys = list(zip(age_bands[adult_unknown], df.loc[adult_unknown, "gender"]))
                imputed_smoking = [
                    self.stratified_smoking_modes_.get(k, self.global_adult_smoking_mode_)
                    for k in keys
                ]
                df.loc[adult_unknown, "smoking_status"] = imputed_smoking
        elif self.adult_unknown_strategy != "keep":
            raise ValueError(
                f"Unsupported adult_unknown_strategy: {self.adult_unknown_strategy}"
            )

        # Age-band & gender-stratified median BMI imputation
        bmi_missing_mask = df["bmi"].isna()
        if bmi_missing_mask.any():
            imputed_bmis = []
            for band, gen in zip(
                age_bands[bmi_missing_mask], df.loc[bmi_missing_mask, "gender"]
            ):
                val = self.stratified_bmi_medians_.get(
                    (band, gen),
                    self.age_bmi_medians_.get(band, self.global_bmi_median_),
                )
                if pd.isna(val):
                    val = self.global_bmi_median_
                imputed_bmis.append(float(val))
            df.loc[bmi_missing_mask, "bmi"] = imputed_bmis

        # Winsorize implausible extreme BMI outliers if configured
        if self.bmi_cap is not None:
            df["bmi"] = df["bmi"].clip(upper=self.bmi_cap)

        # Enforce clean numeric dtypes
        df["age"] = df["age"].astype(float)
        df["avg_glucose_level"] = df["avg_glucose_level"].astype(float)
        df["bmi"] = df["bmi"].astype(float)
        df["hypertension"] = df["hypertension"].astype(int)
        df["heart_disease"] = df["heart_disease"].astype(int)
        if "stroke" in df.columns:
            df["stroke"] = df["stroke"].astype(int)

        return df


def drop_other_gender_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude the single rare 'Other' gender row (0.02%) for statistical modeling."""
    return df[df["gender"] != "Other"].reset_index(drop=True)


def clean_for_eda(
    raw_path: Path | str = DATA_RAW_PATH,
    save_path: Path | str | None = CLEANED_EDA_PATH,
    adult_unknown_strategy: str = "keep",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load raw data, run audit, apply clinical cleaning, and export EDA dataset."""
    raw_df = load_raw_data(raw_path)
    audit = audit_raw_data(raw_df)

    filtered_df = drop_other_gender_rows(raw_df)
    cleaner = ClinicalDataCleaner(adult_unknown_strategy=adult_unknown_strategy)
    cleaned_df = cleaner.fit_transform(filtered_df)

    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(out_path, index=False)

    return cleaned_df, audit


if __name__ == "__main__":
    cleaned, raw_audit = clean_for_eda()
    assert cleaned.isna().sum().sum() == 0, "Unexpected NaNs remain after cleaning!"
    assert cleaned.shape[0] == 5109, f"Expected 5109 rows, got {cleaned.shape[0]}"
    assert "id" not in cleaned.columns, "id column should be dropped"
    assert "Other" not in cleaned["gender"].values, "Other gender row should be removed"
    assert cleaned["bmi"].max() <= BMI_CAP, "BMI cap exceeded"
    print("Module 1 cleaning unit checks PASSED!")
    print("Raw audit summary:", raw_audit)
    print("Cleaned shape:", cleaned.shape)
    print("Smoking status distribution after cleaning:\n", cleaned["smoking_status"].value_counts())
