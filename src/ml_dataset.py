from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from .data_loader import (
        SELECT_COLUMNS as DIAGNOSIS_COLUMNS,
        load_cdss_diagnoses,
        prepare_similarity_columns,
        resolve_db_path,
        connect_duckdb,
    )
except ImportError:  # pragma: no cover
    from data_loader import (
        SELECT_COLUMNS as DIAGNOSIS_COLUMNS,
        load_cdss_diagnoses,
        prepare_similarity_columns,
        resolve_db_path,
        connect_duckdb,
    )


BASE_REQUIRED_COLUMNS = (
    "hadm_id",
    "subject_id",
    "age",
    "gender",
    "race",
    "los_days",
    "mortality",
)
BASE_OPTIONAL_COLUMNS = (
    "readmission_30day",
    "admission_type",
    "admission_location",
    "discharge_location",
    "marital_status",
    "insurance",
    "language",
    "height_inches",
    "weight_lbs",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
)
OUTCOME_COLUMNS = ("mortality", "readmission_30day", "los_days")


def get_table_columns(db_path: str | Path | None, table_name: str) -> list[str]:
    con = connect_duckdb(db_path=db_path, read_only=True)
    try:
        description = con.execute(f"DESCRIBE {table_name}").fetchdf()
    finally:
        con.close()
    return description["column_name"].tolist()


def _select_available_columns(available_columns: list[str], required: tuple[str, ...], optional: tuple[str, ...]) -> list[str]:
    missing_required = [column for column in required if column not in available_columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")
    return [column for column in (*required, *optional) if column in available_columns]


def load_cdss_base(db_path: str | Path | None = None) -> pd.DataFrame:
    available_columns = get_table_columns(db_path=db_path, table_name="cdss_base")
    selected_columns = _select_available_columns(available_columns, BASE_REQUIRED_COLUMNS, BASE_OPTIONAL_COLUMNS)
    query = f"SELECT {', '.join(selected_columns)} FROM cdss_base"

    con = connect_duckdb(db_path=db_path, read_only=True)
    try:
        df = con.execute(query).fetchdf()
    finally:
        con.close()

    if not df["hadm_id"].is_unique:
        duplicate_count = int(df["hadm_id"].duplicated().sum())
        raise ValueError(f"cdss_base.hadm_id must be unique. Found {duplicate_count} duplicate rows.")

    return df.set_index("hadm_id", drop=False)


def load_cdss_treatment_summary(db_path: str | Path | None = None) -> pd.DataFrame:
    query = """
    SELECT
        hadm_id,
        COUNT(*) AS treatment_row_count,
        COALESCE(SUM(treatment_count), COUNT(*)) AS treatment_event_count,
        COUNT(DISTINCT treatment_key) AS unique_treatment_count,
        COUNT(DISTINCT treatment_source) AS unique_treatment_source_count
    FROM cdss_treatment
    GROUP BY hadm_id
    """

    con = connect_duckdb(db_path=db_path, read_only=True)
    try:
        df = con.execute(query).fetchdf()
    finally:
        con.close()

    if df.empty:
        return pd.DataFrame(
            columns=[
                "hadm_id",
                "treatment_row_count",
                "treatment_event_count",
                "unique_treatment_count",
                "unique_treatment_source_count",
            ]
        ).set_index("hadm_id", drop=False)

    return df.set_index("hadm_id", drop=False)


def load_joined_cdss_dataset(
    db_path: str | Path | None = None,
    join_how: str = "inner",
    include_treatment_features: bool = True,
    prepare_sets: bool = True,
) -> pd.DataFrame:
    resolved_db_path = resolve_db_path(db_path)
    base_df = load_cdss_base(resolved_db_path)
    diagnoses_df = load_cdss_diagnoses(resolved_db_path, validate=True, prepare_sets=False)

    diagnosis_columns = [column for column in DIAGNOSIS_COLUMNS if column != "subject_id"]

    df = base_df.reset_index(drop=True).merge(
        diagnoses_df[diagnosis_columns].reset_index(drop=True),
        on="hadm_id",
        how=join_how,
    )

    if "diagnoses_icd_list" in df.columns:
        df["diagnoses_icd_list"] = df["diagnoses_icd_list"].apply(lambda value: value if isinstance(value, list) else [])
    if "diagnoses_3digit_list" in df.columns:
        df["diagnoses_3digit_list"] = df["diagnoses_3digit_list"].apply(lambda value: value if isinstance(value, list) else [])
    if "primary_diagnosis_icd" in df.columns:
        df["primary_diagnosis_icd"] = df["primary_diagnosis_icd"].where(df["primary_diagnosis_icd"].notna(), None)
    if "primary_icd_3digit" in df.columns:
        df["primary_icd_3digit"] = df["primary_icd_3digit"].where(df["primary_icd_3digit"].notna(), None)
    if "icd_version_mix" in df.columns:
        df["icd_version_mix"] = df["icd_version_mix"].fillna("unknown")
    for column in ("diagnosis_count", "unique_icd_count"):
        if column in df.columns:
            df[column] = df[column].fillna(0).astype("int64")
    if "diagnosis_diversity_ratio" in df.columns:
        df["diagnosis_diversity_ratio"] = df["diagnosis_diversity_ratio"].fillna(0.0)

    df = df.set_index("hadm_id", drop=False)

    if include_treatment_features:
        treatment_df = load_cdss_treatment_summary(resolved_db_path)
        df = df.reset_index(drop=True).merge(
            treatment_df.reset_index(drop=True),
            on="hadm_id",
            how="left",
            suffixes=("", "_treatment"),
        ).set_index("hadm_id", drop=False)

        for column in (
            "treatment_row_count",
            "treatment_event_count",
            "unique_treatment_count",
            "unique_treatment_source_count",
        ):
            if column in df.columns:
                df[column] = df[column].fillna(0).astype("int64")

    if prepare_sets:
        prepare_similarity_columns(df)

    return df


def available_outcome_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in OUTCOME_COLUMNS if column in df.columns]


def pick_outcome_column(df: pd.DataFrame, preferred: str | None = None) -> str:
    outcomes = available_outcome_columns(df)
    if preferred is not None:
        if preferred not in outcomes:
            raise ValueError(f"Outcome column '{preferred}' not available. Available: {outcomes}")
        return preferred
    if "mortality" in outcomes:
        return "mortality"
    if outcomes:
        return outcomes[0]
    raise ValueError("No supported outcome columns available in dataframe.")


def build_phase3_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    features["age"] = df["age"].fillna(df["age"].median())
    features["elderly"] = (features["age"] > 65).astype("int8")
    features["diagnosis_count"] = df["diagnosis_count"].fillna(0).astype("int64")
    features["high_diagnosis_burden"] = (features["diagnosis_count"] > 10).astype("int8")
    features["diagnosis_diversity_ratio"] = df["diagnosis_diversity_ratio"].fillna(0.0)

    for column in (
        "treatment_row_count",
        "treatment_event_count",
        "unique_treatment_count",
        "unique_treatment_source_count",
    ):
        if column in df.columns:
            features[column] = df[column].fillna(0).astype("int64")

    return features
