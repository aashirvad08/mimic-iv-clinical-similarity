from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from .ml_dataset import load_joined_cdss_dataset, pick_outcome_column
    from .similarity import jaccard_similarity, score_all_patients
except ImportError:  # pragma: no cover
    from ml_dataset import load_joined_cdss_dataset, pick_outcome_column
    from similarity import jaccard_similarity, score_all_patients


@dataclass(frozen=True)
class Phase1ValidationSummary:
    row_count: int
    available_outcomes: tuple[str, ...]
    mortality_rate: float | None
    readmission_rate: float | None
    average_diagnosis_count: float
    average_treatment_event_count: float | None
    list_columns_are_python_lists: bool
    quick_similarity_min: float
    quick_similarity_max: float
    quick_similarity_mean: float


@dataclass(frozen=True)
class Phase2EvaluationSummary:
    outcome_column: str
    sampled_patients: int
    pair_count: int
    same_version_only: bool
    correlation: float
    mean_similarity: float
    mean_outcome_match_rate: float


def summarize_phase1_dataset(
    df: pd.DataFrame,
    query_hadm_id: int | None = None,
    same_version_only: bool = False,
) -> Phase1ValidationSummary:
    if df.empty:
        raise ValueError("Joined dataframe is empty.")

    query_id = int(df["hadm_id"].iloc[0] if query_hadm_id is None else query_hadm_id)
    similarity_scores = score_all_patients(
        df=df,
        query_hadm_id=query_id,
        same_version_only=same_version_only,
        exclude_self=False,
    )["similarity_score"]

    mortality_rate = float(df["mortality"].mean()) if "mortality" in df.columns else None
    readmission_rate = float(df["readmission_30day"].mean()) if "readmission_30day" in df.columns else None
    avg_treatment_event_count = (
        float(df["treatment_event_count"].mean()) if "treatment_event_count" in df.columns else None
    )

    return Phase1ValidationSummary(
        row_count=int(len(df)),
        available_outcomes=tuple(column for column in ("mortality", "readmission_30day", "los_days") if column in df.columns),
        mortality_rate=mortality_rate,
        readmission_rate=readmission_rate,
        average_diagnosis_count=float(df["diagnosis_count"].mean()),
        average_treatment_event_count=avg_treatment_event_count,
        list_columns_are_python_lists=bool(df["diagnoses_icd_list"].map(lambda value: isinstance(value, list)).all()),
        quick_similarity_min=float(similarity_scores.min()),
        quick_similarity_max=float(similarity_scores.max()),
        quick_similarity_mean=float(similarity_scores.mean()),
    )


def run_phase1_validation(
    db_path: str | None = None,
    join_how: str = "inner",
    include_treatment_features: bool = True,
    same_version_only: bool = False,
) -> tuple[pd.DataFrame, Phase1ValidationSummary]:
    df = load_joined_cdss_dataset(
        db_path=db_path,
        join_how=join_how,
        include_treatment_features=include_treatment_features,
        prepare_sets=True,
    )
    summary = summarize_phase1_dataset(df, same_version_only=same_version_only)
    return df, summary


def evaluate_similarity_outcome_alignment(
    df: pd.DataFrame,
    outcome_column: str | None = None,
    sample_size: int = 1000,
    random_state: int = 42,
    same_version_only: bool = False,
    bins: int = 5,
) -> tuple[Phase2EvaluationSummary, pd.DataFrame]:
    if bins < 1:
        raise ValueError("bins must be >= 1")

    active_outcome = pick_outcome_column(df, preferred=outcome_column)
    eligible = df[df[active_outcome].notna()].copy()
    if eligible.empty:
        raise ValueError(f"No rows available with outcome column '{active_outcome}'.")

    sampled = eligible.sample(n=min(sample_size, len(eligible)), random_state=random_state).reset_index(drop=True)
    dx_sets = sampled["dx_set"].tolist()
    outcomes = sampled[active_outcome].to_numpy(dtype=float if active_outcome == "los_days" else object)
    versions = sampled["icd_version_mix"].to_numpy(dtype=object)

    similarities: list[float] = []
    outcome_matches: list[float] = []

    for i in range(len(sampled) - 1):
        base_set = dx_sets[i]
        base_outcome = outcomes[i]
        base_version = versions[i]

        for j in range(i + 1, len(sampled)):
            if same_version_only and base_version != versions[j]:
                continue

            similarities.append(jaccard_similarity(base_set, dx_sets[j]))

            if active_outcome == "los_days":
                outcome_matches.append(float(np.exp(-0.05 * abs(float(base_outcome) - float(outcomes[j])))))
            else:
                outcome_matches.append(float(base_outcome == outcomes[j]))

    if not similarities:
        raise ValueError("No valid patient pairs were available for similarity evaluation.")

    pair_df = pd.DataFrame(
        {
            "similarity": np.asarray(similarities, dtype=float),
            "outcome_match": np.asarray(outcome_matches, dtype=float),
        }
    )
    edges = np.linspace(0.0, 1.0, num=bins + 1)
    pair_df["sim_band"] = pd.cut(pair_df["similarity"], bins=edges, include_lowest=True)
    band_summary = (
        pair_df.groupby("sim_band", observed=False)
        .agg(
            pair_count=("outcome_match", "size"),
            similarity_mean=("similarity", "mean"),
            outcome_match_rate=("outcome_match", "mean"),
        )
        .reset_index()
    )

    correlation = float(pair_df["similarity"].corr(pair_df["outcome_match"]))
    summary = Phase2EvaluationSummary(
        outcome_column=active_outcome,
        sampled_patients=int(len(sampled)),
        pair_count=int(len(pair_df)),
        same_version_only=same_version_only,
        correlation=correlation,
        mean_similarity=float(pair_df["similarity"].mean()),
        mean_outcome_match_rate=float(pair_df["outcome_match"].mean()),
    )
    return summary, band_summary
