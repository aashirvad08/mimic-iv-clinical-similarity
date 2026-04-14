from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.data_loader import load_cdss_diagnoses, validate_cdss_diagnoses
from src.similarity import baseline_patient_similarity, find_similar_patients, jaccard_similarity


def build_test_db(db_path: Path) -> None:
    con = duckdb.connect(str(db_path))
    diagnoses_df = pd.DataFrame(
        [
            {
                "hadm_id": 101,
                "subject_id": 1,
                "primary_diagnosis_icd": "I10",
                "primary_icd_3digit": "10_I10",
                "diagnosis_count": 2,
                "unique_icd_count": 2,
                "diagnosis_diversity_ratio": 1.0,
                "diagnoses_icd_list": ["I10", "E119"],
                "diagnoses_3digit_list": ["10_I10", "10_E11"],
                "icd_version_mix": "ICD10_only",
            },
            {
                "hadm_id": 102,
                "subject_id": 2,
                "primary_diagnosis_icd": "I10",
                "primary_icd_3digit": "10_I10",
                "diagnosis_count": 2,
                "unique_icd_count": 2,
                "diagnosis_diversity_ratio": 1.0,
                "diagnoses_icd_list": ["I10", "E119"],
                "diagnoses_3digit_list": ["10_I10", "10_E11"],
                "icd_version_mix": "ICD10_only",
            },
            {
                "hadm_id": 103,
                "subject_id": 3,
                "primary_diagnosis_icd": None,
                "primary_icd_3digit": None,
                "diagnosis_count": 1,
                "unique_icd_count": 1,
                "diagnosis_diversity_ratio": 1.0,
                "diagnoses_icd_list": ["J189"],
                "diagnoses_3digit_list": ["10_J18"],
                "icd_version_mix": "ICD10_only",
            },
            {
                "hadm_id": 104,
                "subject_id": 4,
                "primary_diagnosis_icd": "4280",
                "primary_icd_3digit": "9_428",
                "diagnosis_count": 0,
                "unique_icd_count": 0,
                "diagnosis_diversity_ratio": 0.0,
                "diagnoses_icd_list": [],
                "diagnoses_3digit_list": [],
                "icd_version_mix": "ICD9_only",
            },
        ]
    )
    con.register("diagnoses_df", diagnoses_df)
    con.execute("CREATE TABLE cdss_diagnoses AS SELECT * FROM diagnoses_df")
    con.close()


def test_load_cdss_diagnoses_normalizes_list_columns_and_prepares_sets(tmp_path: Path):
    db_path = tmp_path / "diagnoses.duckdb"
    build_test_db(db_path)

    df = load_cdss_diagnoses(db_path=db_path)
    summary = validate_cdss_diagnoses(df)

    assert summary.is_valid
    assert df.index.name == "hadm_id"
    assert isinstance(df.iloc[0]["diagnoses_icd_list"], list)
    assert isinstance(df.iloc[0]["diagnoses_3digit_list"], list)
    assert isinstance(df.iloc[0]["dx_set"], set)
    assert isinstance(df.iloc[0]["dx_3_set"], set)


def test_jaccard_similarity_handles_empty_lists():
    assert jaccard_similarity([], []) == 0.0
    assert jaccard_similarity(["A", "B"], ["B", "C"]) == 1 / 3


def test_baseline_patient_similarity_handles_null_primary():
    patient_a = {
        "primary_diagnosis_icd": None,
        "dx_set": {"I10", "E119"},
        "dx_3_set": {"10_I10", "10_E11"},
    }
    patient_b = {
        "primary_diagnosis_icd": "I10",
        "dx_set": {"I10", "E119"},
        "dx_3_set": {"10_I10", "10_E11"},
    }

    assert baseline_patient_similarity(patient_a, patient_b) == 0.6


def test_find_similar_patients_returns_sorted_top_k(tmp_path: Path):
    db_path = tmp_path / "diagnoses.duckdb"
    build_test_db(db_path)
    df = load_cdss_diagnoses(db_path=db_path)

    results = find_similar_patients(df=df, query_hadm_id=101, k=3, same_version_only=True)

    assert list(results["hadm_id"]) == [102, 101, 103]
    assert results.iloc[0]["similarity_score"] == 1.0
    assert results.iloc[1]["similarity_score"] == 1.0
    assert results.iloc[2]["similarity_score"] == 0.0


def test_find_similar_patients_can_exclude_query_row(tmp_path: Path):
    db_path = tmp_path / "diagnoses.duckdb"
    build_test_db(db_path)
    df = load_cdss_diagnoses(db_path=db_path)

    results = find_similar_patients(df=df, query_hadm_id=101, k=2, exclude_self=True)

    assert 101 not in set(results["hadm_id"])

