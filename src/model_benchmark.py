from __future__ import annotations

import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

try:
    from .clinical_similarity import load_clinical_similarity_dataset
except ImportError:  # pragma: no cover
    from clinical_similarity import load_clinical_similarity_dataset


DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CUSTOM_TOP_K = 15
DEFAULT_CUSTOM_CANDIDATE_LIMIT = 2000

NUMERIC_FEATURE_COLUMNS = (
    "age",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "diagnosis_count",
    "unique_icd_count",
    "diagnosis_diversity_ratio",
    "proc_count",
    "proc_group_count",
    "surgery_count",
    "rx_total_count",
    "rx_unique_drugs",
    "rx_unique_formulary_drugs",
    "rx_iv_count",
    "rx_oral_count",
    "rx_iv_ratio",
    "treatment_days",
    "avg_rx_duration_days",
    "treatment_complexity_score",
)
CATEGORICAL_FEATURE_COLUMNS = (
    "gender",
    "race",
    "admission_type",
    "admission_location",
    "discharge_location",
    "insurance",
    "language",
    "primary_icd_3digit",
    "icd_version_mix",
    "treatment_intensity_label",
)
BINARY_OUTCOME_COLUMNS = ("mortality", "readmission_30day")


@dataclass(frozen=True)
class BenchmarkRunSummary:
    target_column: str
    row_count: int
    train_size: int
    test_size: int
    train_positive_rate: float
    test_positive_rate: float
    xgboost_available: bool
    best_model_name: str


@dataclass
class BenchmarkArtifacts:
    comparison_df: pd.DataFrame
    summary: BenchmarkRunSummary
    analysis: str
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_features: pd.DataFrame
    test_features: pd.DataFrame
    fitted_models: dict[str, Any]
    predicted_labels: dict[str, np.ndarray]
    predicted_scores: dict[str, np.ndarray]


class AccuracyTunedLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        validation_fraction: float = 0.2,
        random_state: int = DEFAULT_RANDOM_STATE,
        max_iter: int = 1000,
        solver: str = "liblinear",
    ) -> None:
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.max_iter = max_iter
        self.solver = solver
        self.threshold_ = 0.5
        self.model_: LogisticRegression | None = None
        self.classes_ = np.array([0, 1], dtype=int)

    def fit(self, X: Any, y: Any) -> "AccuracyTunedLogisticRegression":
        y_array = np.asarray(y, dtype=int)
        base_model = LogisticRegression(max_iter=self.max_iter, solver=self.solver)

        enough_for_validation = len(y_array) >= 20 and len(np.unique(y_array)) > 1 and y_array.sum() >= 2
        if enough_for_validation:
            x_fit, x_val, y_fit, y_val = train_test_split(
                X,
                y_array,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=y_array,
            )
            tuning_model = LogisticRegression(max_iter=self.max_iter, solver=self.solver)
            tuning_model.fit(x_fit, y_fit)
            val_scores = tuning_model.predict_proba(x_val)[:, 1]
            candidate_thresholds = np.unique(np.concatenate([np.array([0.5]), val_scores]))

            best_threshold = 0.5
            best_accuracy = -1.0
            best_f1 = -1.0
            for threshold in candidate_thresholds:
                predictions = (val_scores >= threshold).astype(int)
                accuracy = accuracy_score(y_val, predictions)
                f1 = f1_score(y_val, predictions, zero_division=0)
                if accuracy > best_accuracy or (accuracy == best_accuracy and f1 > best_f1):
                    best_accuracy = float(accuracy)
                    best_f1 = float(f1)
                    best_threshold = float(threshold)
            self.threshold_ = best_threshold
        else:
            self.threshold_ = 0.5

        base_model.fit(X, y_array)
        self.model_ = base_model
        self.classes_ = np.asarray(base_model.classes_)
        return self

    @property
    def coef_(self) -> np.ndarray:
        if self.model_ is None:
            raise AttributeError("Model is not fitted yet.")
        return self.model_.coef_

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("Model must be fitted before calling predict_proba.")
        return self.model_.predict_proba(X)

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= self.threshold_).astype(int)


def _is_binary_series(series: pd.Series) -> bool:
    values = series.dropna().unique()
    if len(values) == 0:
        return False
    return set(pd.Series(values).astype(int).tolist()).issubset({0, 1})


def pick_binary_target_column(df: pd.DataFrame, preferred: str | None = None) -> str:
    available = [column for column in BINARY_OUTCOME_COLUMNS if column in df.columns and _is_binary_series(df[column])]
    if preferred is not None:
        if preferred not in available:
            raise ValueError(f"Binary target column '{preferred}' not available. Available binary targets: {available}")
        return preferred
    if "mortality" in available:
        return "mortality"
    if available:
        return available[0]
    raise ValueError("No supported binary target column found. Expected one of: mortality, readmission_30day.")


def load_benchmark_dataset(db_path: str | Path | None = None) -> pd.DataFrame:
    df = load_clinical_similarity_dataset(db_path=db_path).copy()
    if "dx_set" not in df.columns:
        df["dx_set"] = df["diagnoses_icd_list"].map(set)
    if "rx_set" not in df.columns:
        df["rx_set"] = df["rx_drug_list"].map(set)
    if "proc_set" not in df.columns:
        df["proc_set"] = df["proc_icd_list"].map(set)
    df["treatment_token_set"] = df.apply(
        lambda row: {f"rx:{code}" for code in row["rx_set"]} | {f"proc:{code}" for code in row["proc_set"]},
        axis=1,
    )
    return df


def _sample_rows(df: pd.DataFrame, target_column: str, max_rows: int | None, random_state: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    sample_size = int(max_rows)
    if sample_size < 2:
        raise ValueError("max_rows must be >= 2 when provided.")
    stratify = df[target_column] if df[target_column].nunique() > 1 and df[target_column].value_counts().min() >= 2 else None
    sampled_df, _ = train_test_split(
        df,
        train_size=sample_size,
        random_state=random_state,
        stratify=stratify,
    )
    return sampled_df.sort_index().set_index("hadm_id", drop=False)


def build_tabular_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [column for column in NUMERIC_FEATURE_COLUMNS if column in df.columns]
    categorical_columns = [column for column in CATEGORICAL_FEATURE_COLUMNS if column in df.columns]
    return df[numeric_columns + categorical_columns].copy()


def _build_tabular_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = [column for column in NUMERIC_FEATURE_COLUMNS if column in feature_frame.columns]
    categorical_columns = [column for column in CATEGORICAL_FEATURE_COLUMNS if column in feature_frame.columns]

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ],
    )


def split_benchmark_dataset(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    target = df[target_column].astype(int)
    stratify = target if target.nunique() > 1 and target.value_counts().min() >= 2 else None
    train_df, test_df, y_train, y_test = train_test_split(
        df,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    sorted_train_df = train_df.sort_index().set_index("hadm_id", drop=False)
    sorted_test_df = test_df.sort_index().set_index("hadm_id", drop=False)
    return (
        sorted_train_df,
        sorted_test_df,
        y_train.loc[sorted_train_df.index],
        y_test.loc[sorted_test_df.index],
    )


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _score_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_true, y_score),
    }


def _positive_probability(estimator: Any, features: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(features)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]
        return probabilities.ravel()
    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(features), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))
    return np.asarray(estimator.predict(features), dtype=float)


def _evaluate_sklearn_model(
    model_name: str,
    estimator: Any,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    started_at = perf_counter()
    estimator.fit(train_features, y_train.to_numpy())
    y_pred = np.asarray(estimator.predict(test_features), dtype=int)
    y_score = _positive_probability(estimator, test_features)
    metrics = _score_predictions(y_test.to_numpy(dtype=int), y_pred, y_score)
    metrics["fit_predict_seconds"] = perf_counter() - started_at
    metrics["model"] = model_name
    return metrics


class TopKJaccardVotingClassifier:
    def __init__(
        self,
        top_k: int = DEFAULT_CUSTOM_TOP_K,
        diagnosis_weight: float = 0.60,
        diagnosis_group_weight: float = 0.00,
        treatment_weight: float = 0.40,
        primary_weight: float = 0.00,
        context_weight: float = 0.00,
        candidate_limit: int | None = DEFAULT_CUSTOM_CANDIDATE_LIMIT,
        similarity_power: float = 2.0,
        validation_fraction: float = 0.2,
        random_state: int = DEFAULT_RANDOM_STATE,
        tune_threshold: bool = False,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.top_k = int(top_k)
        self.diagnosis_weight = float(diagnosis_weight)
        self.diagnosis_group_weight = float(diagnosis_group_weight)
        self.treatment_weight = float(treatment_weight)
        self.primary_weight = float(primary_weight)
        self.context_weight = float(context_weight)
        self.candidate_limit = candidate_limit
        self.similarity_power = float(similarity_power)
        self.validation_fraction = float(validation_fraction)
        self.random_state = int(random_state)
        self.tune_threshold = bool(tune_threshold)
        self._token_index: dict[str, list[int]] = {}
        self._train_df: pd.DataFrame | None = None
        self._y_train: np.ndarray | None = None
        self._base_rate: float = 0.0
        self.threshold_: float = 0.5
        self._train_dx_sets: np.ndarray | None = None
        self._train_dx_3_sets: np.ndarray | None = None
        self._train_treatment_sets: np.ndarray | None = None
        self._train_primary: np.ndarray | None = None
        self._train_age: np.ndarray | None = None
        self._train_diagnosis_count: np.ndarray | None = None
        self._train_treatment_complexity: np.ndarray | None = None
        self._train_gender: np.ndarray | None = None
        self._train_race: np.ndarray | None = None

    @staticmethod
    def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
        union_size = len(set_a | set_b)
        if union_size == 0:
            return 0.0
        return len(set_a & set_b) / union_size

    @staticmethod
    def _safe_string(value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        return str(value)

    @staticmethod
    def _numeric_similarity(value_a: Any, value_b: Any, scale: float) -> float:
        if value_a is None or value_b is None or pd.isna(value_a) or pd.isna(value_b):
            return 0.5
        difference = float(value_a) - float(value_b)
        return float(np.exp(-((difference / scale) ** 2)))

    def _context_similarity(self, row_a: pd.Series, row_b: pd.Series) -> float:
        return self._context_similarity_values(
            row_a.get("age"),
            row_b.get("age"),
            row_a.get("diagnosis_count"),
            row_b.get("diagnosis_count"),
            row_a.get("treatment_complexity_score"),
            row_b.get("treatment_complexity_score"),
            row_a.get("gender"),
            row_b.get("gender"),
            row_a.get("race"),
            row_b.get("race"),
        )

    def _context_similarity_values(
        self,
        age_a: Any,
        age_b: Any,
        diagnosis_count_a: Any,
        diagnosis_count_b: Any,
        treatment_complexity_a: Any,
        treatment_complexity_b: Any,
        gender_a: Any,
        gender_b: Any,
        race_a: Any,
        race_b: Any,
    ) -> float:
        age_sim = self._numeric_similarity(age_a, age_b, scale=15.0)
        diagnosis_burden_sim = self._numeric_similarity(
            diagnosis_count_a,
            diagnosis_count_b,
            scale=6.0,
        )
        treatment_complexity_sim = self._numeric_similarity(
            treatment_complexity_a,
            treatment_complexity_b,
            scale=0.25,
        )
        gender_match = float(
            self._safe_string(gender_a) is not None
            and self._safe_string(gender_a) == self._safe_string(gender_b)
        )
        race_match = float(
            self._safe_string(race_a) is not None
            and self._safe_string(race_a) == self._safe_string(race_b)
        )
        return (
            0.30 * age_sim
            + 0.20 * diagnosis_burden_sim
            + 0.20 * treatment_complexity_sim
            + 0.15 * gender_match
            + 0.15 * race_match
        )

    def _row_similarity(self, row_a: pd.Series, row_b: pd.Series) -> float:
        diagnosis_sim = self._jaccard_similarity(row_a["dx_set"], row_b["dx_set"])
        diagnosis_group_sim = self._jaccard_similarity(row_a["dx_3_set"], row_b["dx_3_set"])
        treatment_sim = self._jaccard_similarity(row_a["treatment_token_set"], row_b["treatment_token_set"])
        primary_match = float(
            self._safe_string(row_a.get("primary_diagnosis_icd")) is not None
            and self._safe_string(row_a.get("primary_diagnosis_icd")) == self._safe_string(row_b.get("primary_diagnosis_icd"))
        )
        context_sim = self._context_similarity(row_a, row_b)
        return (
            self.diagnosis_weight * diagnosis_sim
            + self.diagnosis_group_weight * diagnosis_group_sim
            + self.treatment_weight * treatment_sim
            + self.primary_weight * primary_match
            + self.context_weight * context_sim
        )

    def _tune_threshold(self, train_df: pd.DataFrame, y_train: np.ndarray) -> float:
        if not self.tune_threshold:
            return 0.5
        enough_for_validation = len(y_train) >= 20 and len(np.unique(y_train)) > 1 and y_train.sum() >= 2
        if not enough_for_validation:
            return 0.5

        fit_df, val_df, fit_y, val_y = train_test_split(
            train_df,
            y_train,
            test_size=self.validation_fraction,
            random_state=self.random_state,
            stratify=y_train,
        )
        tuning_model = TopKJaccardVotingClassifier(
            top_k=self.top_k,
            diagnosis_weight=self.diagnosis_weight,
            diagnosis_group_weight=self.diagnosis_group_weight,
            treatment_weight=self.treatment_weight,
            primary_weight=self.primary_weight,
            context_weight=self.context_weight,
            candidate_limit=self.candidate_limit,
            similarity_power=self.similarity_power,
            validation_fraction=self.validation_fraction,
            random_state=self.random_state,
            tune_threshold=False,
        )
        tuning_model.fit(fit_df, fit_y)
        validation_scores = tuning_model.predict_proba(val_df)[:, 1]
        candidate_thresholds = np.unique(np.concatenate([np.array([0.5]), validation_scores]))

        best_threshold = 0.5
        best_accuracy = -1.0
        best_f1 = -1.0
        for threshold in candidate_thresholds:
            predictions = (validation_scores >= threshold).astype(int)
            accuracy = accuracy_score(val_y, predictions)
            f1 = f1_score(val_y, predictions, zero_division=0)
            if accuracy > best_accuracy or (accuracy == best_accuracy and f1 > best_f1):
                best_accuracy = float(accuracy)
                best_f1 = float(f1)
                best_threshold = float(threshold)
        return best_threshold

    def fit(self, train_df: pd.DataFrame, y_train: Iterable[int]) -> "TopKJaccardVotingClassifier":
        self._train_df = train_df.copy()
        self._y_train = np.asarray(list(y_train), dtype=int)
        self._base_rate = float(self._y_train.mean()) if len(self._y_train) else 0.0
        self._train_dx_sets = self._train_df["dx_set"].to_numpy(dtype=object)
        self._train_dx_3_sets = self._train_df["dx_3_set"].to_numpy(dtype=object)
        self._train_treatment_sets = self._train_df["treatment_token_set"].to_numpy(dtype=object)
        self._train_primary = self._train_df["primary_diagnosis_icd"].to_numpy(dtype=object)
        self._train_age = self._train_df["age"].to_numpy(dtype=object)
        self._train_diagnosis_count = self._train_df["diagnosis_count"].to_numpy(dtype=object)
        self._train_treatment_complexity = self._train_df["treatment_complexity_score"].to_numpy(dtype=object)
        self._train_gender = self._train_df["gender"].to_numpy(dtype=object)
        self._train_race = self._train_df["race"].to_numpy(dtype=object)

        token_index: dict[str, list[int]] = defaultdict(list)
        for row_idx, (_, row) in enumerate(self._train_df.iterrows()):
            for token in row["dx_set"]:
                token_index[f"dx:{token}"].append(row_idx)
            for token in row["dx_3_set"]:
                token_index[f"dx3:{token}"].append(row_idx)
            for token in row["treatment_token_set"]:
                token_index[token].append(row_idx)
            primary_diagnosis = self._safe_string(row.get("primary_diagnosis_icd"))
            if primary_diagnosis is not None:
                token_index[f"primary:{primary_diagnosis}"].append(row_idx)
        self._token_index = dict(token_index)
        self.threshold_ = self._tune_threshold(self._train_df, self._y_train)
        return self

    def _select_candidate_positions(self, row: pd.Series) -> np.ndarray:
        assert self._train_df is not None
        token_counts: Counter[int] = Counter()
        for token in row["dx_set"]:
            token_counts.update(self._token_index.get(f"dx:{token}", []))
        for token in row["dx_3_set"]:
            token_counts.update(self._token_index.get(f"dx3:{token}", []))
        for token in row["treatment_token_set"]:
            token_counts.update(self._token_index.get(token, []))
        primary_diagnosis = self._safe_string(row.get("primary_diagnosis_icd"))
        if primary_diagnosis is not None:
            token_counts.update(self._token_index.get(f"primary:{primary_diagnosis}", []))

        if not token_counts:
            return np.arange(len(self._train_df), dtype=int)

        candidate_positions = np.fromiter(token_counts.keys(), dtype=int)
        if self.candidate_limit is None or len(candidate_positions) <= self.candidate_limit:
            return candidate_positions

        counts = np.fromiter((token_counts[position] for position in candidate_positions), dtype=int)
        top_n = min(len(candidate_positions), int(self.candidate_limit))
        top_positions = np.argpartition(counts, -top_n)[-top_n:]
        return candidate_positions[top_positions]

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        if self._train_df is None or self._y_train is None:
            raise ValueError("Model must be fitted before calling predict_proba.")
        if (
            self._train_dx_sets is None
            or self._train_dx_3_sets is None
            or self._train_treatment_sets is None
            or self._train_primary is None
            or self._train_age is None
            or self._train_diagnosis_count is None
            or self._train_treatment_complexity is None
            or self._train_gender is None
            or self._train_race is None
        ):
            raise ValueError("Cached training arrays are missing. Fit the model again.")

        scores = np.empty((len(test_df), 2), dtype=float)
        train_dx_sets = self._train_dx_sets
        train_dx_3_sets = self._train_dx_3_sets
        train_treatment_sets = self._train_treatment_sets
        train_primary = self._train_primary
        train_age = self._train_age
        train_diagnosis_count = self._train_diagnosis_count
        train_treatment_complexity = self._train_treatment_complexity
        train_gender = self._train_gender
        train_race = self._train_race

        for idx, (_, row) in enumerate(test_df.iterrows()):
            candidate_positions = self._select_candidate_positions(row)
            if len(candidate_positions) == 0:
                positive_probability = self._base_rate
            else:
                query_primary = self._safe_string(row.get("primary_diagnosis_icd"))
                query_age = row.get("age")
                query_diagnosis_count = row.get("diagnosis_count")
                query_treatment_complexity = row.get("treatment_complexity_score")
                query_gender = row.get("gender")
                query_race = row.get("race")
                similarities = np.fromiter(
                    (
                        self.diagnosis_weight * self._jaccard_similarity(row["dx_set"], train_dx_sets[position])
                        + self.diagnosis_group_weight * self._jaccard_similarity(row["dx_3_set"], train_dx_3_sets[position])
                        + self.treatment_weight
                        * self._jaccard_similarity(row["treatment_token_set"], train_treatment_sets[position])
                        + self.primary_weight
                        * float(
                            query_primary is not None and query_primary == self._safe_string(train_primary[position])
                        )
                        + self.context_weight
                        * self._context_similarity_values(
                            query_age,
                            train_age[position],
                            query_diagnosis_count,
                            train_diagnosis_count[position],
                            query_treatment_complexity,
                            train_treatment_complexity[position],
                            query_gender,
                            train_gender[position],
                            query_race,
                            train_race[position],
                        )
                        for position in candidate_positions
                    ),
                    dtype=float,
                    count=len(candidate_positions),
                )
                top_k = min(self.top_k, len(candidate_positions))
                if top_k == len(candidate_positions):
                    top_positions = np.argsort(similarities)[::-1]
                else:
                    top_positions = np.argpartition(similarities, -top_k)[-top_k:]
                    top_positions = top_positions[np.argsort(similarities[top_positions])[::-1]]

                neighbor_labels = self._y_train[candidate_positions[top_positions]]
                neighbor_similarities = similarities[top_positions]
                if len(neighbor_labels) and float(np.sum(neighbor_similarities)) > 0.0:
                    weights = np.power(np.clip(neighbor_similarities, a_min=0.0, a_max=None), self.similarity_power)
                    weight_sum = float(np.sum(weights))
                    positive_probability = (
                        float(np.dot(weights, neighbor_labels) / weight_sum)
                        if weight_sum > 0.0
                        else float(neighbor_labels.mean())
                    )
                else:
                    positive_probability = float(neighbor_labels.mean()) if len(neighbor_labels) else self._base_rate

            scores[idx, 1] = positive_probability
            scores[idx, 0] = 1.0 - positive_probability

        return scores

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(test_df)[:, 1]
        return (probabilities >= self.threshold_).astype(int)


def _build_model_registry(
    feature_frame: pd.DataFrame,
    random_state: int,
    knn_neighbors: int,
    include_xgboost: bool,
) -> dict[str, Any]:
    def make_pipeline(classifier: Any) -> Pipeline:
        return Pipeline(
            [
                ("preprocessor", _build_tabular_preprocessor(feature_frame)),
                ("classifier", classifier),
            ]
        )

    model_registry: dict[str, Any] = {
        "Logistic Regression": make_pipeline(
            LogisticRegression(max_iter=1000, solver="liblinear")
        ),
        "Accuracy-Tuned Logistic Regression": make_pipeline(
            AccuracyTunedLogisticRegression(
                random_state=random_state,
                max_iter=1000,
                solver="liblinear",
            )
        ),
        "Random Forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1,
                min_samples_leaf=2,
            )
        ),
        "kNN (cosine)": make_pipeline(
            KNeighborsClassifier(
                n_neighbors=knn_neighbors,
                metric="cosine",
                algorithm="brute",
                weights="distance",
                n_jobs=-1,
            )
        ),
    }
    if include_xgboost and XGBClassifier is not None:
        model_registry["XGBoost"] = make_pipeline(
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=random_state,
                n_jobs=1,
                eval_metric="logloss",
            )
        )
    return model_registry


def generate_benchmark_analysis(comparison_df: pd.DataFrame, target_column: str) -> str:
    available = comparison_df.dropna(subset=["roc_auc"]).sort_values(["roc_auc", "f1_score", "accuracy"], ascending=False)
    if available.empty:
        return f"No benchmark analysis available for {target_column}: none of the models produced a valid ROC-AUC score."

    best_row = available.iloc[0]
    model_name = best_row["model"]
    if model_name == "Accuracy-Tuned Logistic Regression":
        reason = (
            "it kept the strong linear signal from the tabular features and tuned the decision threshold "
            "specifically for holdout-set accuracy, which helped on this imbalanced outcome."
        )
    elif model_name == "Logistic Regression":
        reason = "it captured the linear risk pattern cleanly and benefited from calibrated probabilities on the engineered tabular features."
    elif model_name == "Random Forest":
        reason = "it handled nonlinear interactions across diagnosis burden, vitals, and treatment complexity better than the linear and neighbor baselines."
    elif model_name == "XGBoost":
        reason = "gradient-boosted trees modeled nonlinear feature interactions and sparse category splits more effectively than the other baselines."
    elif model_name == "kNN (cosine)":
        reason = "the cosine neighborhood over the encoded feature space preserved local patient similarity well enough to outperform the parametric baselines."
    else:
        reason = "explicit diagnosis+treatment overlap was the strongest signal, so retrieving similar cases beat the generic tabular classifiers."

    return (
        f"Best model on {target_column}: {model_name} "
        f"(ROC-AUC={best_row['roc_auc']:.4f}, F1={best_row['f1_score']:.4f}). "
        f"This model performed best because {reason}"
    )


def _fit_benchmark_models(
    db_path: str | Path | None = None,
    target_column: str | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    knn_neighbors: int = 25,
    custom_top_k: int = DEFAULT_CUSTOM_TOP_K,
    custom_candidate_limit: int | None = DEFAULT_CUSTOM_CANDIDATE_LIMIT,
    max_rows: int | None = None,
    include_xgboost: bool = True,
) -> BenchmarkArtifacts:
    df = load_benchmark_dataset(db_path=db_path)
    resolved_target = pick_binary_target_column(df, preferred=target_column)
    df = _sample_rows(df, target_column=resolved_target, max_rows=max_rows, random_state=random_state)

    train_df, test_df, y_train, y_test = split_benchmark_dataset(
        df=df,
        target_column=resolved_target,
        test_size=test_size,
        random_state=random_state,
    )
    train_features = build_tabular_feature_frame(train_df)
    test_features = build_tabular_feature_frame(test_df)

    comparison_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
    predicted_labels: dict[str, np.ndarray] = {}
    predicted_scores: dict[str, np.ndarray] = {}
    model_registry = _build_model_registry(
        feature_frame=train_features,
        random_state=random_state,
        knn_neighbors=knn_neighbors,
        include_xgboost=include_xgboost,
    )

    for model_name, estimator in model_registry.items():
        if model_name == "XGBoost" and XGBClassifier is None:
            continue
        started_at = perf_counter()
        estimator.fit(train_features, y_train.to_numpy())
        y_pred = np.asarray(estimator.predict(test_features), dtype=int)
        y_score = _positive_probability(estimator, test_features)
        metrics = _score_predictions(y_test.to_numpy(dtype=int), y_pred, y_score)
        metrics["fit_predict_seconds"] = perf_counter() - started_at
        metrics["model"] = model_name
        comparison_rows.append(metrics)
        fitted_models[model_name] = estimator
        predicted_labels[model_name] = y_pred
        predicted_scores[model_name] = y_score

    custom_model = TopKJaccardVotingClassifier(
        top_k=custom_top_k,
        candidate_limit=custom_candidate_limit,
    )
    started_at = perf_counter()
    custom_model.fit(train_df, y_train.to_numpy())
    custom_probabilities = custom_model.predict_proba(test_df)[:, 1]
    custom_predictions = (custom_probabilities >= 0.5).astype(int)
    custom_metrics = _score_predictions(y_test.to_numpy(dtype=int), custom_predictions, custom_probabilities)
    custom_metrics["fit_predict_seconds"] = perf_counter() - started_at
    custom_metrics["model"] = "Custom Jaccard KNN"
    comparison_rows.append(custom_metrics)
    fitted_models["Custom Jaccard KNN"] = custom_model
    predicted_labels["Custom Jaccard KNN"] = custom_predictions
    predicted_scores["Custom Jaccard KNN"] = custom_probabilities

    comparison_df = (
        pd.DataFrame(comparison_rows)
        .sort_values(["roc_auc", "f1_score", "accuracy"], ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    analysis = generate_benchmark_analysis(comparison_df, target_column=resolved_target)
    best_model_name = str(comparison_df.iloc[0]["model"])

    summary = BenchmarkRunSummary(
        target_column=resolved_target,
        row_count=int(len(df)),
        train_size=int(len(train_df)),
        test_size=int(len(test_df)),
        train_positive_rate=float(y_train.mean()),
        test_positive_rate=float(y_test.mean()),
        xgboost_available=bool(XGBClassifier is not None),
        best_model_name=best_model_name,
    )
    return BenchmarkArtifacts(
        comparison_df=comparison_df,
        summary=summary,
        analysis=analysis,
        train_df=train_df,
        test_df=test_df,
        y_train=y_train,
        y_test=y_test,
        train_features=train_features,
        test_features=test_features,
        fitted_models=fitted_models,
        predicted_labels=predicted_labels,
        predicted_scores=predicted_scores,
    )


def run_model_benchmark(
    db_path: str | Path | None = None,
    target_column: str | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    knn_neighbors: int = 25,
    custom_top_k: int = DEFAULT_CUSTOM_TOP_K,
    custom_candidate_limit: int | None = DEFAULT_CUSTOM_CANDIDATE_LIMIT,
    max_rows: int | None = None,
    include_xgboost: bool = True,
) -> tuple[pd.DataFrame, BenchmarkRunSummary, str]:
    artifacts = _fit_benchmark_models(
        db_path=db_path,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
        knn_neighbors=knn_neighbors,
        custom_top_k=custom_top_k,
        custom_candidate_limit=custom_candidate_limit,
        max_rows=max_rows,
        include_xgboost=include_xgboost,
    )
    return artifacts.comparison_df, artifacts.summary, artifacts.analysis


def build_confusion_matrix_table(artifacts: BenchmarkArtifacts) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    y_true = artifacts.y_test.to_numpy(dtype=int)
    for model_name, y_pred in artifacts.predicted_labels.items():
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "model": model_name,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def _clean_feature_name(name: str) -> str:
    for prefix in ("numeric__", "categorical__"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def build_feature_importance_tables(
    artifacts: BenchmarkArtifacts,
    top_n: int = 10,
) -> dict[str, pd.DataFrame]:
    feature_tables: dict[str, pd.DataFrame] = {}

    for model_name, estimator in artifacts.fitted_models.items():
        if not isinstance(estimator, Pipeline):
            continue

        classifier = estimator.named_steps["classifier"]
        preprocessor = estimator.named_steps["preprocessor"]
        if not hasattr(preprocessor, "get_feature_names_out"):
            continue

        feature_names = [_clean_feature_name(name) for name in preprocessor.get_feature_names_out()]

        if hasattr(classifier, "coef_"):
            raw_values = np.asarray(classifier.coef_, dtype=float)
            if raw_values.ndim == 2:
                raw_values = raw_values[0]
            table = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": np.abs(raw_values),
                    "raw_value": raw_values,
                }
            )
        elif hasattr(classifier, "feature_importances_"):
            table = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": np.asarray(classifier.feature_importances_, dtype=float),
                }
            )
        else:
            continue

        feature_tables[model_name] = table.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

    return feature_tables


def run_model_diagnostics(
    db_path: str | Path | None = None,
    target_column: str | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    knn_neighbors: int = 25,
    custom_top_k: int = DEFAULT_CUSTOM_TOP_K,
    custom_candidate_limit: int | None = DEFAULT_CUSTOM_CANDIDATE_LIMIT,
    max_rows: int | None = None,
    include_xgboost: bool = True,
    top_n_features: int = 10,
) -> tuple[pd.DataFrame, BenchmarkRunSummary, str, pd.DataFrame, dict[str, pd.DataFrame]]:
    artifacts = _fit_benchmark_models(
        db_path=db_path,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
        knn_neighbors=knn_neighbors,
        custom_top_k=custom_top_k,
        custom_candidate_limit=custom_candidate_limit,
        max_rows=max_rows,
        include_xgboost=include_xgboost,
    )
    confusion_df = build_confusion_matrix_table(artifacts)
    feature_tables = build_feature_importance_tables(artifacts, top_n=top_n_features)
    return artifacts.comparison_df, artifacts.summary, artifacts.analysis, confusion_df, feature_tables


def save_residual_plot(
    artifacts: BenchmarkArtifacts,
    output_path: str | Path,
    model_name: str | None = None,
) -> Path:
    selected_model = model_name or artifacts.summary.best_model_name
    if selected_model not in artifacts.predicted_scores:
        raise ValueError(f"Model '{selected_model}' not found. Available: {sorted(artifacts.predicted_scores)}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mpl_config_dir = output_path.parent / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(output_path.parent))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = artifacts.y_test.to_numpy(dtype=float)
    y_score = np.asarray(artifacts.predicted_scores[selected_model], dtype=float)
    residuals = y_true - y_score

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_score, residuals, alpha=0.35, s=18, edgecolors="none")
    axes[0].axhline(0.0, color="crimson", linestyle="--", linewidth=1.0)
    axes[0].set_title(f"{selected_model} Residuals")
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_ylabel("Residual (y - p)")
    axes[0].set_xlim(-0.02, 1.02)

    axes[1].hist(residuals, bins=30, color="#4C78A8", alpha=0.85)
    axes[1].axvline(0.0, color="crimson", linestyle="--", linewidth=1.0)
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual (y - p)")
    axes[1].set_ylabel("Count")

    positive_rate = float(y_true.mean()) if len(y_true) else float("nan")
    figure.suptitle(
        f"{selected_model} residual diagnostic | target={artifacts.summary.target_column} | test positive rate={positive_rate:.4f}",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path
