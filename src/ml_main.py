from __future__ import annotations

import argparse

try:
    from .model_benchmark import run_model_benchmark, run_model_diagnostics, _fit_benchmark_models, save_residual_plot
    from .ml_dataset import load_joined_cdss_dataset, pick_outcome_column
    from .ml_workflow import evaluate_similarity_outcome_alignment, run_phase1_validation
except ImportError:  # pragma: no cover
    from model_benchmark import run_model_benchmark, run_model_diagnostics, _fit_benchmark_models, save_residual_plot
    from ml_dataset import load_joined_cdss_dataset, pick_outcome_column
    from ml_workflow import evaluate_similarity_outcome_alignment, run_phase1_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ML-team workflow utilities for CDSS data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    phase1 = subparsers.add_parser("phase1", help="Run joined dataset validation and similarity smoke test")
    phase1.add_argument("--db-path", default=None, help="Path to mimic.db")
    phase1.add_argument("--join-how", default="inner", choices=["inner", "left"], help="Join type for base/diagnosis")
    phase1.add_argument("--same-version-only", action="store_true", help="Restrict smoke similarity to same ICD version")

    phase2 = subparsers.add_parser("phase2", help="Evaluate diagnosis similarity vs outcome alignment")
    phase2.add_argument("--db-path", default=None, help="Path to mimic.db")
    phase2.add_argument("--join-how", default="inner", choices=["inner", "left"], help="Join type for base/diagnosis")
    phase2.add_argument("--sample-size", type=int, default=1000, help="Sample size for pairwise evaluation")
    phase2.add_argument("--outcome-column", default=None, help="Outcome column to evaluate")
    phase2.add_argument("--same-version-only", action="store_true", help="Restrict pair comparisons to same ICD version")
    phase2.add_argument("--bins", type=int, default=5, help="Number of similarity bands")

    benchmark = subparsers.add_parser("benchmark", help="Train and compare baseline classifiers plus custom Jaccard retrieval")
    benchmark.add_argument("--db-path", default=None, help="Path to mimic.db")
    benchmark.add_argument("--target-column", default=None, help="Binary target column, e.g. mortality")
    benchmark.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    benchmark.add_argument("--random-state", type=int, default=42, help="Random seed")
    benchmark.add_argument("--knn-neighbors", type=int, default=25, help="Neighbors for cosine kNN")
    benchmark.add_argument("--custom-top-k", type=int, default=15, help="Top-K neighbors for custom Jaccard voting")
    benchmark.add_argument(
        "--custom-candidate-limit",
        type=int,
        default=2000,
        help="Max candidate neighbors scored per query in the custom model",
    )
    benchmark.add_argument("--max-rows", type=int, default=None, help="Optional stratified sample size for faster runs")
    benchmark.add_argument("--skip-xgboost", action="store_true", help="Skip XGBoost even if the package is installed")

    diagnostics = subparsers.add_parser("diagnostics", help="Show confusion matrices and feature importance for the benchmark models")
    diagnostics.add_argument("--db-path", default=None, help="Path to mimic.db")
    diagnostics.add_argument("--target-column", default=None, help="Binary target column, e.g. mortality")
    diagnostics.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    diagnostics.add_argument("--random-state", type=int, default=42, help="Random seed")
    diagnostics.add_argument("--knn-neighbors", type=int, default=25, help="Neighbors for cosine kNN")
    diagnostics.add_argument("--custom-top-k", type=int, default=15, help="Top-K neighbors for custom Jaccard voting")
    diagnostics.add_argument(
        "--custom-candidate-limit",
        type=int,
        default=2000,
        help="Max candidate neighbors scored per query in the custom model",
    )
    diagnostics.add_argument("--max-rows", type=int, default=None, help="Optional stratified sample size for faster runs")
    diagnostics.add_argument("--skip-xgboost", action="store_true", help="Skip XGBoost even if the package is installed")
    diagnostics.add_argument("--top-n-features", type=int, default=10, help="Top feature-importance rows per supported model")

    residual_plot = subparsers.add_parser("residual-plot", help="Generate a residual plot for a benchmark model")
    residual_plot.add_argument("--db-path", default=None, help="Path to mimic.db")
    residual_plot.add_argument("--target-column", default=None, help="Binary target column, e.g. mortality")
    residual_plot.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    residual_plot.add_argument("--random-state", type=int, default=42, help="Random seed")
    residual_plot.add_argument("--knn-neighbors", type=int, default=25, help="Neighbors for cosine kNN")
    residual_plot.add_argument("--custom-top-k", type=int, default=15, help="Top-K neighbors for custom Jaccard voting")
    residual_plot.add_argument("--custom-candidate-limit", type=int, default=2000, help="Max candidate neighbors for the custom model")
    residual_plot.add_argument("--max-rows", type=int, default=None, help="Optional stratified sample size for faster runs")
    residual_plot.add_argument("--skip-xgboost", action="store_true", help="Skip XGBoost even if the package is installed")
    residual_plot.add_argument("--model-name", default=None, help="Specific model to plot; defaults to the best model")
    residual_plot.add_argument(
        "--output-path",
        default="artifacts/residual_plot.png",
        help="Path to save the residual plot PNG",
    )

    return parser


def run_phase1(args: argparse.Namespace) -> None:
    _, summary = run_phase1_validation(
        db_path=args.db_path,
        join_how=args.join_how,
        include_treatment_features=True,
        same_version_only=args.same_version_only,
    )
    print(f"rows={summary.row_count}")
    print(f"available_outcomes={list(summary.available_outcomes)}")
    print(f"mortality_rate={summary.mortality_rate}")
    print(f"readmission_rate={summary.readmission_rate}")
    print(f"average_diagnosis_count={summary.average_diagnosis_count:.4f}")
    print(f"average_treatment_event_count={summary.average_treatment_event_count}")
    print(f"list_columns_are_python_lists={summary.list_columns_are_python_lists}")
    print(f"quick_similarity_min={summary.quick_similarity_min:.6f}")
    print(f"quick_similarity_max={summary.quick_similarity_max:.6f}")
    print(f"quick_similarity_mean={summary.quick_similarity_mean:.6f}")


def run_phase2(args: argparse.Namespace) -> None:
    df = load_joined_cdss_dataset(
        db_path=args.db_path,
        join_how=args.join_how,
        include_treatment_features=True,
        prepare_sets=True,
    )
    outcome_column = pick_outcome_column(df, preferred=args.outcome_column) if args.outcome_column else None
    summary, band_summary = evaluate_similarity_outcome_alignment(
        df=df,
        outcome_column=outcome_column,
        sample_size=args.sample_size,
        same_version_only=args.same_version_only,
        bins=args.bins,
    )
    print(f"outcome_column={summary.outcome_column}")
    print(f"sampled_patients={summary.sampled_patients}")
    print(f"pair_count={summary.pair_count}")
    print(f"same_version_only={summary.same_version_only}")
    print(f"correlation={summary.correlation:.6f}")
    print(f"mean_similarity={summary.mean_similarity:.6f}")
    print(f"mean_outcome_match_rate={summary.mean_outcome_match_rate:.6f}")
    print(band_summary.to_string(index=False))


def run_benchmark(args: argparse.Namespace) -> None:
    comparison_df, summary, analysis = run_model_benchmark(
        db_path=args.db_path,
        target_column=args.target_column,
        test_size=args.test_size,
        random_state=args.random_state,
        knn_neighbors=args.knn_neighbors,
        custom_top_k=args.custom_top_k,
        custom_candidate_limit=args.custom_candidate_limit,
        max_rows=args.max_rows,
        include_xgboost=not args.skip_xgboost,
    )
    print(f"target_column={summary.target_column}")
    print(f"rows={summary.row_count}")
    print(f"train_size={summary.train_size}")
    print(f"test_size={summary.test_size}")
    print(f"train_positive_rate={summary.train_positive_rate:.6f}")
    print(f"test_positive_rate={summary.test_positive_rate:.6f}")
    print(f"xgboost_available={summary.xgboost_available}")
    print(f"best_model={summary.best_model_name}")
    print(comparison_df.to_string(index=False))
    print(f"analysis={analysis}")


def run_diagnostics(args: argparse.Namespace) -> None:
    comparison_df, summary, analysis, confusion_df, feature_tables = run_model_diagnostics(
        db_path=args.db_path,
        target_column=args.target_column,
        test_size=args.test_size,
        random_state=args.random_state,
        knn_neighbors=args.knn_neighbors,
        custom_top_k=args.custom_top_k,
        custom_candidate_limit=args.custom_candidate_limit,
        max_rows=args.max_rows,
        include_xgboost=not args.skip_xgboost,
        top_n_features=args.top_n_features,
    )
    print(f"target_column={summary.target_column}")
    print(f"rows={summary.row_count}")
    print(f"train_size={summary.train_size}")
    print(f"test_size={summary.test_size}")
    print(f"train_positive_rate={summary.train_positive_rate:.6f}")
    print(f"test_positive_rate={summary.test_positive_rate:.6f}")
    print(f"xgboost_available={summary.xgboost_available}")
    print(f"best_model={summary.best_model_name}")
    print(comparison_df.to_string(index=False))
    print("confusion_matrices")
    print(confusion_df.to_string(index=False))
    for model_name, feature_df in feature_tables.items():
        print(f"feature_importance::{model_name}")
        print(feature_df.to_string(index=False))
    unsupported = [model for model in comparison_df["model"] if model not in feature_tables]
    if unsupported:
        print(f"feature_importance_unavailable={unsupported}")
    print(f"analysis={analysis}")


def run_residual_plot(args: argparse.Namespace) -> None:
    artifacts = _fit_benchmark_models(
        db_path=args.db_path,
        target_column=args.target_column,
        test_size=args.test_size,
        random_state=args.random_state,
        knn_neighbors=args.knn_neighbors,
        custom_top_k=args.custom_top_k,
        custom_candidate_limit=args.custom_candidate_limit,
        max_rows=args.max_rows,
        include_xgboost=not args.skip_xgboost,
    )
    output_path = save_residual_plot(
        artifacts=artifacts,
        output_path=args.output_path,
        model_name=args.model_name,
    )
    selected_model = args.model_name or artifacts.summary.best_model_name
    print(f"target_column={artifacts.summary.target_column}")
    print(f"model={selected_model}")
    print(f"output_path={output_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "phase1":
        run_phase1(args)
        return

    if args.command == "phase2":
        run_phase2(args)
        return

    if args.command == "benchmark":
        run_benchmark(args)
        return

    if args.command == "diagnostics":
        run_diagnostics(args)
        return

    if args.command == "residual-plot":
        run_residual_plot(args)
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
