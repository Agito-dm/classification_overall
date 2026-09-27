import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from finnews_sentiment.models.train_model import (
    build_baseline_pipeline,
    evaluate_model,
    LABEL_IDS,
    LABEL_NAMES,
    load_config,
    load_processed_dataset,
    split_dataset,
)
from finnews_sentiment.paths import find_project_root


def build_experiment_pipelines(
    max_iter: int,
    random_state: int,
) -> dict[str, Pipeline]:
    """Создаёт baseline и варианты улучшения модели."""
    return {
        "baseline": build_baseline_pipeline(
            max_iter=max_iter,
            random_state=random_state,
        ),
        "bigrams_lr": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=max_iter,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "balanced_lr": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 1),
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=max_iter,
                        random_state=random_state,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "linear_svc": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 1),
                    ),
                ),
                (
                    "classifier",
                    LinearSVC(
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def run_experiments(
    pipelines: dict[str, Pipeline],
    X_train: pd.Series,
    X_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[
    pd.DataFrame,
    dict[str, object],
]:
    """Обучает все модели и собирает метрики и predictions."""
    rows = []
    predictions = {}

    for name, pipeline in pipelines.items():
        pipeline.fit(
            X_train,
            y_train,
        )

        y_pred = pipeline.predict(
            X_test
        )

        predictions[name] = y_pred

        metrics = evaluate_model(
            y_test,
            y_pred,
        )

        rows.append(
            {
                "model": name,
                **metrics,
            }
        )

    return (
        pd.DataFrame(rows),
        predictions,
    )


def add_improvement_columns(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Считает улучшение Macro F1 относительно baseline."""
    result = results.copy()

    baseline_macro_f1 = result.loc[
        result["model"] == "baseline",
        "macro_f1",
    ].iloc[0]

    result["macro_f1_absolute_gain"] = (
        result["macro_f1"]
        - baseline_macro_f1
    )

    result["macro_f1_relative_gain_pct"] = (
        result["macro_f1_absolute_gain"]
        / baseline_macro_f1
        * 100
    )

    return result


def save_comparison_plot(
    results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Сохраняет сравнение Macro F1 моделей."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_data = results.sort_values(
        "macro_f1",
        ascending=True,
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.barh(
        plot_data["model"],
        plot_data["macro_f1"],
    )

    plt.xlabel("Macro F1")
    plt.ylabel("Model")
    plt.title(
        "Day 4 model comparison"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def main(config_path: Path) -> None:
    project_root = find_project_root()
    config = load_config(config_path)

    input_path = (
        project_root
        / config["input_path"]
    )

    df = load_processed_dataset(
        input_path,
        text_col=config["text_col"],
        target_col=config["target_col"],
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_dataset(
        df,
        text_col=config["text_col"],
        target_col=config["target_col"],
        test_size=config["test_size"],
        random_state=config["random_state"],
    )

    pipelines = build_experiment_pipelines(
        max_iter=config["max_iter"],
        random_state=config["random_state"],
    )

    (
        results,
        predictions,
    ) = run_experiments(
        pipelines,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    results = add_improvement_columns(
        results
    )

    results = results.sort_values(
        "macro_f1",
        ascending=False,
    ).reset_index(drop=True)

    best_model_name = (
        results.iloc[0]["model"]
    )

    best_predictions = predictions[
        best_model_name
    ]

    best_report = classification_report(
        y_test,
        best_predictions,
        labels=LABEL_IDS,
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )

    reports_path = (
        project_root
        / "reports"
    )

    figures_path = (
        reports_path
        / "figures"
    )

    reports_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_csv_path = (
        reports_path
        / "day04_model_comparison.csv"
    )

    comparison_json_path = (
        reports_path
        / "day04_model_comparison.json"
    )

    best_report_path = (
        reports_path
        / "day04_best_classification_report.txt"
    )

    plot_path = (
        figures_path
        / "day04_macro_f1_comparison.png"
    )

    results.to_csv(
        comparison_csv_path,
        index=False,
    )

    comparison_json_path.write_text(
        json.dumps(
            results.to_dict(
                orient="records"
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    best_report_path.write_text(
        (
            f"Best model: "
            f"{best_model_name}\n\n"
            f"{best_report}"
        ),
        encoding="utf-8",
    )

    save_comparison_plot(
        results,
        plot_path,
    )

    print("=== DAY 4 MODEL COMPARISON ===")
    print(
        results.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print(
        "\nBest model:",
        best_model_name,
    )

    print(
        "\n=== BEST MODEL "
        "CLASSIFICATION REPORT ==="
    )

    print(best_report)

    print("\n=== OUTPUT FILES ===")
    print(comparison_csv_path.resolve())
    print(comparison_json_path.resolve())
    print(best_report_path.resolve())
    print(plot_path.resolve())


if __name__ == "__main__":
    default_config = (
        find_project_root()
        / "configs"
        / "train_baseline.json"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline model "
            "with Day 4 experiments."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Path to training config.",
    )

    args = parser.parse_args()

    main(args.config)
