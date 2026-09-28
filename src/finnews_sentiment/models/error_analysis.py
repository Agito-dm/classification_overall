import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
)

from finnews_sentiment.models.compare_models import (
    build_experiment_pipelines,
)
from finnews_sentiment.models.train_model import (
    LABEL_IDS,
    LABEL_NAMES,
    load_config,
    load_processed_dataset,
    split_dataset,
)
from finnews_sentiment.paths import find_project_root


BEST_MODEL_NAME = "linear_svc"


def build_error_dataframe(
    df: pd.DataFrame,
    X_test: pd.Series,
    y_test: pd.Series,
    y_pred,
) -> pd.DataFrame:
    """Формирует таблицу ошибочных предсказаний."""
    id_to_label = dict(
        zip(
            LABEL_IDS,
            LABEL_NAMES,
        )
    )

    result = df.loc[
        X_test.index,
        [
            "text",
            "text_clean",
        ],
    ].copy()

    result["true_label"] = y_test
    result["predicted_label"] = y_pred

    result["true_sentiment"] = (
        result["true_label"]
        .map(id_to_label)
    )

    result["predicted_sentiment"] = (
        result["predicted_label"]
        .map(id_to_label)
    )

    result = result[
        result["true_label"]
        != result["predicted_label"]
    ].copy()

    return result.reset_index(
        drop=True
    )


def build_error_summary(
    errors: pd.DataFrame,
) -> pd.DataFrame:
    """Считает основные направления ошибок модели."""
    summary = (
        errors.groupby(
            [
                "true_sentiment",
                "predicted_sentiment",
            ]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    summary["share_of_errors_pct"] = (
        summary["count"]
        / len(errors)
        * 100
    )

    return summary


def build_confusion_matrix(
    y_test: pd.Series,
    y_pred,
) -> pd.DataFrame:
    """Строит confusion matrix как DataFrame."""
    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=LABEL_IDS,
    )

    return pd.DataFrame(
        matrix,
        index=LABEL_NAMES,
        columns=LABEL_NAMES,
    )


def save_confusion_matrix_plot(
    matrix: pd.DataFrame,
    output_path: Path,
) -> None:
    """Сохраняет confusion matrix как изображение."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix.to_numpy(),
        display_labels=LABEL_NAMES,
    )

    display.plot(
        ax=axis,
    )

    axis.set_title(
        "Day 5 confusion matrix"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=150,
    )

    plt.close(
        figure
    )


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

    model = pipelines[
        BEST_MODEL_NAME
    ]

    model.fit(
        X_train,
        y_train,
    )

    y_pred = model.predict(
        X_test
    )

    matrix = build_confusion_matrix(
        y_test,
        y_pred,
    )

    errors = build_error_dataframe(
        df,
        X_test,
        y_test,
        y_pred,
    )

    error_summary = build_error_summary(
        errors
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

    confusion_csv_path = (
        reports_path
        / "day05_confusion_matrix.csv"
    )

    errors_path = (
        reports_path
        / "day05_errors.csv"
    )

    error_summary_path = (
        reports_path
        / "day05_error_directions.csv"
    )

    confusion_plot_path = (
        figures_path
        / "day05_confusion_matrix.png"
    )

    matrix.to_csv(
        confusion_csv_path,
    )

    errors.to_csv(
        errors_path,
        index=False,
    )

    error_summary.to_csv(
        error_summary_path,
        index=False,
    )

    save_confusion_matrix_plot(
        matrix,
        confusion_plot_path,
    )

    print("=== DAY 5 ERROR ANALYSIS ===")
    print(
        f"Model: {BEST_MODEL_NAME}"
    )
    print(
        f"Test samples: {len(y_test)}"
    )
    print(
        f"Errors: {len(errors)}"
    )

    print(
        "\n=== ERROR DIRECTIONS ==="
    )

    print(
        error_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.2f}"
            ),
        )
    )

    print(
        f"Correct: {len(y_test) - len(errors)}"
    )

    print(
        "\n=== CONFUSION MATRIX ==="
    )
    print(matrix)

    print("\n=== OUTPUT FILES ===")
    print(confusion_csv_path.resolve())
    print(errors_path.resolve())
    print(error_summary_path.resolve())
    print(confusion_plot_path.resolve())


if __name__ == "__main__":
    default_config = (
        find_project_root()
        / "configs"
        / "train_baseline.json"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Run error analysis for "
            "the best sentiment model."
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
