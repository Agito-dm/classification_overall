import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from finnews_sentiment.features.preprocess import LABEL_MAPPING
from finnews_sentiment.paths import find_project_root


LABEL_IDS = sorted(
    LABEL_MAPPING.values()
)

LABEL_NAMES = [
    name
    for name, label_id in sorted(
        LABEL_MAPPING.items(),
        key=lambda item: item[1],
    )
]


def load_config(path: Path) -> dict[str, Any]:
    """Загружает параметры baseline из JSON."""
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_processed_dataset(
    path: Path,
    text_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Загружает и проверяет processed dataset."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Processed dataset does not exist: {path.resolve()}"
        )

    df = pd.read_csv(path)

    required_columns = {
        text_col,
        target_col,
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        missing = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


def split_dataset(
    df: pd.DataFrame,
    text_col: str,
    target_col: str,
    test_size: float,
    random_state: int,
) -> tuple[
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Разделяет данные на train/test с сохранением пропорций классов."""
    X = df[text_col]
    y = df[target_col]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_baseline_pipeline(
    max_iter: int,
    random_state: int,
) -> Pipeline:
    """Создаёт baseline TF-IDF + Logistic Regression."""
    return Pipeline(
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
                ),
            ),
        ]
    )


def evaluate_model(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    """Вычисляет основные baseline-метрики."""
    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
        ),
    }


def main(config_path: Path) -> None:
    project_root = find_project_root()
    config = load_config(config_path)

    input_path = (
        project_root
        / config["input_path"]
    )

    model_path = (
        project_root
        / config["model_path"]
    )

    metrics_path = (
        project_root
        / config["metrics_path"]
    )

    classification_report_path = (
        project_root
        / config[
            "classification_report_path"
        ]
    )

    text_col = config["text_col"]
    target_col = config["target_col"]

    df = load_processed_dataset(
        input_path,
        text_col=text_col,
        target_col=target_col,
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_dataset(
        df,
        text_col=text_col,
        target_col=target_col,
        test_size=config["test_size"],
        random_state=config["random_state"],
    )

    pipeline = build_baseline_pipeline(
        max_iter=config["max_iter"],
        random_state=config["random_state"],
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    y_pred = pipeline.predict(
        X_test
    )

    metrics = evaluate_model(
        y_test,
        y_pred,
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=LABEL_IDS,
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )

    metrics.update(
        {
            "model": "LogisticRegression",
            "vectorizer": "TfidfVectorizer",
            "ngram_range": [1, 1],
            "max_iter": config["max_iter"],
            "train_size": len(X_train),
            "test_size": len(X_test),
            "test_fraction": config["test_size"],
            "random_state": config[
                "random_state"
            ],
        }
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    classification_report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        model_path,
    )

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    classification_report_path.write_text(
        report,
        encoding="utf-8",
    )

    print("=== BASELINE MODEL ===")
    print("TF-IDF + Logistic Regression")

    print(
        f"Train size: {len(X_train)}"
    )
    print(
        f"Test size: {len(X_test)}"
    )

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{metrics['weighted_f1']:.4f}"
    )

    print("\n=== CLASSIFICATION REPORT ===")
    print(report)

    print("=== OUTPUT FILES ===")
    print(model_path.resolve())
    print(metrics_path.resolve())
    print(
        classification_report_path.resolve()
    )


if __name__ == "__main__":
    default_config = (
        find_project_root()
        / "configs"
        / "train_baseline.json"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Train baseline financial "
            "sentiment classifier."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Path to baseline JSON config.",
    )

    args = parser.parse_args()

    main(args.config)
