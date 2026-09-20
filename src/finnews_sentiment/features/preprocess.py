import re

import argparse
from pathlib import Path
import pandas as pd
from finnews_sentiment.data.load_data import (
    basic_cleaning,
    load_news_dataset,
)
from finnews_sentiment.paths import (
    data_dir,
    reports_dir,
)


LABEL_MAPPING = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: object) -> str:
    """Выполняет базовую очистку одного текста."""
    if not isinstance(text, str):
        return ""

    cleaned = text.lower()

    cleaned = HTML_TAG_PATTERN.sub(
        " ",
        cleaned,
    )

    cleaned = WHITESPACE_PATTERN.sub(
        " ",
        cleaned,
    )

    return cleaned.strip()


def normalize_sentiment(value: object) -> str:
    """Нормализует текстовую метку класса."""
    if not isinstance(value, str):
        return ""

    return value.strip().lower()


def encode_labels(
    df: pd.DataFrame,
    sentiment_col: str = "sentiment",
) -> pd.DataFrame:
    """Преобразует текстовые sentiment labels в числовые."""
    result = df.copy()

    if sentiment_col not in result.columns:
        raise ValueError(
            f"Column '{sentiment_col}' is missing."
        )

    result[sentiment_col] = (
        result[sentiment_col]
        .map(normalize_sentiment)
    )

    unknown_labels = (
        set(result[sentiment_col].unique())
        - set(LABEL_MAPPING)
    )

    if unknown_labels:
        unknown = ", ".join(
            sorted(str(label) for label in unknown_labels)
        )

        raise ValueError(
            f"Unknown sentiment labels: {unknown}"
        )

    result["label"] = (
        result[sentiment_col]
        .map(LABEL_MAPPING)
        .astype("int64")
    )

    return result


def add_basic_features(
    df: pd.DataFrame,
    text_col: str = "text_clean",
) -> pd.DataFrame:
    """Добавляет простые признаки очищенного текста."""
    result = df.copy()

    if text_col not in result.columns:
        raise ValueError(
            f"Column '{text_col}' is missing."
        )

    result["word_count_clean"] = (
        result[text_col]
        .str.split()
        .str.len()
    )

    result["char_count"] = (
        result[text_col]
        .str.len()
    )

    result["dollar_count"] = (
        result[text_col]
        .str.count(r"\$")
    )

    result["number_count"] = (
        result[text_col]
        .str.findall(r"\d+(?:\.\d+)?")
        .str.len()
    )

    return result


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    sentiment_col: str = "sentiment",
) -> pd.DataFrame:
    """Выполняет preprocessing датасета для классификации."""
    result = df.copy()

    required_columns = {
        text_col,
        sentiment_col,
    }

    missing_columns = (
        required_columns
        - set(result.columns)
    )

    if missing_columns:
        missing = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    result["text_clean"] = (
        result[text_col]
        .map(clean_text)
    )

    result = result[
        result["text_clean"].str.len() > 0
    ].copy()

    result = encode_labels(
        result,
        sentiment_col=sentiment_col,
    )

    result = add_basic_features(
        result,
        text_col="text_clean",
    )

    result = result.reset_index(drop=True)

    return result

def get_preprocessing_diagnostics(
    df: pd.DataFrame,
) -> dict[str, int]:
    """Собирает диагностические показатели preprocessing."""
    duplicate_clean_texts = (
        df.duplicated(
            subset=["text_clean"]
        ).sum()
    )

    duplicate_clean_text_labels = (
        df.duplicated(
            subset=[
                "text_clean",
                "sentiment",
            ]
        ).sum()
    )

    labels_per_text = (
        df.groupby("text_clean")["sentiment"]
        .nunique()
    )

    conflicting_texts = (
        labels_per_text > 1
    ).sum()

    return {
        "duplicate_clean_texts": int(
            duplicate_clean_texts
        ),
        "duplicate_clean_text_labels": int(
            duplicate_clean_text_labels
        ),
        "conflicting_texts": int(
            conflicting_texts
        ),
    }


def build_preprocessing_report(
    raw_rows: int,
    structural_rows: int,
    df: pd.DataFrame,
) -> str:
    """Формирует текстовый отчёт Day 2."""
    diagnostics = get_preprocessing_diagnostics(
        df
    )

    empty_removed = (
        structural_rows
        - len(df)
    )

    lines = [
        "=== DAY 2 PREPROCESSING ===",
        f"Rows in raw dataset: {raw_rows}",
        (
            "Rows after Day 1 structural cleaning: "
            f"{structural_rows}"
        ),
        (
            "Empty texts removed after preprocessing: "
            f"{empty_removed}"
        ),
        (
            "Rows after preprocessing: "
            f"{len(df)}"
        ),
        (
            "Duplicate cleaned texts: "
            f"{diagnostics['duplicate_clean_texts']}"
        ),
        (
            "Duplicate cleaned text + label: "
            f"{diagnostics['duplicate_clean_text_labels']}"
        ),
        (
            "Cleaned texts with conflicting labels: "
            f"{diagnostics['conflicting_texts']}"
        ),
        "",
        "=== LABEL DISTRIBUTION ===",
        df["sentiment"]
        .value_counts()
        .to_string(),
        "",
        "=== PROCESSED COLUMNS ===",
        ", ".join(df.columns),
    ]

    return "\n".join(lines)


def save_examples_report(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Сохраняет примеры preprocessing для ручной проверки."""
    sections = [
        "# Day 2 preprocessing examples",
        "",
    ]

    for sentiment in LABEL_MAPPING:
        examples = df.loc[
            df["sentiment"] == sentiment
        ]

        if examples.empty:
            continue

        row = examples.iloc[0]

        sections.extend(
            [
                f"## {sentiment}",
                "",
                f"Original: {row['text']}",
                "",
                f"Cleaned: {row['text_clean']}",
                "",
                f"Label: {row['label']}",
                "",
            ]
        )

    output_path.write_text(
        "\n".join(sections),
        encoding="utf-8",
    )


def main() -> None:
    default_input = (
        data_dir()
        / "raw"
        / "financial_phrasebank_75agree.csv"
    )

    default_output = (
        data_dir()
        / "processed"
        / "financial_phrasebank_75agree_processed.csv"
    )

    parser = argparse.ArgumentParser(
        description="Run Day 2 text preprocessing."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Path to raw input CSV.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Path to processed CSV.",
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=reports_dir(),
        help="Directory for preprocessing reports.",
    )

    args = parser.parse_args()

    raw_df = load_news_dataset(
        args.input
    )

    structural_df = basic_cleaning(
        raw_df
    )

    processed_df = preprocess_dataframe(
        structural_df
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_df.to_csv(
        args.output,
        index=False,
    )

    args.report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = build_preprocessing_report(
        raw_rows=len(raw_df),
        structural_rows=len(structural_df),
        df=processed_df,
    )

    report_path = (
        args.report_dir
        / "day02_preprocessing_report.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    examples_path = (
        args.report_dir
        / "day02_examples.md"
    )

    save_examples_report(
        processed_df,
        examples_path,
    )

    print(report)

    print("\n=== OUTPUT FILES ===")
    print(args.output.resolve())
    print(report_path.resolve())
    print(examples_path.resolve())


if __name__ == "__main__":
    main()
