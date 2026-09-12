import argparse
import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from finnews_sentiment.data.load_data import (
    basic_cleaning,
    load_news_dataset,
)
from finnews_sentiment.paths import data_dir, reports_dir


SENTIMENT_ORDER = ["negative", "neutral", "positive"]


def add_text_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет длину текста в символах и количество слов."""
    result = df.copy()

    result["text_length"] = result["text"].astype(str).str.len()

    result["word_count"] = (
        result["text"]
        .astype(str)
        .str.split()
        .str.len()
    )

    return result


def plot_sentiment_distribution(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Строит и сохраняет распределение классов тональности."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    present_classes = set(
        df["sentiment"].dropna().astype(str)
    )

    order = [
        sentiment
        for sentiment in SENTIMENT_ORDER
        if sentiment in present_classes
    ]

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="sentiment",
        order=order or None,
    )

    plt.title("Sentiment class distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def plot_text_length_distribution(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Строит и сохраняет распределение длины текстов."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))

    sns.histplot(
        data=df,
        x="text_length",
        bins=50,
    )

    plt.title("Text length distribution")
    plt.xlabel("Text length, characters")
    plt.ylabel("Count")
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def build_statistics_report(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
) -> str:
    """Формирует текстовый отчёт по результатам Day 1 EDA."""
    buffer = io.StringIO()

    print("=== RAW DATA QUALITY ===", file=buffer)
    print(
        f"Rows before cleaning: {len(raw_df)}",
        file=buffer,
    )
    print(
        f"Missing text rows: {raw_df['text'].isna().sum()}",
        file=buffer,
    )
    print(
        f"Duplicated rows: {raw_df.duplicated().sum()}",
        file=buffer,
    )
    print(
        f"Rows after cleaning: {len(df)}",
        file=buffer,
    )

    print("\n=== DATASET SHAPE ===", file=buffer)
    print(df.shape, file=buffer)

    print("\n=== FIRST 5 ROWS ===", file=buffer)
    print(
        df.head().to_string(index=False),
        file=buffer,
    )

    print("\n=== DATAFRAME INFO ===", file=buffer)
    df.info(buf=buffer)

    print("\n=== MISSING VALUES AFTER CLEANING ===", file=buffer)
    print(
        df.isna().sum().to_string(),
        file=buffer,
    )

    print("\n=== DUPLICATED ROWS AFTER CLEANING ===", file=buffer)
    print(
        df.duplicated().sum(),
        file=buffer,
    )

    print("\n=== CLASS DISTRIBUTION ===", file=buffer)
    print(
        df["sentiment"]
        .value_counts(dropna=False)
        .to_string(),
        file=buffer,
    )

    print("\n=== TEXT LENGTH STATISTICS ===", file=buffer)
    print(
        df[["text_length", "word_count"]]
        .describe()
        .to_string(),
        file=buffer,
    )

    print("\n=== EXAMPLES BY CLASS ===", file=buffer)

    for sentiment in SENTIMENT_ORDER:
        examples = df.loc[
            df["sentiment"] == sentiment,
            "text",
        ]

        if not examples.empty:
            print(
                f"\n[{sentiment}]\n{examples.iloc[0]}",
                file=buffer,
            )

    return buffer.getvalue()


def main() -> None:
    default_input = (
        data_dir()
        / "raw"
        / "financial_phrasebank_75agree.csv"
    )

    default_report_dir = reports_dir()

    parser = argparse.ArgumentParser(
        description="Run Day 1 exploratory data analysis."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Path to the input CSV dataset.",
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=default_report_dir,
        help="Directory for EDA reports and figures.",
    )

    args = parser.parse_args()

    figure_dir = args.report_dir / "figures"

    raw_df = load_news_dataset(args.input)

    missing_text_count = raw_df["text"].isna().sum()
    duplicate_count = raw_df.duplicated().sum()

    print(f"Missing text rows: {missing_text_count}")
    print(f"Duplicated rows: {duplicate_count}")

    df = basic_cleaning(raw_df)
    df = add_text_statistics(df)

    print(f"Rows before cleaning: {len(raw_df)}")
    print(f"Rows after cleaning: {len(df)}")

    report = build_statistics_report(
        raw_df,
        df,
    )

    args.report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        args.report_dir
        / "day01_statistics.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    sentiment_plot_path = (
        figure_dir
        / "sentiment_distribution.png"
    )

    text_length_plot_path = (
        figure_dir
        / "text_length_distribution.png"
    )

    plot_sentiment_distribution(
        df,
        sentiment_plot_path,
    )

    plot_text_length_distribution(
        df,
        text_length_plot_path,
    )

    print(report)

    print("\n=== OUTPUT FILES ===")
    print(report_path.resolve())
    print(sentiment_plot_path.resolve())
    print(text_length_plot_path.resolve())


if __name__ == "__main__":
    main()