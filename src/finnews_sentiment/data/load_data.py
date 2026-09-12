from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"text", "sentiment"}

COLUMN_ALIASES = {
    "sentence": "text",
    "headline": "text",
    "label": "sentiment",
}


def load_raw_news(path: Path) -> pd.DataFrame:
    """Load raw financial-news data from CSV."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset file does not exist: {path.resolve()}"
        )

    return pd.read_csv(path)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and known aliases."""
    result = df.copy()

    result.columns = [
        str(column).strip().lower()
        for column in result.columns
    ]

    rename_map = {}

    for source, target in COLUMN_ALIASES.items():
        if source in result.columns and target not in result.columns:
            rename_map[source] = target

    return result.rename(columns=rename_map)


def validate_schema(df: pd.DataFrame) -> None:
    """Validate the minimal dataset schema."""
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Dataset is missing required columns: {missing}"
        )


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply minimal structural cleaning for Day 1."""
    result = df.copy()

    result = result.dropna(subset=["text"])
    result = result.drop_duplicates()

    return result


def load_news_dataset(path: Path) -> pd.DataFrame:
    """Load, normalize and validate dataset."""
    df = load_raw_news(path)
    df = normalize_columns(df)
    validate_schema(df)

    return df