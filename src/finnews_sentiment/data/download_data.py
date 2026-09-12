import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from finnews_sentiment.paths import data_dir


DATASET_NAME = "takala/financial_phrasebank"
DEFAULT_CONFIG = "sentences_75agree"
DATASET_REVISION = "4f231b6c1b92db2a943966b01cc412cebb49f0f9"


def download_financial_phrasebank(
    output_path: Path,
    config: str = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Download Financial PhraseBank and save it in project CSV format."""

    dataset = load_dataset(
        DATASET_NAME,
        config,
        split="train",
        revision=DATASET_REVISION,
    )

    label_names = dataset.features["label"].names

    df = dataset.to_pandas()[["sentence", "label"]].copy()

    df["sentiment"] = df["label"].map(
        lambda label: label_names[int(label)]
    )

    df = (
        df.rename(columns={"sentence": "text"})
        [["text", "sentiment"]]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


def main() -> None:
    default_output = (
        data_dir()
        / "raw"
        / "financial_phrasebank_75agree.csv"
    )

    parser = argparse.ArgumentParser(
        description="Download Financial PhraseBank dataset."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Financial PhraseBank configuration.",
    )

    args = parser.parse_args()

    df = download_financial_phrasebank(
        output_path=args.output,
        config=args.config,
    )

    print(f"Dataset: {DATASET_NAME}")
    print(f"Config: {args.config}")
    print(f"Revision: {DATASET_REVISION}")
    print(f"Rows: {len(df)}")
    print(f"Saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
