import argparse
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from finnews_sentiment.features.preprocess import clean_text
from finnews_sentiment.models.train_model import (
    LABEL_IDS,
    LABEL_NAMES,
)
from finnews_sentiment.paths import find_project_root


ID_TO_LABEL = dict(
    zip(
        LABEL_IDS,
        LABEL_NAMES,
    )
)


def load_model(model_path: Path) -> Pipeline:
    """Загружает сохранённый обученный Pipeline."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Run train_best_model first."
        )

    model = joblib.load(
        model_path
    )

    if not isinstance(model, Pipeline):
        raise TypeError(
            "Saved model must be a sklearn Pipeline."
        )

    return model


def predict_sentiment(
    texts: list[str],
    model: Pipeline,
) -> list[dict[str, str | int]]:
    """Классифицирует новые тексты с preprocessing."""
    if not texts:
        raise ValueError(
            "At least one text is required."
        )

    cleaned_texts = []

    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise ValueError(
                f"Text at index {index} must be a string."
            )

        cleaned_text = clean_text(
            text
        )

        if not cleaned_text:
            raise ValueError(
                f"Text at index {index} is empty after cleaning."
            )

        cleaned_texts.append(
            cleaned_text
        )

    predictions = model.predict(
        cleaned_texts
    )

    results = []

    for text, prediction in zip(
        texts,
        predictions,
    ):
        label = int(prediction)

        results.append(
            {
                "text": text,
                "sentiment": ID_TO_LABEL[label],
                "label": label,
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Predict financial news sentiment "
            "using a saved model."
        )
    )

    parser.add_argument(
        "--text",
        action="append",
        required=True,
        help=(
            "Financial text to classify. "
            "Repeat for multiple texts."
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "models/best_model.joblib"
        ),
        help="Path to saved model.",
    )

    args = parser.parse_args()

    project_root = find_project_root()
    model_path = args.model

    if not model_path.is_absolute():
        model_path = (
            project_root
            / model_path
        )

    try:
        model = load_model(
            model_path
        )

        results = predict_sentiment(
            args.text,
            model,
        )

    except (FileNotFoundError, TypeError, ValueError) as error:
        parser.error(
            str(error)
        )

    print("=== SENTIMENT PREDICTIONS ===")

    for result in results:
        print()
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Label: {result['label']}")


if __name__ == "__main__":
    main()
