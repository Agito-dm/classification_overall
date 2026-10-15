
import json
from pathlib import Path

from finnews_sentiment.models.predict import (
    load_model,
    predict_sentiment,
)
from finnews_sentiment.paths import find_project_root


DEMO_CASES = [
    {
        "case": "Revenue growth",
        "text": (
            "The company's revenue increased "
            "by 18 percent compared with last year."
        ),
    },
    {
        "case": "Corporate meeting",
        "text": (
            "The company will hold its annual "
            "general meeting on Friday."
        ),
    },
    {
        "case": "Falling profit",
        "text": (
            "Operating profit fell by 23 percent "
            "due to weaker demand."
        ),
    },
    {
        "case": "Lower operating costs",
        "text": (
            "Operating costs decreased by 12 percent "
            "while revenue remained stable."
        ),
    },
    {
        "case": "Mixed financial signals",
        "text": (
            "Revenue grew, but net profit declined "
            "because of higher expenses."
        ),
    },
    {
        "case": "Product announcement",
        "text": (
            "The company introduced a new "
            "smartphone model in Europe."
        ),
    },
    {
        "case": "Uncertain outlook",
        "text": (
            "Management warned that demand "
            "may weaken next quarter."
        ),
    },
]


def run_demo(
    model_path: Path,
    output_path: Path,
) -> list[dict[str, str | int]]:
    """Запускает demo и сохраняет реальные предсказания."""
    model = load_model(
        model_path
    )

    texts = [
        case["text"]
        for case in DEMO_CASES
    ]

    predictions = predict_sentiment(
        texts,
        model,
    )

    results = []

    for case, prediction in zip(
        DEMO_CASES,
        predictions,
        strict=True,
    ):
        results.append(
            {
                "case": case["case"],
                **prediction,
            }
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    return results


def main() -> None:
    project_root = find_project_root()

    model_path = (
        project_root
        / "models"
        / "best_model.joblib"
    )

    output_path = (
        project_root
        / "reports"
        / "demo_predictions.json"
    )

    results = run_demo(
        model_path,
        output_path,
    )

    print("=== FINANCIAL SENTIMENT DEMO ===")

    for result in results:
        print()
        print(f"Case: {result['case']}")
        print(f"Text: {result['text']}")
        print(f"Prediction: {result['sentiment']}")
        print(f"Label: {result['label']}")

    print()
    print(f"Results saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()

