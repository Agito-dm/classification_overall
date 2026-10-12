import argparse
from pathlib import Path

import joblib

from finnews_sentiment.models.best_model import (
    BEST_MODEL_NAME,
    build_best_pipeline,
)
from finnews_sentiment.models.train_model import (
    load_config,
    load_processed_dataset,
)
from finnews_sentiment.paths import find_project_root


def train_best_model(
    config_path: Path,
) -> Path:
    """Обучает лучшую модель на всех данных и сохраняет Pipeline."""
    project_root = find_project_root()

    if not config_path.is_absolute():
        config_path = project_root / config_path

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

    X = df[config["text_col"]]
    y = df[config["target_col"]]

    model = build_best_pipeline(
        random_state=config["random_state"],
    )

    model.fit(
        X,
        y,
    )

    model_path = (
        project_root
        / "models"
        / "best_model.joblib"
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        model_path,
    )

    print("=== FINAL MODEL TRAINING ===")
    print(f"Model: {BEST_MODEL_NAME}")
    print(f"Training samples: {len(df)}")
    print(f"Classes: {sorted(y.unique().tolist())}")
    print(f"Model saved to: {model_path.resolve()}")

    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train and save the final "
            "financial sentiment model."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/train_baseline.json"
        ),
        help="Path to training config.",
    )

    args = parser.parse_args()

    train_best_model(
        args.config
    )
