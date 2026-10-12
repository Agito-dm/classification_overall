import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import joblib

from finnews_sentiment.models.best_model import build_best_pipeline
from finnews_sentiment.models.predict import (
    load_model,
    predict_sentiment,
)


class PredictTests(unittest.TestCase):
    def test_save_load_preserves_predictions(
        self,
    ) -> None:
        X_train = [
            "profit increased",
            "revenue rose significantly",
            "company announced a meeting",
            "board published a report",
            "sales declined sharply",
            "company reported heavy losses",
        ]

        y_train = [
            2,
            2,
            1,
            1,
            0,
            0,
        ]

        model = build_best_pipeline(
            random_state=42,
        )

        model.fit(
            X_train,
            y_train,
        )

        texts = [
            "Revenue rose significantly",
            "Company announced a meeting",
            "Sales declined sharply",
        ]

        predictions_before = predict_sentiment(
            texts,
            model,
        )

        with TemporaryDirectory() as temp_dir:
            model_path = (
                Path(temp_dir)
                / "test_model.joblib"
            )

            joblib.dump(
                model,
                model_path,
            )

            loaded_model = load_model(
                model_path
            )

            predictions_after = predict_sentiment(
                texts,
                loaded_model,
            )

        self.assertEqual(
            predictions_before,
            predictions_after,
        )

    def test_inference_applies_preprocessing(
        self,
    ) -> None:
        model = Mock()

        model.predict.return_value = [
            2,
            1,
        ]

        texts = [
            "<b>REVENUE</b>   GREW",
            "Board    MEETING",
        ]

        results = predict_sentiment(
            texts,
            model,
        )

        model.predict.assert_called_once_with(
            [
                "revenue grew",
                "board meeting",
            ]
        )

        self.assertEqual(
            results,
            [
                {
                    "text": texts[0],
                    "sentiment": "positive",
                    "label": 2,
                },
                {
                    "text": texts[1],
                    "sentiment": "neutral",
                    "label": 1,
                },
            ],
        )

    def test_rejects_invalid_texts(
        self,
    ) -> None:
        model = Mock()

        with self.assertRaises(ValueError):
            predict_sentiment(
                [],
                model,
            )

        with self.assertRaises(ValueError):
            predict_sentiment(
                ["   "],
                model,
            )

        with self.assertRaises(ValueError):
            predict_sentiment(
                ["Valid text", None],
                model,
            )

        model.predict.assert_not_called()

    def test_missing_model_raises_error(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            model_path = (
                Path(temp_dir)
                / "missing_model.joblib"
            )

            with self.assertRaises(
                FileNotFoundError
            ):
                load_model(
                    model_path
                )

    def test_rejects_non_pipeline_artifact(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            model_path = (
                Path(temp_dir)
                / "invalid_model.joblib"
            )

            joblib.dump(
                {"not": "a pipeline"},
                model_path,
            )

            with self.assertRaises(TypeError):
                load_model(
                    model_path
                )


if __name__ == "__main__":
    unittest.main()

