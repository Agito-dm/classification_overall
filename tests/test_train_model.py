import unittest

import pandas as pd
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)
from sklearn.linear_model import (
    LogisticRegression,
)

from finnews_sentiment.models.train_model import (
    build_baseline_pipeline,
    evaluate_model,
)


class TrainModelTests(unittest.TestCase):
    def test_baseline_pipeline_structure(self) -> None:
        pipeline = build_baseline_pipeline(
            max_iter=1000,
            random_state=42,
        )

        tfidf = pipeline.named_steps["tfidf"]
        classifier = pipeline.named_steps[
            "classifier"
        ]

        self.assertIsInstance(
            tfidf,
            TfidfVectorizer,
        )

        self.assertEqual(
            tfidf.ngram_range,
            (1, 1),
        )

        self.assertIsInstance(
            classifier,
            LogisticRegression,
        )

        self.assertEqual(
            classifier.max_iter,
            1000,
        )

    def test_baseline_pipeline_can_fit_and_predict(
        self,
    ) -> None:
        X_train = pd.Series(
            [
                "company reported strong profit",
                "sales increased significantly",
                "revenue growth continued",
                "company announced a meeting",
                "board approved the proposal",
                "company released its report",
                "company reported major losses",
                "sales declined significantly",
                "revenue fell sharply",
            ]
        )

        y_train = pd.Series(
            [
                2,
                2,
                2,
                1,
                1,
                1,
                0,
                0,
                0,
            ]
        )

        pipeline = build_baseline_pipeline(
            max_iter=1000,
            random_state=42,
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        predictions = pipeline.predict(
            pd.Series(
                [
                    "profit increased",
                    "company meeting",
                    "sales declined",
                ]
            )
        )

        self.assertEqual(
            len(predictions),
            3,
        )

        self.assertTrue(
            set(predictions).issubset(
                {0, 1, 2}
            )
        )

    def test_evaluate_model_returns_expected_metrics(
        self,
    ) -> None:
        y_true = pd.Series(
            [0, 1, 2, 2]
        )

        y_pred = pd.Series(
            [0, 1, 1, 2]
        )

        metrics = evaluate_model(
            y_true,
            y_pred,
        )

        self.assertAlmostEqual(
            metrics["accuracy"],
            0.75,
        )

        self.assertIn(
            "macro_f1",
            metrics,
        )

        self.assertIn(
            "weighted_f1",
            metrics,
        )

        for value in metrics.values():
            self.assertGreaterEqual(
                value,
                0.0,
            )
            self.assertLessEqual(
                value,
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
