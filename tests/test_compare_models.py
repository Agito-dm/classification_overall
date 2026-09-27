import unittest

import pandas as pd
from sklearn.svm import LinearSVC

from finnews_sentiment.models.compare_models import (
    add_improvement_columns,
    build_experiment_pipelines,
    run_experiments,
)


class CompareModelsTests(unittest.TestCase):
    def test_experiment_pipelines_configuration(
        self,
    ) -> None:
        pipelines = build_experiment_pipelines(
            max_iter=1000,
            random_state=42,
        )

        self.assertEqual(
            set(pipelines),
            {
                "baseline",
                "bigrams_lr",
                "balanced_lr",
                "linear_svc",
            },
        )

        baseline_tfidf = (
            pipelines["baseline"]
            .named_steps["tfidf"]
        )

        bigrams_tfidf = (
            pipelines["bigrams_lr"]
            .named_steps["tfidf"]
        )

        balanced_classifier = (
            pipelines["balanced_lr"]
            .named_steps["classifier"]
        )

        linear_svc_classifier = (
            pipelines["linear_svc"]
            .named_steps["classifier"]
        )

        self.assertEqual(
            baseline_tfidf.ngram_range,
            (1, 1),
        )

        self.assertEqual(
            bigrams_tfidf.ngram_range,
            (1, 2),
        )

        self.assertEqual(
            balanced_classifier.class_weight,
            "balanced",
        )

        self.assertIsInstance(
            linear_svc_classifier,
            LinearSVC,
        )

    def test_improvement_columns_are_calculated(
        self,
    ) -> None:
        results = pd.DataFrame(
            {
                "model": [
                    "baseline",
                    "improved",
                ],
                "accuracy": [
                    0.80,
                    0.85,
                ],
                "macro_f1": [
                    0.75,
                    0.80,
                ],
                "weighted_f1": [
                    0.79,
                    0.84,
                ],
            }
        )

        result = add_improvement_columns(
            results
        )

        improved = result.loc[
            result["model"] == "improved"
        ].iloc[0]

        self.assertAlmostEqual(
            improved[
                "macro_f1_absolute_gain"
            ],
            0.05,
        )

        self.assertAlmostEqual(
            improved[
                "macro_f1_relative_gain_pct"
            ],
            6.6666666667,
        )

    def test_experiments_can_fit_and_predict(
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

        X_test = pd.Series(
            [
                "profit increased",
                "company meeting",
                "sales declined",
            ]
        )

        y_test = pd.Series(
            [
                2,
                1,
                0,
            ]
        )

        pipelines = build_experiment_pipelines(
            max_iter=1000,
            random_state=42,
        )

        results, predictions = run_experiments(
            pipelines,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        self.assertEqual(
            len(results),
            4,
        )

        self.assertEqual(
            set(predictions),
            set(pipelines),
        )

        for prediction in predictions.values():
            self.assertEqual(
                len(prediction),
                len(X_test),
            )


if __name__ == "__main__":
    unittest.main()
