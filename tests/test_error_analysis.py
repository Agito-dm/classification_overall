import unittest

import pandas as pd

from finnews_sentiment.models.error_analysis import (
    build_confusion_matrix,
    build_error_dataframe,
    build_error_summary,
)


class ErrorAnalysisTests(unittest.TestCase):
    def test_confusion_matrix_contains_all_samples(
        self,
    ) -> None:
        y_true = pd.Series(
            [0, 0, 1, 1, 2, 2]
        )

        y_pred = [
            0,
            1,
            1,
            2,
            2,
            1,
        ]

        matrix = build_confusion_matrix(
            y_true,
            y_pred,
        )

        self.assertEqual(
            matrix.shape,
            (3, 3),
        )

        self.assertEqual(
            int(matrix.to_numpy().sum()),
            len(y_true),
        )

        self.assertEqual(
            matrix.loc[
                "negative",
                "negative",
            ],
            1,
        )

    def test_error_dataframe_contains_only_errors(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "text": [
                    "negative example",
                    "neutral example",
                    "positive example",
                ],
                "text_clean": [
                    "negative example",
                    "neutral example",
                    "positive example",
                ],
            }
        )

        X_test = pd.Series(
            [
                "negative example",
                "neutral example",
                "positive example",
            ],
            index=[0, 1, 2],
        )

        y_test = pd.Series(
            [0, 1, 2],
            index=[0, 1, 2],
        )

        y_pred = [
            0,
            2,
            1,
        ]

        errors = build_error_dataframe(
            df,
            X_test,
            y_test,
            y_pred,
        )

        self.assertEqual(
            len(errors),
            2,
        )

        self.assertTrue(
            (
                errors["true_label"]
                != errors["predicted_label"]
            ).all()
        )

    def test_error_summary_counts_directions(
        self,
    ) -> None:
        errors = pd.DataFrame(
            {
                "true_sentiment": [
                    "positive",
                    "positive",
                    "negative",
                ],
                "predicted_sentiment": [
                    "neutral",
                    "neutral",
                    "neutral",
                ],
            }
        )

        summary = build_error_summary(
            errors
        )

        first_row = summary.iloc[0]

        self.assertEqual(
            first_row["true_sentiment"],
            "positive",
        )

        self.assertEqual(
            first_row["predicted_sentiment"],
            "neutral",
        )

        self.assertEqual(
            first_row["count"],
            2,
        )

        self.assertAlmostEqual(
            first_row[
                "share_of_errors_pct"
            ],
            66.6666666667,
        )


if __name__ == "__main__":
    unittest.main()
