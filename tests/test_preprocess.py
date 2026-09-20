import unittest

import pandas as pd

from finnews_sentiment.features.preprocess import (
    clean_text,
    preprocess_dataframe,
)


class PreprocessTests(unittest.TestCase):
    def test_clean_text_normalizes_html_and_whitespace(self) -> None:
        text = "  <p>COMPANY   PROFIT ROSE</p>  "

        result = clean_text(text)

        self.assertEqual(
            result,
            "company profit rose",
        )

    def test_clean_text_preserves_financial_values(self) -> None:
        text = "Shares rose to $25 by 15 % from EUR 7.5m"

        result = clean_text(text)

        self.assertEqual(
            result,
            "shares rose to $25 by 15 % from eur 7.5m",
        )

    def test_preprocess_dataframe_removes_empty_text(self) -> None:
        df = pd.DataFrame(
            {
                "text": [
                    "Profit increased",
                    "Shares rose to $25",
                    "   ",
                ],
                "sentiment": [
                    "positive",
                    "neutral",
                    "negative",
                ],
            }
        )

        result = preprocess_dataframe(df)

        self.assertEqual(len(result), 2)

        self.assertEqual(
            result["label"].tolist(),
            [2, 1],
        )

        self.assertEqual(
            result["dollar_count"].tolist(),
            [0, 1],
        )

    def test_unknown_label_raises_error(self) -> None:
        df = pd.DataFrame(
            {
                "text": ["Some financial news"],
                "sentiment": ["mixed"],
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "Unknown sentiment labels: mixed",
        ):
            preprocess_dataframe(df)

    def test_sentiment_is_normalized(self) -> None:
        df = pd.DataFrame(
            {
                "text": ["Profit increased"],
                "sentiment": ["  POSITIVE  "],
            }
        )

        result = preprocess_dataframe(df)

        self.assertEqual(
            result.loc[0, "sentiment"],
            "positive",
        )
        self.assertEqual(
            result.loc[0, "label"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
