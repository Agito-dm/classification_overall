
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib

from finnews_sentiment.demo import (
    DEMO_CASES,
    run_demo,
)
from finnews_sentiment.models.best_model import (
    build_best_pipeline,
)
from finnews_sentiment.models.train_model import (
    LABEL_IDS,
    LABEL_NAMES,
)


class DemoTests(unittest.TestCase):
    def test_demo_cases_are_valid(self) -> None:
        self.assertEqual(
            len(DEMO_CASES),
            7,
        )

        names = [
            case["case"]
            for case in DEMO_CASES
        ]

        texts = [
            case["text"]
            for case in DEMO_CASES
        ]

        self.assertEqual(
            len(names),
            len(set(names)),
        )

        self.assertEqual(
            len(texts),
            len(set(texts)),
        )

        for case in DEMO_CASES:
            self.assertTrue(
                case["case"].strip()
            )

            self.assertTrue(
                case["text"].strip()
            )

    def test_demo_saves_real_predictions(self) -> None:
        X_train = [
            "company reported strong profit",
            "revenue increased significantly",
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

        id_to_label = dict(
            zip(
                LABEL_IDS,
                LABEL_NAMES,
            )
        )

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            model_path = (
                temp_path
                / "best_model.joblib"
            )

            output_path = (
                temp_path
                / "reports"
                / "demo_predictions.json"
            )

            joblib.dump(
                model,
                model_path,
            )

            results = run_demo(
                model_path,
                output_path,
            )

            self.assertTrue(
                output_path.is_file()
            )

            saved_results = json.loads(
                output_path.read_text(
                    encoding="utf-8",
                )
            )

            self.assertEqual(
                results,
                saved_results,
            )

            self.assertEqual(
                len(results),
                len(DEMO_CASES),
            )

            for case, result in zip(
                DEMO_CASES,
                results,
            ):
                self.assertEqual(
                    result["case"],
                    case["case"],
                )

                self.assertEqual(
                    result["text"],
                    case["text"],
                )

                self.assertIn(
                    result["label"],
                    LABEL_IDS,
                )

                self.assertEqual(
                    result["sentiment"],
                    id_to_label[result["label"]],
                )


if __name__ == "__main__":
    unittest.main()

