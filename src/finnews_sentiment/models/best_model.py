from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


BEST_MODEL_NAME = "linear_svc"


def build_best_pipeline(
    random_state: int,
) -> Pipeline:
    """Создаёт pipeline выбранной лучшей модели."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 1),
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    random_state=random_state,
                ),
            ),
        ]
    )
