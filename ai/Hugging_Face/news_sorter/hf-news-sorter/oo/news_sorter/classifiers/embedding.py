"""Approach B: sentence embeddings + a simple classifier trained on top."""

from collections.abc import Sequence

import joblib
import matplotlib

matplotlib.use("Agg")  # draw plots into files; no window (and no display) needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from news_sorter.classifiers.base import NewsClassifier
from news_sorter.config import Config
from news_sorter.data import NewsData
from news_sorter.timing import Timer


class EmbeddingClassifier(NewsClassifier):
    """Turns texts into vectors and trains logistic regression on those vectors.

    Two stages:

    1. A pre-trained *sentence-embedding* model (the "encoder") converts each
       text into a vector of 384 numbers. Texts with a similar meaning get
       similar vectors. The encoder is NOT trained here - it is used as it is.
    2. A logistic regression learns to separate the four topics in that vector
       space. This is the only part that is trained, and it takes about a second.

    The expensive language understanding is thus reused from the pre-trained
    model, and only a tiny model has to learn our specific task. That is why
    this approach needs few labelled examples and is fast at inference time
    (one encoder call per text, versus four for zero-shot).
    """

    key = "B"
    name = "B - Embeddings + LR"

    def __init__(self, config: Config, batch_size: int = 64) -> None:
        super().__init__(config)
        self.batch_size = batch_size
        self._encoder = SentenceTransformer(config.embedding_model)
        self._classifier = self._new_classifier()
        # Training embeddings are kept after fit(), so the learning curve can
        # reuse them instead of encoding 2,000 texts a second time.
        self._X_train: np.ndarray | None = None

    def fit(self, data: NewsData) -> None:
        """Encode the training texts and train the logistic regression on them."""
        # Encoding the training set is counted as training time: without it
        # there is nothing to train on.
        with Timer() as encode_timer:
            self._X_train = self.encode(data.train_texts, show_progress_bar=True)
        print("Train embedding shape:", self._X_train.shape)  # (n_train, 384)

        with Timer() as fit_timer:
            self._classifier.fit(self._X_train, data.y_train)

        self.train_time_s = encode_timer.seconds + fit_timer.seconds
        self.n_labelled = len(data.y_train)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Encode ``texts`` and classify the resulting vectors."""
        return self._classifier.predict(self.encode(texts))

    def encode(self, texts: Sequence[str], show_progress_bar: bool = False) -> np.ndarray:
        """Convert texts to embedding vectors (array of shape ``(len(texts), 384)``)."""
        return self._encoder.encode(list(texts), batch_size=self.batch_size, show_progress_bar=show_progress_bar)

    def run_experiments(self, data: NewsData) -> None:
        """Learning curve: how does accuracy grow with the number of labelled examples?

        This answers a very practical question - "how many texts do we have to
        label by hand?". If 500 examples are nearly as good as 2,000, labelling
        more is wasted effort.
        """
        if self._X_train is None:
            raise RuntimeError("Call fit() before run_experiments().")

        X_test = self.encode(data.test_texts)
        rows = []
        for n in self._curve_sizes(len(data.y_train)):
            # The training subset is already shuffled, so the first n examples
            # are a random, roughly balanced sample.
            classifier = self._new_classifier().fit(self._X_train[:n], data.y_train[:n])
            preds = classifier.predict(X_test)
            rows.append(
                {
                    "Labelled examples": n,
                    "Accuracy": round(accuracy_score(data.y_test, preds), 3),
                    "Macro-F1": round(f1_score(data.y_test, preds, average="macro"), 3),
                }
            )

        curve = pd.DataFrame(rows)
        print("\nLearning curve:\n" + curve.to_string(index=False))
        curve.to_csv(self.config.output_dir / "learning_curve.csv", index=False)
        self._plot_learning_curve(curve)

    def save(self) -> None:
        """Keep the trained logistic regression (e.g. for a Gradio demo).

        Only the small classifier is saved; the encoder can always be
        re-downloaded from the Hub by its name.
        """
        path = self.config.output_dir / "news_sorter_lr.joblib"
        joblib.dump(self._classifier, path)
        print("Saved", path)

    @staticmethod
    def _new_classifier() -> LogisticRegression:
        """A fresh, untrained logistic regression.

        ``max_iter=1000`` gives the optimiser enough steps to converge on
        384-dimensional inputs (the default of 100 may stop too early).
        """
        return LogisticRegression(max_iter=1000)

    def _curve_sizes(self, n_train: int) -> list[int]:
        """Training-set sizes for the learning curve, capped at what is available.

        The full training size is always included as the last point.
        """
        smaller = {n for n in self.config.learning_curve_sizes if n < n_train}
        return sorted(smaller | {n_train})

    def _plot_learning_curve(self, curve: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(curve["Labelled examples"], curve["Accuracy"], marker="o")
        ax.set_xscale("log")  # sizes grow multiplicatively (100, 500, 2000), so use a log axis
        ax.set_xlabel("Labelled training examples (log)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Approach B - learning curve")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = self.config.output_dir / "learning_curve.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)  # free the figure's memory
        print("Saved", path)
