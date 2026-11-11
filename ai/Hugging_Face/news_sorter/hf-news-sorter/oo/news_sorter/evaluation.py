"""Measuring one approach: metrics, confusion matrix and the list of errors."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw plots into files; no window (and no display) needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score

from news_sorter.classifiers.base import NewsClassifier
from news_sorter.data import NewsData


@dataclass(frozen=True)
class ApproachResult:
    """Everything measured for one approach - one row of the final results table.

    The first two metrics describe quality, the remaining three describe cost;
    the recommendation in REPORT.md weighs one against the other.

    Attributes:
        key: One-letter id of the approach (``"A"``, ``"B"``, ``"C"``).
        name: Human-readable name of the approach.
        accuracy: Share of test texts classified correctly.
        macro_f1: F1 score computed per class and then averaged, so every class
            counts equally (it drops when even one class is handled badly).
        sec_per_100: Inference time in seconds per 100 texts.
        train_time_s: Training time in seconds.
        n_labelled: Number of labelled examples needed.
    """

    key: str
    name: str
    accuracy: float
    macro_f1: float
    sec_per_100: float
    train_time_s: float
    n_labelled: int

    def to_row(self) -> dict:
        """The result as a table row with reader-friendly, rounded columns."""
        return {
            "Accuracy": round(self.accuracy, 3),
            "Macro-F1": round(self.macro_f1, 3),
            "Sec / 100 texts": round(self.sec_per_100, 2),
            "Training time (s)": round(self.train_time_s, 1),
            "Labelled examples": self.n_labelled,
        }


class Evaluator:
    """Evaluates predictions against the true test labels.

    One evaluator is shared by all approaches, so they are all measured in
    exactly the same way - a precondition for a fair comparison.
    """

    def __init__(self, data: NewsData, output_dir: Path) -> None:
        self.data = data
        self.output_dir = output_dir

    def evaluate(self, classifier: NewsClassifier, preds: np.ndarray, sec_per_100: float) -> ApproachResult:
        """Compute the metrics of one approach and write its analysis files.

        Args:
            classifier: The (already fitted) classifier; only its name, key and
                effort numbers are read here.
            preds: Its predictions for the test texts (class ids).
            sec_per_100: Its measured inference speed.

        Files written (``X`` = classifier key): ``preds_X.npy``,
        ``confusion_X.png``, ``errors_X.csv``.
        """
        y_true = self.data.y_test
        result = ApproachResult(
            key=classifier.key,
            name=classifier.name,
            accuracy=float(accuracy_score(y_true, preds)),
            macro_f1=float(f1_score(y_true, preds, average="macro")),
            sec_per_100=sec_per_100,
            train_time_s=classifier.train_time_s,
            n_labelled=classifier.n_labelled,
        )

        print(f"\n=== {result.name} ===")
        print(
            f"accuracy={result.accuracy:.3f}  macro-F1={result.macro_f1:.3f}  "
            f"{result.sec_per_100:.2f} s/100 texts  train={result.train_time_s:.1f} s"
        )
        # Precision / recall / F1 for every class separately.
        # labels=... keeps the report working even if a class is missing from a tiny test subset.
        class_ids = list(range(len(self.data.labels)))
        print(classification_report(y_true, preds, labels=class_ids, target_names=self.data.labels, digits=3, zero_division=0))

        # Raw predictions are kept so they can be analysed later without re-running the model.
        np.save(self.output_dir / f"preds_{result.key}.npy", np.asarray(preds))
        self._save_confusion_matrix(result, preds)
        self._save_errors(result, preds)
        return result

    def errors_table(self, preds: np.ndarray) -> pd.DataFrame:
        """All misclassified test texts with their true and predicted class.

        Reading the actual mistakes is the most instructive part of an
        evaluation: it shows *why* a model fails, not just how often.
        """
        data = self.data
        wrong = np.where(np.asarray(preds) != data.y_test)[0]  # positions where prediction != truth
        test_texts = data.test_texts
        return pd.DataFrame(
            {
                "idx": wrong,
                "true": data.label_names(data.y_test[wrong]),
                "pred": data.label_names(np.asarray(preds)[wrong]),
                "text": [test_texts[i] for i in wrong],
            }
        )

    def _save_confusion_matrix(self, result: ApproachResult, preds: np.ndarray) -> None:
        """Plot which true class (rows) was predicted as which class (columns).

        The diagonal holds the correct predictions; every off-diagonal cell is
        one specific kind of confusion (e.g. Business predicted as Sci/Tech).
        """
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            self.data.y_test,
            preds,
            labels=list(range(len(self.data.labels))),
            display_labels=self.data.labels,
            ax=ax,
            colorbar=False,
            cmap="Blues",
        )
        ax.set_title(result.name)
        fig.tight_layout()
        path = self.output_dir / f"confusion_{result.key}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print("Saved", path)

    def _save_errors(self, result: ApproachResult, preds: np.ndarray, n_shown: int = 8) -> None:
        """Write all errors to CSV and print the first few."""
        errors = self.errors_table(preds)
        path = self.output_dir / f"errors_{result.key}.csv"
        errors.to_csv(path, index=False)
        print(f"{len(errors)} errors out of {len(self.data.y_test)} (all saved to {path}). First {n_shown}:")
        for row in errors.head(n_shown).itertuples():
            print(f"  [{row.idx}] TRUE={row.true:<9} PRED={row.pred:<9} | {row.text[:160]}")
