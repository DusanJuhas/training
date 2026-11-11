"""The common interface of all classification approaches."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from news_sorter.config import Config
from news_sorter.data import NewsData
from news_sorter.timing import Timer


class NewsClassifier(ABC):
    """Abstract base class: "something that can sort news texts into the four topics".

    This is the heart of the object-oriented design. Every approach (zero-shot,
    embeddings, fine-tuning) is a subclass that fills in two methods:

    * :meth:`fit`     - prepare the classifier (train it, or pick the best prompt)
    * :meth:`predict` - turn texts into class ids

    Code that *uses* a classifier (the pipeline, the evaluator) is written
    against this base class only. It can therefore treat all approaches
    identically - this is called polymorphism.

    Life cycle, as driven by :class:`~news_sorter.pipeline.NewsSorterPipeline`::

        classifier.fit(data)                    # 1. prepare / train
        classifier.timed_predict(test_texts)    # 2. predict + measure speed
        classifier.run_experiments(data)        # 3. optional extra analysis
        classifier.save()                       # 4. optional: keep the model

    Class attributes (set by each subclass):
        key: One-letter id used in file names, e.g. ``"A"`` -> ``confusion_A.png``.
        name: Human-readable name used in tables and plot titles.

    Instance attributes (set by :meth:`fit`):
        train_time_s: Seconds spent on training (0 for approaches without training).
        n_labelled: How many labelled examples the approach needed. Together with
            the training time this is the "effort" side of the comparison.
    """

    key: str
    name: str

    def __init__(self, config: Config) -> None:
        self.config = config
        self.train_time_s: float = 0.0
        self.n_labelled: int = 0

    # ------------------------------------------------------------------
    # Methods every subclass MUST implement (@abstractmethod enforces it:
    # Python refuses to create an object of a class that still has one).
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, data: NewsData) -> None:
        """Prepare the classifier so that :meth:`predict` can be called.

        Implementations should also set :attr:`train_time_s` and :attr:`n_labelled`.
        """

    @abstractmethod
    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Classify ``texts``.

        Returns:
            A NumPy array with one class id (0..3, the index into
            ``NewsData.labels``) per input text, in the same order.
        """

    # ------------------------------------------------------------------
    # Optional hooks: a subclass overrides them only if it has something
    # to do. The default implementation does nothing.
    # ------------------------------------------------------------------

    def run_experiments(self, data: NewsData) -> None:
        """Run approach-specific extra analysis (e.g. a learning curve)."""

    def save(self) -> None:
        """Store the trained model in the output folder for later reuse."""

    # ------------------------------------------------------------------
    # Shared behaviour, inherited by all subclasses.
    # ------------------------------------------------------------------

    def timed_predict(self, texts: Sequence[str]) -> tuple[np.ndarray, float]:
        """Call :meth:`predict` and measure the inference speed.

        The time is measured end to end (for approach B this includes computing
        the embeddings), because that is what a real deployment would pay.
        It is normalised to "seconds per 100 texts" so that runs with different
        test-set sizes stay comparable.

        Returns:
            ``(predictions, seconds_per_100_texts)``
        """
        with Timer() as timer:
            preds = self.predict(texts)
        return preds, timer.seconds / len(texts) * 100
