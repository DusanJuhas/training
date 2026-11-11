"""Loading the AG News dataset and holding the train/test subsets."""

from dataclasses import dataclass

import numpy as np
from datasets import Dataset, load_dataset

from news_sorter.config import Config


@dataclass(frozen=True)
class NewsData:
    """The data every approach works with: one train subset and one test subset.

    The object is created once (see :meth:`load`) and then shared by all
    classifiers, which guarantees they are trained and evaluated on exactly the
    same texts.

    Attributes:
        train: Training subset as a Hugging Face ``Dataset`` (columns ``text``, ``label``).
        test: Test subset, same structure.
        labels: Class names; the position is the class id, i.e.
            ``['World', 'Sports', 'Business', 'Sci/Tech']`` -> id 0..3.
        y_train: True class ids of the training texts (NumPy array).
        y_test: True class ids of the test texts (NumPy array).
    """

    train: Dataset
    test: Dataset
    labels: list[str]
    y_train: np.ndarray
    y_test: np.ndarray

    @classmethod
    def load(cls, config: Config) -> "NewsData":
        """Download (or read from the local cache) AG News and cut reproducible subsets.

        ``shuffle(seed=...)`` + ``select(range(n))`` picks a random sample that
        is identical on every run. Because the full dataset is perfectly
        balanced, a random sample is roughly balanced too.
        """
        dataset = load_dataset(config.dataset_name)
        train = dataset["train"].shuffle(seed=config.seed).select(range(config.n_train))
        test = dataset["test"].shuffle(seed=config.seed).select(range(config.n_test))
        return cls(
            train=train,
            test=test,
            # The label column is a ClassLabel feature, which knows the class names.
            labels=train.features["label"].names,
            y_train=np.array(train["label"]),
            y_test=np.array(test["label"]),
        )

    # The properties below return plain Python lists. Models and pipelines accept
    # lists everywhere, and a list supports normal slicing (texts[:200]).

    @property
    def train_texts(self) -> list[str]:
        """All training texts, in the same order as :attr:`y_train`."""
        return list(self.train["text"])

    @property
    def test_texts(self) -> list[str]:
        """All test texts, in the same order as :attr:`y_test`."""
        return list(self.test["text"])

    def label_names(self, class_ids) -> list[str]:
        """Translate class ids (e.g. ``[1, 3]``) to names (``['Sports', 'Sci/Tech']``)."""
        return [self.labels[int(i)] for i in class_ids]
