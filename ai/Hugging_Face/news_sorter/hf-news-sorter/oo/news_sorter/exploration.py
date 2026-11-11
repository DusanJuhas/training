"""Step 1 of the challenge: look at the data before modelling anything."""

from pathlib import Path

import pandas as pd

from news_sorter.data import NewsData


class DataExplorer:
    """Prints a few example texts per class and checks the class balance.

    Looking at raw examples first is what lets you form a hypothesis
    ("Business and Sci/Tech will be confused, because tech-company news fits
    both") *before* seeing any model output.
    """

    def __init__(self, data: NewsData, output_dir: Path) -> None:
        self.data = data
        self.output_dir = output_dir

    def report(self, examples_per_class: int = 5) -> None:
        """Run the whole exploration: sizes, examples, class balance."""
        data = self.data
        print("Labels:", data.labels)
        print("Subset sizes:", len(data.train), len(data.test))
        self._print_examples(examples_per_class)

        balance = self.class_balance()
        print("\nClass balance:\n" + balance.to_string())
        balance.to_csv(self.output_dir / "class_balance.csv")
        print("\n-> Now write your hypothesis (which classes will be confused?) into REPORT.md")

    def class_balance(self) -> pd.DataFrame:
        """Number of examples per class in the train and the test subset.

        A balanced dataset matters for the evaluation: with balanced classes,
        plain accuracy is a trustworthy metric (on a 90/10 split, always
        answering the majority class would already score 90 %).
        """
        data = self.data
        return pd.DataFrame(
            {
                "train": pd.Series(data.label_names(data.y_train)).value_counts(),
                "test": pd.Series(data.label_names(data.y_test)).value_counts(),
            }
        ).loc[data.labels]  # .loc[...] puts the rows into the official class order

    def _print_examples(self, per_class: int) -> None:
        """Print the first ``per_class`` training texts of every class."""
        df = pd.DataFrame({"text": self.data.train_texts, "label": self.data.label_names(self.data.y_train)})
        for name in self.data.labels:
            print(f"\n===== {name} =====")
            for text in df.loc[df["label"] == name, "text"].head(per_class):
                print("-", text)
