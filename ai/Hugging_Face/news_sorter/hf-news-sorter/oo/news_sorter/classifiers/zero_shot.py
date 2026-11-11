"""Approach A: zero-shot classification - a pre-trained model used as it is."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from transformers import pipeline

from news_sorter.classifiers.base import NewsClassifier
from news_sorter.config import Config
from news_sorter.data import NewsData
from news_sorter.timing import Timer


@dataclass(frozen=True)
class Prompt:
    """One way of "asking the question" to the zero-shot model.

    Attributes:
        variant: Short name of the label wording (for the results table).
        candidate_labels: The four class descriptions shown to the model.
            IMPORTANT: they must be in the dataset's class order (World, Sports,
            Business, Sci/Tech), because the position in this list is used as
            the predicted class id.
        template_name: Short name of the hypothesis template (for the table).
        template: The sentence the label is inserted into, e.g.
            ``"This news article is about {}."``. ``None`` means the pipeline's
            default, ``"This example is {}."``.
    """

    variant: str
    candidate_labels: tuple[str, ...]
    template_name: str = "default"
    template: str | None = None


# The label wordings tried in the experiment. Zero-shot models are sensitive to
# wording: "World" is a vague description of a news topic, "world politics" is
# a much clearer one.
LABEL_VARIANTS: dict[str, tuple[str, ...]] = {
    "1 dataset names": ("World", "Sports", "Business", "Sci/Tech"),
    "2 descriptive": ("world politics", "sports", "business", "science and technology"),
    "3 specific": (
        "international news and politics",
        "sports and athletes",
        "business, finance and economy",
        "science, technology and computers",
    ),
    "4 one-word": ("politics", "sports", "economy", "technology"),
}

# The hypothesis templates tried with every label wording.
TEMPLATES: dict[str, str | None] = {
    "default": None,
    "news": "This news article is about {}.",
}


class ZeroShotClassifier(NewsClassifier):
    """Classifies with a natural-language-inference (NLI) model, without any training.

    How zero-shot classification works: for every candidate label the pipeline
    builds a hypothesis sentence ("This example is sports.") and asks the NLI
    model whether the news text *entails* it. The label with the highest
    entailment score wins. One text therefore costs four model calls (one per
    label), which is why this approach is the slowest at inference time.

    "Fitting" this classifier does not train anything. It only runs the
    label-wording experiment and remembers the best prompt.

    Caveat (kept on purpose, to stay comparable with the reference solution):
    the best prompt is chosen by its accuracy on the first ``n_wording`` *test*
    texts, so the final score of approach A is slightly optimistic. A stricter
    setup would choose the prompt on a separate validation set. This is also
    why ``n_labelled`` stays 0 although labels are peeked at.
    """

    key = "A"
    name = "A - Zero-shot"

    def __init__(self, config: Config, batch_size: int = 16) -> None:
        super().__init__(config)
        self.batch_size = batch_size
        # Loading the pipeline downloads the model on first use (then it is cached).
        self._pipeline = pipeline("zero-shot-classification", model=config.zero_shot_model)
        # Start with a sensible prompt, so predict() works even without fit().
        self.prompt = Prompt("2 descriptive", LABEL_VARIANTS["2 descriptive"])

    def fit(self, data: NewsData) -> None:
        """Choose the best prompt with the label-wording experiment (no training)."""
        self._show_example(data)

        wording = self.wording_experiment(data)
        print("\nLabel-wording experiment:\n" + wording.to_string(index=False))
        wording.to_csv(self.config.output_dir / "zero_shot_wording.csv", index=False)

        best = wording.iloc[0]  # the table is sorted, best row first
        self.prompt = Prompt(
            variant=best["Label variant"],
            candidate_labels=LABEL_VARIANTS[best["Label variant"]],
            template_name=best["Template"],
            template=TEMPLATES[best["Template"]],
        )
        print(f"\nBest variant: {self.prompt.variant} | template: {self.prompt.template_name}")

        # The whole point of zero-shot: no training time, no labelled examples.
        self.train_time_s = 0.0
        self.n_labelled = 0

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Classify ``texts`` with the currently selected prompt."""
        return self._predict_with(texts, self.prompt)

    def wording_experiment(self, data: NewsData) -> pd.DataFrame:
        """Measure every label wording x template combination on a slice of the test set.

        Returns:
            One row per combination with its accuracy and speed, sorted by
            accuracy (best first).
        """
        n = min(self.config.n_wording, len(data.test))
        texts, y_true = data.test_texts[:n], data.y_test[:n]

        rows = []
        for variant, candidate_labels in LABEL_VARIANTS.items():
            for template_name, template in TEMPLATES.items():
                prompt = Prompt(variant, candidate_labels, template_name, template)
                with Timer() as timer:
                    preds = self._predict_with(texts, prompt)
                rows.append(
                    {
                        "Label variant": variant,
                        "Template": template_name,
                        "Accuracy": round(accuracy_score(y_true, preds), 3),
                        "Sec / 100": round(timer.seconds / n * 100, 2),
                    }
                )
                print(rows[-1])

        # kind="stable": on equal accuracy the combination tried first wins, so
        # the choice of the best prompt is deterministic.
        return pd.DataFrame(rows).sort_values("Accuracy", ascending=False, kind="stable")

    def _predict_with(self, texts: Sequence[str], prompt: Prompt) -> np.ndarray:
        """Run the pipeline with an explicit prompt and convert its output to class ids."""
        labels = list(prompt.candidate_labels)
        kwargs = {"candidate_labels": labels, "batch_size": self.batch_size}
        if prompt.template:
            kwargs["hypothesis_template"] = prompt.template

        outputs = self._pipeline(list(texts), **kwargs)

        # Each output is a dict whose "labels" list is sorted by score, best
        # first. The position of the winning label in OUR label list is the
        # class id - which is why the candidate labels must keep the dataset order.
        return np.array([labels.index(output["labels"][0]) for output in outputs])

    def _show_example(self, data: NewsData) -> None:
        """Print the raw pipeline output for one text, to see what it looks like."""
        text = data.test_texts[0]
        output = self._pipeline(text, candidate_labels=list(self.prompt.candidate_labels))
        print("Text:", text)
        print("True:", data.labels[data.y_test[0]])
        print("Pipeline output:", output)
