"""Approach C (optional): fine-tuning a small transformer on the task."""

from collections.abc import Sequence

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from news_sorter.classifiers.base import NewsClassifier
from news_sorter.config import Config
from news_sorter.data import NewsData
from news_sorter.timing import Timer


class FineTunedClassifier(NewsClassifier):
    """DistilBERT with a new 4-class head, trained end to end on the labelled data.

    Unlike approach B, where the pre-trained model stays frozen, fine-tuning
    updates *all* weights of the transformer so that it specialises in this
    task. It usually reaches the best accuracy, but costs by far the most
    training time (about 15-30 minutes for one epoch on a CPU).
    """

    key = "C"
    name = "C - Fine-tuned DistilBERT"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        # The tokenizer splits text into the sub-word tokens the model was
        # pre-trained with; it must always match the model.
        self._tokenizer = AutoTokenizer.from_pretrained(config.finetune_model)
        self._trainer: Trainer | None = None  # created in fit()

    def fit(self, data: NewsData) -> None:
        """Fine-tune the model on the training subset."""
        # A classification head with 4 outputs is put on top of the pre-trained
        # body. The head starts with random weights, hence the expected warning
        # "Some weights ... were not initialized".
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.finetune_model,
            num_labels=len(data.labels),
            # Store the class names inside the model, so that it reports
            # "Sports" instead of "LABEL_1" when reused elsewhere.
            id2label=dict(enumerate(data.labels)),
            label2id={label: i for i, label in enumerate(data.labels)},
        )

        args = TrainingArguments(
            output_dir=str(self.config.output_dir / "distilbert-checkpoints"),
            num_train_epochs=self.config.finetune_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=5e-5,  # small on purpose: adapt the pre-trained weights, do not destroy them
            weight_decay=0.01,
            logging_steps=25,
            save_strategy="no",  # no intermediate checkpoints on disk
            report_to="none",  # no external loggers (wandb, ...)
            seed=self.config.seed,
        )
        self._trainer = Trainer(
            model=model,
            args=args,
            train_dataset=self._tokenize(data.train),
            # Pads every batch only to the length of its longest text, which is
            # faster than padding everything to the global maximum.
            data_collator=DataCollatorWithPadding(self._tokenizer),
        )

        with Timer() as timer:
            self._trainer.train()
        print(f"Training took {timer.seconds / 60:.1f} min")

        self.train_time_s = timer.seconds
        self.n_labelled = len(data.y_train)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Classify ``texts`` with the fine-tuned model."""
        if self._trainer is None:
            raise RuntimeError("Call fit() before predict().")
        dataset = self._tokenize(Dataset.from_dict({"text": list(texts)}))
        # The model outputs one score ("logit") per class for every text; the
        # class with the highest score is the prediction.
        logits = self._trainer.predict(dataset).predictions
        return logits.argmax(axis=-1)

    def save(self) -> None:
        """Save model + tokenizer together (ready for reuse or ``push_to_hub``)."""
        if self._trainer is None:
            raise RuntimeError("Call fit() before save().")
        path = str(self.config.output_dir / "distilbert-agnews")
        self._trainer.save_model(path)
        self._tokenizer.save_pretrained(path)
        print("Model saved to", path)

    def _tokenize(self, dataset: Dataset) -> Dataset:
        """Add the tokenizer's columns (``input_ids``, ``attention_mask``) to a dataset.

        Texts longer than ``finetune_max_length`` tokens are cut off
        (``truncation=True``); padding is left to the data collator.
        """

        def tokenize(batch: dict) -> dict:
            return self._tokenizer(batch["text"], truncation=True, max_length=self.config.finetune_max_length)

        return dataset.map(tokenize, batched=True)
