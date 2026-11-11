"""Central configuration: every number and model name the experiment depends on.

Keeping the settings in one immutable object (instead of module-level globals
scattered over several scripts) has two benefits:

* the whole experiment is described in a single place, and
* a different setup (smaller subsets for a quick smoke test, another output
  folder, ...) is just another ``Config(...)`` instance - no code edits needed.
"""

from dataclasses import dataclass
from pathlib import Path

# Default output folder: "<...>/oo/outputs", next to the package. It is derived
# from this file's location, so the scripts work no matter which directory they
# are started from (the naive version only worked when started from its own folder).
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


@dataclass(frozen=True)  # frozen = read-only after creation, so nobody changes settings mid-run
class Config:
    """All settings of one experiment run.

    Attributes:
        seed: Random seed for shuffling the dataset and for fine-tuning. The same
            seed means every approach sees exactly the same train/test texts,
            which is what makes the comparison fair and reproducible.
        n_train: Number of labelled training examples (the full set has 120,000;
            a small subset keeps everything runnable on a CPU).
        n_test: Number of test examples all approaches are evaluated on.
        dataset_name: Hugging Face Hub id of the dataset.
        output_dir: Folder for every produced file (CSV tables, PNG plots, models).
        zero_shot_model: NLI model used by approach A.
        n_wording: How many test texts the zero-shot label-wording experiment
            uses (zero-shot is slow, so the experiment runs on a slice only).
        embedding_model: Sentence-embedding model used by approach B.
        learning_curve_sizes: Training-set sizes for approach B's learning curve.
        finetune_model: Base transformer fine-tuned by approach C.
        finetune_max_length: Token limit per text for approach C. 64 tokens are
            enough for news headlines and keep CPU training fast.
        finetune_epochs: Number of passes over the training data for approach C.
    """

    # --- data ---------------------------------------------------------------
    seed: int = 42
    n_train: int = 2000
    n_test: int = 500
    dataset_name: str = "fancyzhx/ag_news"
    output_dir: Path = DEFAULT_OUTPUT_DIR

    # --- approach A: zero-shot ------------------------------------------------
    zero_shot_model: str = "cross-encoder/nli-distilroberta-base"
    n_wording: int = 200

    # --- approach B: embeddings + logistic regression -------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # ~90 MB
    learning_curve_sizes: tuple[int, ...] = (100, 500, 2000)

    # --- approach C: fine-tuning (optional) -----------------------------------
    finetune_model: str = "distilbert-base-uncased"  # ~260 MB
    finetune_max_length: int = 64
    finetune_epochs: int = 1
