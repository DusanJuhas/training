"""Entry point: run the whole News Sorter experiment with one command.

    python run_all.py                    # approaches A + B (fast, ~15 min on a CPU)
    python run_all.py --with-finetune    # also approach C (adds ~15-30 min)

    # quick smoke test on tiny subsets, written to a separate folder:
    python run_all.py --n-train 100 --n-test 40 --n-wording 10 --output-dir outputs_smoke

All results (tables, plots, models) are written to the ``outputs`` folder next
to this script, whichever directory the script is started from.

This file only parses the command line and wires the objects together; all the
real work lives in the ``news_sorter`` package (start reading at
``news_sorter/__init__.py``).
"""

import argparse
from pathlib import Path

from news_sorter import Config, NewsSorterPipeline
from news_sorter.classifiers import EmbeddingClassifier, NewsClassifier, ZeroShotClassifier
from news_sorter.config import DEFAULT_OUTPUT_DIR


def parse_args() -> argparse.Namespace:
    """Define and read the command-line options (defaults come from ``Config``)."""
    defaults = Config()
    parser = argparse.ArgumentParser(description="Compare three Hugging Face approaches on AG News.")
    parser.add_argument("--with-finetune", action="store_true", help="also run approach C (fine-tuning, slow)")
    parser.add_argument("--n-train", type=int, default=defaults.n_train, help="training examples (default: %(default)s)")
    parser.add_argument("--n-test", type=int, default=defaults.n_test, help="test examples (default: %(default)s)")
    parser.add_argument(
        "--n-wording",
        type=int,
        default=defaults.n_wording,
        help="test texts used by the zero-shot label-wording experiment (default: %(default)s)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="folder for all produced files")
    return parser.parse_args()


def build_classifiers(config: Config, with_finetune: bool) -> list[NewsClassifier]:
    """Create the classifier objects to compare, in the order they should run."""
    classifiers: list[NewsClassifier] = [ZeroShotClassifier(config), EmbeddingClassifier(config)]
    if with_finetune:
        # Imported only when needed: it loads the heavy training stack.
        from news_sorter.classifiers.fine_tuned import FineTunedClassifier

        classifiers.append(FineTunedClassifier(config))
    return classifiers


def main() -> None:
    args = parse_args()
    config = Config(
        n_train=args.n_train,
        n_test=args.n_test,
        n_wording=args.n_wording,
        output_dir=args.output_dir.resolve(),
    )
    NewsSorterPipeline(config, build_classifiers(config, args.with_finetune)).run()


# The guard makes sure the experiment starts only when this file is executed
# directly (python run_all.py), not when it is imported from other code.
if __name__ == "__main__":
    main()
