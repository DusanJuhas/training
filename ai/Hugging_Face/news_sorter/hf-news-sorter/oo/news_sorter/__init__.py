"""News Sorter - an object-oriented solution of the Hugging Face AG News challenge.

The challenge: route short news texts to one of four topics (World, Sports,
Business, Sci/Tech) and find out which Hugging Face approach gives the best
accuracy for the least effort, on an ordinary CPU-only laptop.

How the package is organised (one responsibility per module)::

    config.py        Config            - every tunable number/model name in one place
    data.py          NewsData          - loads AG News, owns the train/test subsets
    exploration.py   DataExplorer      - step 1: example texts + class balance
    classifiers/     NewsClassifier    - abstract base: fit() + predict()
                     ZeroShotClassifier     (approach A, no training)
                     EmbeddingClassifier    (approach B, embeddings + log. regression)
                     FineTunedClassifier    (approach C, fine-tuned DistilBERT)
    evaluation.py    Evaluator         - metrics, confusion matrix, error list
                     ApproachResult    - the numbers measured for one approach
    results.py       ResultsReport     - final comparison table + bar chart
    pipeline.py      NewsSorterPipeline - runs everything in the right order
    timing.py        Timer             - tiny stopwatch context manager

The key design idea: all three approaches implement the same small interface
(:class:`~news_sorter.classifiers.base.NewsClassifier`). The pipeline, the
evaluator and the report only ever talk to that interface, so they do not care
*how* a classifier works - and adding a fourth approach means writing one new
subclass, without touching any other file.

Typical use (this is what ``run_all.py`` does)::

    from news_sorter import Config, NewsSorterPipeline
    from news_sorter.classifiers import ZeroShotClassifier, EmbeddingClassifier

    config = Config()
    pipeline = NewsSorterPipeline(config, [ZeroShotClassifier(config), EmbeddingClassifier(config)])
    pipeline.run()
"""

from news_sorter.config import Config
from news_sorter.data import NewsData
from news_sorter.pipeline import NewsSorterPipeline

__all__ = ["Config", "NewsData", "NewsSorterPipeline"]
