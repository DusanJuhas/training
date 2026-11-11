"""The three classification approaches, all behind one common interface.

``FineTunedClassifier`` is deliberately *not* imported here: it pulls in the
heavy training stack (``Trainer``, ``accelerate``), which the fast default run
does not need. Import it explicitly when wanted::

    from news_sorter.classifiers.fine_tuned import FineTunedClassifier
"""

from news_sorter.classifiers.base import NewsClassifier
from news_sorter.classifiers.embedding import EmbeddingClassifier
from news_sorter.classifiers.zero_shot import ZeroShotClassifier

__all__ = ["NewsClassifier", "ZeroShotClassifier", "EmbeddingClassifier"]
