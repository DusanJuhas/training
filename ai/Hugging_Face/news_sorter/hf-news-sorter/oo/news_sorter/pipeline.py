"""The orchestrator: runs the whole experiment from loading data to the final table."""

from collections.abc import Sequence

from news_sorter.classifiers.base import NewsClassifier
from news_sorter.config import Config
from news_sorter.data import NewsData
from news_sorter.evaluation import Evaluator
from news_sorter.exploration import DataExplorer
from news_sorter.results import ResultsReport
from news_sorter.timing import Timer


class NewsSorterPipeline:
    """Runs every given classifier through the same steps and compares the results.

    The pipeline knows the *order* of the work but none of its details: how the
    data is loaded, how a classifier predicts and how metrics are computed is
    the business of the other classes. It also does not know which approaches
    exist - it receives ready-made classifier objects from outside (this is
    called dependency injection) and only uses the ``NewsClassifier`` interface.
    """

    def __init__(self, config: Config, classifiers: Sequence[NewsClassifier]) -> None:
        self.config = config
        self.classifiers = list(classifiers)

    def run(self) -> ResultsReport:
        """Execute the experiment end to end.

        Returns:
            The report with the results of all approaches (already saved to disk).
        """
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        self._remove_stale_predictions()

        with Timer() as total:
            # Step 1: load the data once and look at it.
            self._headline("Step 1 - load and explore the data")
            data = NewsData.load(self.config)
            DataExplorer(data, output_dir).report()

            # Steps 2-5: the same four actions for every approach.
            evaluator = Evaluator(data, output_dir)
            report = ResultsReport(output_dir)
            for classifier in self.classifiers:
                self._headline(classifier.name)
                classifier.fit(data)
                preds, sec_per_100 = classifier.timed_predict(data.test_texts)
                report.add(evaluator.evaluate(classifier, preds, sec_per_100))
                classifier.run_experiments(data)
                classifier.save()

            # Step 6: the comparison of all approaches.
            self._headline("Final comparison")
            report.save()

        print(f"\nAll steps finished in {total.seconds / 60:.1f} min.")
        return report

    def _remove_stale_predictions(self) -> None:
        """Delete prediction files of earlier runs.

        Without this, ``preds_C.npy`` of an old run with fine-tuning would stay
        next to the fresh files of a run without it and look current.
        """
        for path in self.config.output_dir.glob("preds_*.npy"):
            path.unlink()

    @staticmethod
    def _headline(title: str) -> None:
        # flush=True: show the headline immediately, even when the output is
        # redirected to a file (where Python otherwise buffers it).
        print(f"\n{'=' * 70}\n>>> {title}\n{'=' * 70}", flush=True)
