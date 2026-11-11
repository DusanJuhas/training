"""The final deliverable: a table and a chart comparing all approaches."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw plots into files; no window (and no display) needed
import matplotlib.pyplot as plt
import pandas as pd

from news_sorter.evaluation import ApproachResult


class ResultsReport:
    """Collects one :class:`ApproachResult` per approach and writes the comparison.

    Files written by :meth:`save`: ``results.json`` (machine-readable),
    ``results_table.csv`` (for REPORT.md) and ``comparison.png`` (bar chart).

    The report always starts empty and overwrites the files, so results of an
    earlier run can never leak into the table.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self._results: list[ApproachResult] = []

    def add(self, result: ApproachResult) -> None:
        """Register the result of one approach."""
        self._results.append(result)

    def to_frame(self) -> pd.DataFrame:
        """The comparison table: one row per approach, one column per metric."""
        table = pd.DataFrame.from_dict({r.name: r.to_row() for r in self._results}, orient="index")
        table.index.name = "Approach"
        return table.sort_index()  # names start with A/B/C, so this is the natural order

    def save(self) -> None:
        """Print the table and write the JSON, CSV and PNG files."""
        if not self._results:
            raise RuntimeError("No results to report - run at least one approach first.")

        table = self.to_frame()
        print("\n" + table.to_string())

        json_path = self.output_dir / "results.json"
        json_path.write_text(json.dumps({r.name: r.to_row() for r in self._results}, indent=2))
        table.to_csv(self.output_dir / "results_table.csv")
        self._plot_comparison(table)

        print("Saved results.json, results_table.csv and comparison.png to", self.output_dir)
        print("\n-> Copy the table into REPORT.md and write your recommendation (<= 150 words)")

    def _plot_comparison(self, table: pd.DataFrame) -> None:
        """Bar chart of the two quality metrics, one group of bars per approach."""
        # ylim=(0, 1): a full, honest scale. A truncated axis would make small
        # differences between approaches look dramatic.
        ax = table[["Accuracy", "Macro-F1"]].plot.bar(figsize=(7, 3.5), rot=0, ylim=(0, 1))
        ax.set_title("Approach comparison")
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.3)
        ax.figure.tight_layout()
        ax.figure.savefig(self.output_dir / "comparison.png", dpi=120)
        plt.close(ax.figure)
