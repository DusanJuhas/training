"""Step 6 - Final comparison table + chart from outputs/results.json."""
import json
import pandas as pd
import matplotlib.pyplot as plt
from common import OUT, RESULTS_FILE

if not RESULTS_FILE.exists():
    raise SystemExit("outputs/results.json not found - run steps 2 and 3 (and optionally 5) first.")

results_df = pd.DataFrame(json.loads(RESULTS_FILE.read_text())).T.sort_index()
results_df.index.name = "Approach"
print(results_df.to_string())
results_df.to_csv(OUT / "results_table.csv")

ax = results_df[["Accuracy", "Macro-F1"]].astype(float).plot.bar(figsize=(7, 3.5), rot=0, ylim=(0, 1))
ax.set_title("Approach comparison")
ax.grid(axis="y", alpha=.3)
plt.tight_layout()
plt.savefig(OUT / "comparison.png", dpi=120)
print("Saved", OUT / "results_table.csv", "and", OUT / "comparison.png")
print("\n-> Copy the table into REPORT.md and write your recommendation (<= 150 words)")
