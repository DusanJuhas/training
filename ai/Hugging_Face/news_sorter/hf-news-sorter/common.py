"""Shared settings and helper functions used by all step scripts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # draw plots into files, no window needed
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

SEED = 42
N_TRAIN, N_TEST = 2000, 500

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
RESULTS_FILE = OUT / "results.json"

pd.set_option("display.max_colwidth", 200)
pd.set_option("display.width", 200)


def load_data():
    """Load AG News and return reproducible subsets (same for every script)."""
    ds = load_dataset("fancyzhx/ag_news")
    train = ds["train"].shuffle(seed=SEED).select(range(N_TRAIN))
    test = ds["test"].shuffle(seed=SEED).select(range(N_TEST))
    labels = train.features["label"].names          # ['World', 'Sports', 'Business', 'Sci/Tech']
    y_train = np.array(train["label"])
    y_test = np.array(test["label"])
    return train, test, labels, y_train, y_test


def save_preds(key, preds):
    np.save(OUT / f"preds_{key}.npy", np.asarray(preds))


def load_preds(key):
    path = OUT / f"preds_{key}.npy"
    return np.load(path) if path.exists() else None


def save_result(name, y_true, preds, sec_per_100, train_time_s, n_labelled):
    """Print metrics and store one row of the final results table in outputs/results.json."""
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="macro")
    print(f"{name}: accuracy={acc:.3f}  macro-F1={f1:.3f}  "
          f"{sec_per_100:.2f} s/100 texts  train={train_time_s:.1f} s")
    results = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else {}
    results[name] = {
        "Accuracy": round(float(acc), 3),
        "Macro-F1": round(float(f1), 3),
        "Sec / 100 texts": round(float(sec_per_100), 2),
        "Training time (s)": round(float(train_time_s), 1),
        "Labelled examples": int(n_labelled),
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2))


def save_confusion(y_true, preds, labels, title, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, preds, display_labels=labels, ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=120)
    plt.close(fig)
    print("Saved", OUT / filename)


def errors_table(test, y_true, preds, labels):
    """All misclassified test examples as a DataFrame."""
    wrong = np.where(np.asarray(preds) != y_true)[0]
    return pd.DataFrame({
        "idx": wrong,
        "true": [labels[y_true[i]] for i in wrong],
        "pred": [labels[preds[i]] for i in wrong],
        "text": [test[int(i)]["text"] for i in wrong],
    })
