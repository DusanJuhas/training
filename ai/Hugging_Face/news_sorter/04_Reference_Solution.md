# Level 4 — Your First Hugging Face NLP Pipeline: Reference Solution (plain Python scripts)

> This is a complete, working solution built from **plain Python scripts**, with no Jupyter required.
> Each file below is shown in full: create it in your project folder with any text editor, then run it from the terminal.
> Don't understand a file? → The matching section in [**Level 3**](03_Detailed_Beginner_Guide.md) explains it line by line.
> Want to know *why* a step exists? → [**Level 2**](02_Structured_Steps.md) describes the goal of each step.

**Checked with:** Python 3.13, `torch` 2.15, `transformers` 5.22, `accelerate`, `datasets` 5.1, `sentence-transformers` 6.2, `scikit-learn` 1.x. Any Python 3.10+ with recent versions should work.
**Total runtime on a 4-core laptop CPU:** about 10 min without Step 5, about 25–40 min with Step 5 (the first run is longer because models download).

---

## 0. Preparation: environment and libraries

Run in a **terminal**:

```bash
# 1) Create the project folder and a virtual environment
mkdir hf-news-sorter
cd hf-news-sorter
python -m venv hf-env

# 2) Activate it (every new terminal!)
# Windows:      hf-env\Scripts\activate
# macOS/Linux:  source hf-env/bin/activate

# 3) CPU build of PyTorch (smaller and faster to install than the CUDA build)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4) Everything else (accelerate is required by the Trainer API in Step 5)
pip install transformers accelerate datasets evaluate sentence-transformers scikit-learn pandas matplotlib

# 5) Optional, for the stretch goals
pip install gradio
```

### Project layout

```
hf-news-sorter/
├── hf-env/                  # virtual environment (created above)
├── common.py                # shared settings + helpers (used by every step)
├── step1_explore.py         # Step 1 – load and explore the data
├── step2_zero_shot.py       # Step 2 – approach A
├── step3_embeddings.py      # Step 3 – approach B
├── step4_evaluate.py        # Step 4 – metrics, confusion matrices, errors
├── step5_finetune.py        # Step 5 – approach C (optional)
├── step6_results.py         # Step 6 – final results table + chart
├── run_all.py               # runs all steps in order
├── REPORT.md                # your written answers (hypothesis, errors, recommendation)
└── outputs/                 # created automatically: CSV, PNG, JSON, saved models
```

**How to run any script:** open a terminal, `cd` into `hf-news-sorter`, activate `hf-env`, then run for example:

```bash
python step1_explore.py
```

Always run scripts **from the project folder**, so that `import common` and the `outputs/` folder are found.

**How the scripts share results:** each step saves what later steps need (predictions as `.npy`, metrics in `outputs/results.json`). That's why you can rerun any single step without repeating the slow ones.

---

## 1. `common.py`: shared settings and helpers

```python
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
```

---

## 2. `step1_explore.py`: load and explore the data

```python
"""Step 1 - Load and explore the data."""
import pandas as pd
from common import load_data, OUT

train, test, labels, y_train, y_test = load_data()
print("Labels:", labels)
print("Subset sizes:", len(train), len(test))

# 5 examples per class
df_train = train.to_pandas()
df_train["label_name"] = df_train["label"].map(lambda i: labels[i])
for name in labels:
    print(f"\n===== {name} =====")
    for t in df_train.loc[df_train["label_name"] == name, "text"].head(5):
        print("-", t)

# Class balance of both subsets
balance = pd.DataFrame({
    "train": df_train["label_name"].value_counts(),
    "test": pd.Series(y_test).map(lambda i: labels[i]).value_counts(),
}).loc[labels]
print("\nClass balance:\n" + balance.to_string())
balance.to_csv(OUT / "class_balance.csv")

print("\n-> Now write your hypothesis (which classes will be confused?) into REPORT.md")
```

---

## 3. `step2_zero_shot.py`: approach A, zero-shot

```python
"""Step 2 - Approach A: zero-shot classification (no training at all)."""
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from transformers import pipeline
from common import load_data, save_preds, save_result, OUT

N_EXP = 200   # texts used for the label-wording experiment (lower to 100 if your laptop is slow)

train, test, labels, y_train, y_test = load_data()
zs = pipeline("zero-shot-classification", model="cross-encoder/nli-distilroberta-base")


def zero_shot_predict(texts, cand_labels, template=None, batch_size=16):
    """Return (class ids, seconds). cand_labels MUST be in order World, Sports, Business, Sci/Tech."""
    kwargs = {"candidate_labels": cand_labels, "batch_size": batch_size}
    if template:
        kwargs["hypothesis_template"] = template
    start = time.perf_counter()
    outs = zs(list(texts), **kwargs)
    elapsed = time.perf_counter() - start
    preds = np.array([cand_labels.index(o["labels"][0]) for o in outs])
    return preds, elapsed


# 1) One example, to see what the pipeline returns
example = zs(test[0]["text"], candidate_labels=["world politics", "sports", "business", "science and technology"])
print("Text:", test[0]["text"])
print("True:", labels[y_test[0]])
print("Pipeline output:", example)

# 2) Label-wording experiment (>= 3 variants) x 2 hypothesis templates
label_variants = {
    "1 dataset names": ["World", "Sports", "Business", "Sci/Tech"],
    "2 descriptive":   ["world politics", "sports", "business", "science and technology"],
    "3 specific":      ["international news and politics", "sports and athletes",
                        "business, finance and economy", "science, technology and computers"],
    "4 one-word":      ["politics", "sports", "economy", "technology"],
}
templates = {
    "default": None,                                   # "This example is {}."
    "news":    "This news article is about {}.",
}

exp_texts, exp_true = test["text"][:N_EXP], y_test[:N_EXP]
rows = []
for v_name, cand in label_variants.items():
    for t_name, tmpl in templates.items():
        preds, secs = zero_shot_predict(exp_texts, cand, template=tmpl)
        rows.append({"Label variant": v_name, "Template": t_name,
                     "Accuracy": round(accuracy_score(exp_true, preds), 3),
                     "Sec / 100": round(secs / N_EXP * 100, 2)})
        print(rows[-1])

wording_df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
print("\nLabel-wording experiment:\n" + wording_df.to_string(index=False))
wording_df.to_csv(OUT / "zero_shot_wording.csv", index=False)

# 3) Best variant on all 500 test texts
best = wording_df.iloc[0]
best_labels = label_variants[best["Label variant"]]
best_template = templates[best["Template"]]
print(f"\nBest variant: {best['Label variant']} | template: {best['Template']}")

preds_A, secs_A = zero_shot_predict(test["text"], best_labels, template=best_template)
time_A = secs_A / len(test) * 100

save_preds("A", preds_A)
save_result("A - Zero-shot", y_test, preds_A, time_A, train_time_s=0, n_labelled=0)
```

---

## 4. `step3_embeddings.py`: approach B, embeddings + logistic regression

```python
"""Step 3 - Approach B: sentence embeddings + logistic regression."""
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from common import load_data, save_preds, save_result, OUT, N_TRAIN

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # ~90 MB

train, test, labels, y_train, y_test = load_data()
enc = SentenceTransformer(EMBED_MODEL)

# 1) Encode texts (encoding the training set counts as part of training time)
start = time.perf_counter()
X_train = enc.encode(train["text"], batch_size=64, show_progress_bar=True)
encode_train_time = time.perf_counter() - start
X_test = enc.encode(test["text"], batch_size=64)
print("Embedding shapes:", X_train.shape, X_test.shape)    # (2000, 384) (500, 384)

# 2) Train logistic regression on all 2,000 examples
start = time.perf_counter()
clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
fit_time = time.perf_counter() - start

# 3) Inference time = encode + predict, measured end to end on the 500 test texts
start = time.perf_counter()
preds_B = clf.predict(enc.encode(test["text"], batch_size=64))
time_B = (time.perf_counter() - start) / len(test) * 100

save_preds("B", preds_B)
save_result("B - Embeddings + LR", y_test, preds_B, time_B,
            train_time_s=encode_train_time + fit_time, n_labelled=N_TRAIN)

# 4) Learning curve: 100 / 500 / 2,000 labelled examples
#    (train is already shuffled, so X_train[:n] is a random, roughly balanced sample)
curve = []
for n in [100, 500, 2000]:
    clf_n = LogisticRegression(max_iter=1000).fit(X_train[:n], y_train[:n])
    p = clf_n.predict(X_test)
    curve.append({"Labelled examples": n,
                  "Accuracy": round(accuracy_score(y_test, p), 3),
                  "Macro-F1": round(f1_score(y_test, p, average="macro"), 3)})
curve_df = pd.DataFrame(curve)
print("\nLearning curve:\n" + curve_df.to_string(index=False))
curve_df.to_csv(OUT / "learning_curve.csv", index=False)

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(curve_df["Labelled examples"], curve_df["Accuracy"], marker="o")
ax.set_xscale("log")
ax.set_xlabel("Labelled training examples (log)")
ax.set_ylabel("Accuracy")
ax.set_title("Approach B - learning curve")
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(OUT / "learning_curve.png", dpi=120)
print("Saved", OUT / "learning_curve.png")

# 5) Keep the trained classifier for the Gradio demo (stretch goal)
joblib.dump(clf, OUT / "news_sorter_lr.joblib")
```

---

## 5. `step4_evaluate.py`: evaluate and compare A and B

```python
"""Step 4 - Evaluate and compare approaches A and B (run steps 2 and 3 first)."""
from sklearn.metrics import classification_report
from common import load_data, load_preds, save_confusion, errors_table, OUT

train, test, labels, y_train, y_test = load_data()

approaches = {"A": "A - Zero-shot", "B": "B - Embeddings + LR"}
for key, name in approaches.items():
    preds = load_preds(key)
    if preds is None:
        print(f"Skipping {name}: outputs/preds_{key}.npy not found - run its step first.")
        continue

    print(f"\n=== {name} ===")
    print(classification_report(y_test, preds, target_names=labels, digits=3))

    save_confusion(y_test, preds, labels, name, f"confusion_{key}.png")

    errs = errors_table(test, y_test, preds, labels)
    errs.to_csv(OUT / f"errors_{key}.csv", index=False)
    print(f"{len(errs)} errors out of {len(y_test)} (all saved to outputs/errors_{key}.csv). First 8:")
    for _, r in errs.head(8).iterrows():
        print(f"  [{r.idx}] TRUE={r.true:<9} PRED={r.pred:<9} | {r.text[:160]}")

print("\n-> Pick 5 errors and explain each in one line in REPORT.md")
```

---

## 6. `step5_finetune.py` (optional): approach C, fine-tune DistilBERT

```python
"""Step 5 (optional) - Approach C: fine-tune DistilBERT for 1 epoch on CPU (~15-30 min)."""
import time
from sklearn.metrics import classification_report
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from common import load_data, save_preds, save_result, save_confusion, errors_table, OUT, SEED, N_TRAIN

MODEL_NAME = "distilbert-base-uncased"   # ~260 MB

train, test, labels, y_train, y_test = load_data()

# 1) Tokenise (max_length=64 is enough for headlines and keeps CPU training fast)
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tok(batch["text"], truncation=True, max_length=64)

train_tok = train.map(tokenize, batched=True)
test_tok = test.map(tokenize, batched=True)

# 2) Model with a new 4-class head ("Some weights ... not initialized" warning is expected)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label={i: l for i, l in enumerate(labels)},
    label2id={l: i for i, l in enumerate(labels)},
)

# 3) Train for 1 epoch
args = TrainingArguments(
    output_dir=str(OUT / "distilbert-checkpoints"),
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=25,
    save_strategy="no",      # no checkpoints on disk
    report_to="none",        # no external loggers
    seed=SEED,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    data_collator=DataCollatorWithPadding(tok),
)

start = time.perf_counter()
trainer.train()
train_time_C = time.perf_counter() - start
print(f"Training took {train_time_C / 60:.1f} min")

# 4) Predict and evaluate
start = time.perf_counter()
pred_out = trainer.predict(test_tok)
time_C = (time.perf_counter() - start) / len(test) * 100
preds_C = pred_out.predictions.argmax(axis=-1)

save_preds("C", preds_C)
save_result("C - Fine-tuned DistilBERT", y_test, preds_C, time_C,
            train_time_s=train_time_C, n_labelled=N_TRAIN)
print(classification_report(y_test, preds_C, target_names=labels, digits=3))
save_confusion(y_test, preds_C, labels, "C - Fine-tuned DistilBERT", "confusion_C.png")
errors_table(test, y_test, preds_C, labels).to_csv(OUT / "errors_C.csv", index=False)

# 5) Save the model (for reuse or push_to_hub)
trainer.save_model(str(OUT / "distilbert-agnews"))
tok.save_pretrained(str(OUT / "distilbert-agnews"))
print("Model saved to", OUT / "distilbert-agnews")
```

---

## 7. `step6_results.py`: final results table

```python
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
```

---

## 8. `run_all.py`: run everything from scratch

```python
"""Run all steps in order from a clean state.

    python run_all.py                  # steps 1-4 + 6 (fast, ~10 min)
    python run_all.py --with-finetune  # also step 5 (adds ~15-30 min)
"""
import subprocess
import sys
import time
from pathlib import Path

steps = ["step1_explore.py", "step2_zero_shot.py", "step3_embeddings.py", "step4_evaluate.py"]
if "--with-finetune" in sys.argv:
    steps.append("step5_finetune.py")
steps.append("step6_results.py")

# Fresh start: remove old results and predictions so nothing stale ends up in the table
out = Path("outputs")
for f in [out / "results.json", *out.glob("preds_*.npy")]:
    f.unlink(missing_ok=True)

t0 = time.perf_counter()
for script in steps:
    print(f"\n{'=' * 70}\n>>> python {script}\n{'=' * 70}", flush=True)
    subprocess.run([sys.executable, script], check=True)   # stops at the first error
print(f"\nAll steps finished in {(time.perf_counter() - t0) / 60:.1f} min.")
```

---

## 9. `REPORT.md`: your written answers (template with example content)

```markdown
# News Sorter – Report

## Hypothesis (Step 1)
I expect **Business ↔ Sci/Tech** to be confused most, because tech-company news (earnings,
mergers, product launches) fits both. **World ↔ Business** may also mix for trade and
economic-policy stories. **Sports** should be the easiest class.

## Label-wording experiment (Step 2) – from outputs/zero_shot_wording.csv
| Label variant | Template | Accuracy |
|---|---|---|
| … | … | … |

## Learning curve (Step 3) – from outputs/learning_curve.csv
| Labelled examples | Accuracy |
|---|---|
| 100 | … |
| 500 | … |
| 2000 | … |

## Results (Step 6) – from outputs/results_table.csv
| Approach | Accuracy | Macro-F1 | Sec / 100 texts | Training time (s) | Labelled examples |
|---|---|---|---|---|---|
| … | … | … | … | … | … |

Confusion matrices: outputs/confusion_A.png, confusion_B.png (confusion_C.png)

## 5 analysed errors (Step 4) – from outputs/errors_*.csv
| # | Idx | True → Pred | Why it went wrong |
|---|---|---|---|
| 1 | … | Business → Sci/Tech | Tech-company earnings story: both labels plausible. |
| 2 | … | Sci/Tech → Business | Software acquisition dominated by finance vocabulary. |
| 3 | … | World → Business | Oil prices / sanctions: economic words in a geopolitical story. |
| 4 | … | Sports → World | Olympic doping scandal reads like a political/legal story. |
| 5 | … | World → Sci/Tech | Space-agency or climate story where science words dominate. |

## Recommendation (≤ 150 words)
**Deploy approach B (sentence embeddings + logistic regression).** On our test set it reached
~87 % accuracy, far above zero-shot and within a point or two of fine-tuned DistilBERT, while
being the fastest at inference and training in seconds instead of half an hour on a CPU. The
learning curve shows that a few hundred labelled examples already get close to the
2,000-example result, so labelling cost is low. Adding or changing a topic only means labelling
some new examples and retraining in under a minute, with no GPU. Zero-shot is useful as a
bootstrap when no labels exist, but it is too sensitive to label wording for production
routing. Fine-tuning is worth revisiting only if we need the last few percent and have more data.
```

**Typical outcome (yours will differ):**

| Approach | Accuracy | Sec / 100 texts | Training time | Labelled examples |
|---|---|---|---|---|
| A – Zero-shot | ~0.55–0.75 | slowest (≈ 5–20 s) | 0 | 0 |
| B – Embeddings + LR | ~0.85–0.88 | fastest (≈ 0.5–2 s) | ≈ 10–60 s | 2,000 |
| C – Fine-tuned DistilBERT | ~0.86–0.90 | ≈ 1–4 s | ≈ 15–30 min | 2,000 |

---

## 10. Final check

```bash
python run_all.py                  # or: python run_all.py --with-finetune
```

It must finish with `All steps finished in … min.` and no traceback.

| Deliverable | Where |
|---|---|
| Runs end to end on CPU | `python run_all.py` |
| Results table (accuracy, macro-F1, s/100, training time, labelled examples) | `outputs/results_table.csv` → REPORT.md |
| Confusion matrix per approach | `outputs/confusion_A.png`, `confusion_B.png`, `confusion_C.png` |
| Label-wording experiment ≥ 3 variants | `outputs/zero_shot_wording.csv` → REPORT.md |
| Learning curve 100 / 500 / 2,000 | `outputs/learning_curve.csv` + `.png` → REPORT.md |
| 5 analysed errors | `outputs/errors_*.csv` → REPORT.md |
| Recommendation ≤ 150 words | REPORT.md |

**Hand in:** the project folder without `hf-env/` (zip it), including `REPORT.md` and `outputs/`.

---

## 11. Stretch goals: extra scripts

### 11.1 `app.py`: Gradio demo of approach B (run step 3 first)

```python
"""Gradio web demo: python app.py, then open the printed URL (http://127.0.0.1:7860)."""
import joblib
import gradio as gr
from sentence_transformers import SentenceTransformer

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
clf = joblib.load("outputs/news_sorter_lr.joblib")

def classify(text):
    probs = clf.predict_proba(enc.encode([text]))[0]
    return {LABELS[i]: float(p) for i, p in enumerate(probs)}

gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=3, label="News headline"),
    outputs=gr.Label(num_top_classes=4),
    title="News Sorter",
    examples=["Apple unveils new iPhone with faster chip",
              "Manchester United win the derby 3-1"],
).launch()
```

**Hugging Face Spaces:** create a Gradio Space, upload `app.py`, `outputs/news_sorter_lr.joblib` and a `requirements.txt` containing `sentence-transformers`, `scikit-learn`, `joblib`, `gradio`.

### 11.2 `push_model.py`: push the fine-tuned model to the Hub (run step 5 first)

First log in once in the terminal with a *write* token from huggingface.co/settings/tokens:

```bash
hf auth login            # older huggingface_hub versions: huggingface-cli login
```

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

REPO = "your-username/distilbert-agnews-mini"
AutoModelForSequenceClassification.from_pretrained("outputs/distilbert-agnews").push_to_hub(REPO)
AutoTokenizer.from_pretrained("outputs/distilbert-agnews").push_to_hub(REPO)
print("Pushed to https://huggingface.co/" + REPO)
```

### 11.3 `own_data.py`: your own domain data (50–100 labelled texts)

```python
"""my_texts.csv must have two columns: text,label  (label = category name, e.g. 'billing')."""
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

own = load_dataset("csv", data_files="my_texts.csv")["train"].train_test_split(test_size=0.3, seed=42)
own_labels = sorted(set(own["train"]["label"]))
to_id = {l: i for i, l in enumerate(own_labels)}

enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
Xo_tr, yo_tr = enc.encode(own["train"]["text"]), [to_id[l] for l in own["train"]["label"]]
Xo_te, yo_te = enc.encode(own["test"]["text"]), [to_id[l] for l in own["test"]["label"]]

clf = LogisticRegression(max_iter=1000).fit(Xo_tr, yo_tr)
preds = clf.predict(Xo_te)
print("Own-data accuracy:", accuracy_score(yo_te, preds))
print(classification_report(yo_te, preds, labels=list(range(len(own_labels))),
                            target_names=own_labels, zero_division=0))
```

### 11.4 `multilingual.py`: Czech / German headlines

```python
"""Train on English AG News, classify Czech/German headlines with no extra labels."""
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from common import load_data

train, test, labels, y_train, y_test = load_data()
enc_ml = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
clf_ml = LogisticRegression(max_iter=1000).fit(enc_ml.encode(train["text"], batch_size=64), y_train)

samples = [
    "Vláda schválila nový státní rozpočet na příští rok",     # CZ
    "Sparta porazila Slavii 2:1 v derby",                      # CZ
    "Siemens meldet Rekordgewinn im dritten Quartal",          # DE
    "Forscher entwickeln neuen Quantencomputer-Chip",          # DE
]
for s, p in zip(samples, clf_ml.predict(enc_ml.encode(samples))):
    print(f"{labels[p]:<9} | {s}")
```

---

## 12. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'common'` | Run the script from the project folder (`cd hf-news-sorter`) |
| `ModuleNotFoundError: No module named 'transformers'` | Activate `hf-env` first; the prompt must start with `(hf-env)` |
| First run is slow | Models download once to `~/.cache/huggingface`; later runs are instant |
| Out of memory | Lower `batch_size` to 8, or shorten texts with `max_length=64` |
| Zero-shot takes too long | Set `N_EXP = 100` in `step2_zero_shot.py`, or remove the `"news"` template |
| Corporate proxy blocks downloads | Set `HTTPS_PROXY` in the terminal before running, or use Colab/Kaggle (upload the `.py` files and run `!python step1_explore.py`) |
| `UnicodeEncodeError` when printing on Windows | Run `set PYTHONUTF8=1` (cmd) or `$env:PYTHONUTF8=1` (PowerShell) before the script |
| "Some weights not initialized" | Expected: the classification head is new |
| `ImportError: ... requires accelerate>=...` | `pip install accelerate` |
| `step4`/`step6` says results not found | Run steps 2 and 3 first (or simply `python run_all.py`) |
| Accuracy ~0.25 for zero-shot | Candidate labels are not in the order World, Sports, Business, Sci/Tech |

**Useful reading:** [Hugging Face LLM Course](https://huggingface.co/learn/llm-course), chapters 1–3.
