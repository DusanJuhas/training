# Level 4 — Your First Hugging Face NLP Pipeline: Reference Solution

> This is a complete, working solution. Each numbered section below is **one notebook cell** (or a small group of cells): copy them in order and run them top to bottom.
> Don't understand a cell? → Look up the matching section in **Level 3**, which explains every line.
> Want to know *why* a step exists? → **Level 2** describes the goal of each step.

**Checked with:** Python 3.13, `torch` 2.15, `transformers` 5.22, `accelerate`, `datasets` 5.1, `sentence-transformers` 6.2, `scikit-learn` 1.x. Any Python 3.10+ with recent versions should work.
**Total runtime on a 4-core laptop CPU:** about 10 min without Step 5, about 25–40 min with Step 5 (the first run is longer because models download).

---

## 0. Preparation: environment and libraries

Run in a **terminal**, not in the notebook:

```bash
# 1) Create and activate a virtual environment
python -m venv hf-env
# Windows:      hf-env\Scripts\activate
# macOS/Linux:  source hf-env/bin/activate

# 2) CPU build of PyTorch (smaller and faster to install than the CUDA build)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3) Everything else
pip install transformers accelerate datasets evaluate sentence-transformers scikit-learn pandas matplotlib jupyter
#    (accelerate is required by the Trainer API in Step 5)

# 4) Optional, for the stretch goal (Gradio demo)
pip install gradio

# 5) Start Jupyter
jupyter notebook
```

**Google Colab / Kaggle:** skip the steps above and run this in the first cell instead:

```python
!pip install -q sentence-transformers datasets evaluate accelerate
```

---

## 1. Smoke test and imports

```python
# Cell 1 — smoke test: confirms that transformers + torch work
from transformers import pipeline
print(pipeline("sentiment-analysis")("Hugging Face runs fine on my laptop!"))
# Expected: [{'label': 'POSITIVE', 'score': 0.99...}]
```

```python
# Cell 2 — all imports and global settings in one place
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

SEED = 42
N_TRAIN, N_TEST = 2000, 500
pd.set_option("display.max_colwidth", 200)

results = []   # one dict per approach, filled as we go -> final results table
```

---

## 2. Step 1 — Load and explore the data

```python
# Cell 3 — load AG News and create reproducible subsets
ds = load_dataset("fancyzhx/ag_news")

train = ds["train"].shuffle(seed=SEED).select(range(N_TRAIN))
test  = ds["test"].shuffle(seed=SEED).select(range(N_TEST))
labels = train.features["label"].names          # ['World', 'Sports', 'Business', 'Sci/Tech']

y_train = np.array(train["label"])
y_test  = np.array(test["label"])

print(ds)
print("Labels:", labels)
print("Subset sizes:", len(train), len(test))
```

```python
# Cell 4 — 5 examples per class
df_train = train.to_pandas()
df_train["label_name"] = df_train["label"].map(lambda i: labels[i])

for name in labels:
    print(f"\n===== {name} =====")
    for t in df_train.loc[df_train["label_name"] == name, "text"].head(5):
        print("-", t)
```

```python
# Cell 5 — class balance of both subsets
balance = pd.DataFrame({
    "train": df_train["label_name"].value_counts(),
    "test":  pd.Series(y_test).map(lambda i: labels[i]).value_counts(),
}).loc[labels]
balance
```

**Markdown cell — hypothesis (example):**

> I expect **Business ↔ Sci/Tech** to be confused most, because tech-company news (earnings, mergers, product launches) fits both. **World ↔ Business** may also be mixed up for economic-policy and trade stories. **Sports** should be the easiest class because of its very distinctive vocabulary.

---

## 3. Helper functions (used by every approach)

```python
# Cell 6 — shared helpers
def evaluate_approach(name, preds, sec_per_100, train_time_s, n_labelled):
    """Print metrics and store one row in the global results table."""
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    print(f"{name}: accuracy={acc:.3f}  macro-F1={f1:.3f}  "
          f"{sec_per_100:.2f} s/100 texts  train={train_time_s:.1f} s")
    results[:] = [r for r in results if r["Approach"] != name]   # avoid duplicates on re-run
    results.append({
        "Approach": name,
        "Accuracy": round(acc, 3),
        "Macro-F1": round(f1, 3),
        "Sec / 100 texts": round(sec_per_100, 2),
        "Training time (s)": round(train_time_s, 1),
        "Labelled examples": n_labelled,
    })


def plot_confusion(preds, title, ax=None):
    ConfusionMatrixDisplay.from_predictions(
        y_test, preds, display_labels=labels, ax=ax, colorbar=False, cmap="Blues"
    )
    (ax or plt.gca()).set_title(title)


def show_errors(preds, n=10):
    """Print the first n misclassified test examples."""
    wrong = np.where(np.asarray(preds) != y_test)[0]
    print(f"{len(wrong)} errors out of {len(y_test)}\n")
    for i in wrong[:n]:
        print(f"[{i}] TRUE={labels[y_test[i]]:<9} PRED={labels[preds[i]]:<9} | {test[int(i)]['text'][:180]}")
```

---

## 4. Step 2 — Approach A: zero-shot classification

```python
# Cell 7 — create the pipeline and try one text
zs = pipeline("zero-shot-classification", model="cross-encoder/nli-distilroberta-base")

# IMPORTANT: same order as the dataset labels (World, Sports, Business, Sci/Tech)
candidate_labels = ["world politics", "sports", "business", "science and technology"]

print(test[0]["text"])
print("True:", labels[y_test[0]])
print(zs(test[0]["text"], candidate_labels=candidate_labels))
```

```python
# Cell 8 — reusable zero-shot runner
def zero_shot_predict(texts, cand_labels, template=None, batch_size=16):
    kwargs = {"candidate_labels": cand_labels, "batch_size": batch_size}
    if template:
        kwargs["hypothesis_template"] = template
    start = time.perf_counter()
    outs = zs(list(texts), **kwargs)
    elapsed = time.perf_counter() - start
    preds = np.array([cand_labels.index(o["labels"][0]) for o in outs])
    return preds, elapsed
```

```python
# Cell 9 — label-wording experiment (≥ 3 variants) on the first 200 test texts
N_EXP = 200   # lower to 100 if your laptop is slow
exp_texts, exp_true = test["text"][:N_EXP], y_test[:N_EXP]

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

rows = []
for v_name, cand in label_variants.items():
    for t_name, tmpl in templates.items():
        preds, secs = zero_shot_predict(exp_texts, cand, template=tmpl)
        rows.append({"Label variant": v_name, "Template": t_name,
                     "Accuracy": round(accuracy_score(exp_true, preds), 3),
                     "Sec / 100": round(secs / N_EXP * 100, 2)})
        print(rows[-1])

wording_df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
wording_df
```

```python
# Cell 10 — run the best variant on all 500 test texts
best = wording_df.iloc[0]
best_labels = label_variants[best["Label variant"]]
best_template = templates[best["Template"]]
print("Best variant:", best["Label variant"], "| template:", best["Template"])

preds_A, secs_A = zero_shot_predict(test["text"], best_labels, template=best_template)
time_A = secs_A / len(test) * 100

evaluate_approach("A – Zero-shot", preds_A, time_A, train_time_s=0, n_labelled=0)
```

---

## 5. Step 3 — Approach B: embeddings + logistic regression

```python
# Cell 11 — encode all texts (training-set encoding counts as part of training time)
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")   # ~90 MB

start = time.perf_counter()
X_train = enc.encode(train["text"], batch_size=64, show_progress_bar=True)
encode_train_time = time.perf_counter() - start

X_test = enc.encode(test["text"], batch_size=64)
print(X_train.shape, X_test.shape)    # (2000, 384) (500, 384)
```

```python
# Cell 12 — train logistic regression on 2,000 examples
start = time.perf_counter()
clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
fit_time = time.perf_counter() - start

# Inference time = encode + predict, measured end to end on the 500 test texts
start = time.perf_counter()
preds_B = clf.predict(enc.encode(test["text"], batch_size=64))
time_B = (time.perf_counter() - start) / len(test) * 100

evaluate_approach("B – Embeddings + LR", preds_B, time_B,
                  train_time_s=encode_train_time + fit_time, n_labelled=N_TRAIN)
```

```python
# Cell 13 — learning curve: 100 / 500 / 2,000 labelled examples
curve = []
for n in [100, 500, 2000]:
    clf_n = LogisticRegression(max_iter=1000).fit(X_train[:n], y_train[:n])
    p = clf_n.predict(X_test)
    curve.append({"Labelled examples": n,
                  "Accuracy": round(accuracy_score(y_test, p), 3),
                  "Macro-F1": round(f1_score(y_test, p, average="macro"), 3)})

curve_df = pd.DataFrame(curve)
display(curve_df)

plt.figure(figsize=(5, 3))
plt.plot(curve_df["Labelled examples"], curve_df["Accuracy"], marker="o")
plt.xscale("log"); plt.xlabel("Labelled training examples (log)"); plt.ylabel("Accuracy")
plt.title("Approach B – learning curve"); plt.grid(alpha=.3); plt.show()
```

> Because `train` is already shuffled, `X_train[:n]` is a random sample with roughly balanced classes. At `n=100` each class still has ~25 examples.

---

## 6. Step 4 — Evaluate and compare

```python
# Cell 14 — metrics per class
for name, p in [("A – Zero-shot", preds_A), ("B – Embeddings + LR", preds_B)]:
    print(f"\n=== {name} ===")
    print(classification_report(y_test, p, target_names=labels, digits=3))
```

```python
# Cell 15 — confusion matrices side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_confusion(preds_A, "A – Zero-shot", ax=axes[0])
plot_confusion(preds_B, "B – Embeddings + LR", ax=axes[1])
plt.tight_layout(); plt.show()
```

```python
# Cell 16 — inspect misclassified examples
print("### Approach A errors"); show_errors(preds_A, n=8)
print("\n### Approach B errors"); show_errors(preds_B, n=8)
```

**Markdown cell — 5 analysed errors (template; fill in with your actual indices):**

| # | Idx | True → Pred | Why it went wrong (one line) |
|---|---|---|---|
| 1 | … | Business → Sci/Tech | Tech-company earnings story: both labels are plausible. |
| 2 | … | Sci/Tech → Business | Text is about a software company's acquisition, dominated by finance words. |
| 3 | … | World → Business | Oil prices / trade sanctions: economic vocabulary in a geopolitical story. |
| 4 | … | Sports → World | Olympic doping scandal reads like a political/legal story. |
| 5 | … | World → Sci/Tech | Space-agency or climate story where the "science" words dominate. |

---

## 7. Step 5 (optional) — Approach C: fine-tune DistilBERT

```python
# Cell 17 — tokenise
model_name = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tok(batch["text"], truncation=True, max_length=64)

train_tok = train.map(tokenize, batched=True)
test_tok  = test.map(tokenize, batched=True)
```

```python
# Cell 18 — model with a new 4-class head ("some weights not initialized" warning is expected)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label={i: l for i, l in enumerate(labels)},
    label2id={l: i for i, l in enumerate(labels)},
)
```

```python
# Cell 19 — train for 1 epoch on CPU (~15–30 min)
args = TrainingArguments(
    output_dir="distilbert-agnews",
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
print(f"Training took {train_time_C/60:.1f} min")
```

```python
# Cell 20 — predict and evaluate
start = time.perf_counter()
pred_out = trainer.predict(test_tok)
time_C = (time.perf_counter() - start) / len(test) * 100

preds_C = pred_out.predictions.argmax(axis=-1)
evaluate_approach("C – Fine-tuned DistilBERT", preds_C, time_C,
                  train_time_s=train_time_C, n_labelled=N_TRAIN)

print(classification_report(y_test, preds_C, target_names=labels, digits=3))
plot_confusion(preds_C, "C – Fine-tuned DistilBERT"); plt.show()
```

> **Skipping Step 5?** Just don't run cells 17–20. The results table then has two rows.

---

## 8. Step 6 — Results table and recommendation

```python
# Cell 21 — final comparison table
results_df = pd.DataFrame(results).set_index("Approach")
results_df
```

```python
# Cell 22 — optional: visual comparison
ax = results_df[["Accuracy", "Macro-F1"]].plot.bar(figsize=(7, 3.5), rot=0, ylim=(0, 1))
ax.set_title("Approach comparison"); ax.grid(axis="y", alpha=.3); plt.show()
```

**Typical outcome (yours will differ):**

| Approach | Accuracy | Macro-F1 | Sec / 100 texts | Training time | Labelled examples |
|---|---|---|---|---|---|
| A – Zero-shot | ~0.55–0.75 | similar | slowest (≈ 5–20 s) | 0 | 0 |
| B – Embeddings + LR | ~0.85–0.88 | similar | fastest (≈ 0.5–2 s) | ≈ 10–60 s | 2,000 |
| C – Fine-tuned DistilBERT | ~0.86–0.90 | similar | ≈ 1–4 s | ≈ 15–30 min | 2,000 |

**Markdown cell — recommendation (example, ≤ 150 words):**

> **Deploy approach B (sentence embeddings + logistic regression).** On our test set it reached ~87 % accuracy, far above zero-shot and within a point or two of fine-tuned DistilBERT, while being the fastest at inference and training in seconds rather than half an hour on a CPU. The learning curve shows that a few hundred labelled examples already get close to the 2,000-example result, so the labelling cost is low. Adding or changing a topic only means labelling some new examples and retraining in under a minute, with no GPU required. Zero-shot is useful as a bootstrap when no labels exist at all, but it is too sensitive to label wording for production routing. Fine-tuning would be worth revisiting only if we need the last few percent of accuracy and have more labelled data.

---

## 9. Final check

1. **Kernel → Restart & Run All**: every cell must finish without errors.
2. Acceptance checklist:

| Deliverable | Where in this notebook |
|---|---|
| Runs end to end on CPU | Restart & Run All |
| Results table (accuracy, macro-F1, s/100, training time, labelled examples) | Cell 21 |
| Confusion matrix per approach | Cells 15 and 20 |
| Label-wording experiment ≥ 3 variants | Cell 9 (`wording_df`) |
| Learning curve 100 / 500 / 2,000 | Cell 13 (`curve_df`) |
| 5 analysed errors | Cell 16 + Markdown table |
| Recommendation ≤ 150 words | Markdown cell, section 8 |

---

## 10. Stretch goals: code snippets

### 10.1 Gradio demo of the best model (approach B)

```python
import gradio as gr

def classify(text):
    probs = clf.predict_proba(enc.encode([text]))[0]
    return {labels[i]: float(p) for i, p in enumerate(probs)}

gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=3, label="News headline"),
    outputs=gr.Label(num_top_classes=4),
    title="News Sorter",
    examples=["Apple unveils new iPhone with faster chip",
              "Manchester United win the derby 3-1"],
).launch()
```

### 10.2 Save approach B for reuse / Hugging Face Spaces

```python
import joblib
joblib.dump(clf, "news_sorter_lr.joblib")
# In a Space's app.py: load the same SentenceTransformer + joblib.load("news_sorter_lr.joblib"),
# put the Gradio code above in app.py, and list
# sentence-transformers, scikit-learn, joblib, gradio in requirements.txt.
```

### 10.3 Push the fine-tuned model to the Hub

```python
from huggingface_hub import notebook_login
notebook_login()                                   # paste a write token from huggingface.co/settings/tokens

trainer.model.push_to_hub("your-username/distilbert-agnews-mini")
tok.push_to_hub("your-username/distilbert-agnews-mini")
```

### 10.4 Your own domain data (50–100 labelled texts)

```python
# my_texts.csv with two columns: text,label  (label = category name as a string)
own = load_dataset("csv", data_files="my_texts.csv")["train"].train_test_split(test_size=0.3, seed=SEED)
own_labels = sorted(set(own["train"]["label"]))
to_id = {l: i for i, l in enumerate(own_labels)}

Xo_tr = enc.encode(own["train"]["text"]); yo_tr = [to_id[l] for l in own["train"]["label"]]
Xo_te = enc.encode(own["test"]["text"]);  yo_te = [to_id[l] for l in own["test"]["label"]]

clf_own = LogisticRegression(max_iter=1000).fit(Xo_tr, yo_tr)
print("Own-data accuracy:", accuracy_score(yo_te, clf_own.predict(Xo_te)))
```

### 10.5 Multilingual embeddings (Czech / German)

```python
enc_ml = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Train on English AG News, then classify non-English headlines without any extra labels
clf_ml = LogisticRegression(max_iter=1000).fit(enc_ml.encode(train["text"], batch_size=64), y_train)

samples = [
    "Vláda schválila nový státní rozpočet na příští rok",     # CZ – World/Business
    "Sparta porazila Slavii 2:1 v derby",                      # CZ – Sports
    "Siemens meldet Rekordgewinn im dritten Quartal",          # DE – Business
    "Forscher entwickeln neuen Quantencomputer-Chip",          # DE – Sci/Tech
]
for s, p in zip(samples, clf_ml.predict(enc_ml.encode(samples))):
    print(f"{labels[p]:<9} | {s}")
```

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| First run is slow | Models download once to `~/.cache/huggingface`; later runs are instant |
| Out of memory | Lower `batch_size` to 8, or shorten texts with `max_length=64` |
| Zero-shot takes too long | Set `N_EXP = 100` in Cell 9, and check fewer templates |
| Corporate proxy blocks downloads | `import os; os.environ["HTTPS_PROXY"] = "http://proxy:port"` before loading anything, or use Colab/Kaggle |
| "Some weights not initialized" | Expected: the classification head is new |
| `ImportError: ... requires accelerate>=...` | `pip install accelerate`, then restart the kernel |
| `display` is not defined | You're running as a `.py` script, not in Jupyter: replace `display(x)` with `print(x)` |
| `evaluation_strategy` error in older tutorials | Newer `transformers` renamed it to `eval_strategy`; this solution does not need it |
| Accuracy ~0.25 for zero-shot | Candidate labels are not in the order World, Sports, Business, Sci/Tech |

**Useful reading:** [Hugging Face LLM Course](https://huggingface.co/learn/llm-course), chapters 1–3.
