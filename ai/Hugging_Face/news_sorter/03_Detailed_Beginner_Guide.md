# Level 3 — Your First Hugging Face NLP Pipeline: Detailed Beginner Guide

> This level explains **how** to do every step, assuming you have little or no experience with Python, Jupyter or machine learning.
> Each step ends with a **✅ Checkpoint**, so you know whether you can move on.
> Still stuck after reading a step? → The matching section in **Level 4** has complete, working code.

---

## Contents

0. [Key terms explained](#0-key-terms-explained)
1. [Set up your computer](#1-set-up-your-computer)
2. [Create your notebook](#2-create-your-notebook)
3. [Step 1: Load and explore the data](#3-step-1--load-and-explore-the-data)
4. [Step 2: Approach A, zero-shot classification](#4-step-2--approach-a-zero-shot-classification)
5. [Step 3: Approach B, embeddings + classifier](#5-step-3--approach-b-embeddings--a-simple-classifier)
6. [Step 4: Evaluate and compare](#6-step-4--evaluate-and-compare)
7. [Step 5 (optional): Approach C, fine-tuning](#7-step-5-optional--approach-c-fine-tune-a-small-transformer)
8. [Step 6: Results table and recommendation](#8-step-6--results-table-and-recommendation)
9. [Final check and hand-in](#9-final-check-and-hand-in)
10. [Troubleshooting](#10-troubleshooting)

---

## 0. Key terms explained

Read this once. You will meet every term below during the task.

| Term | Plain-language meaning |
|---|---|
| **NLP** | Natural Language Processing: making computers work with human text. |
| **Text classification** | Putting a text into one of several categories (here: World, Sports, Business, Sci/Tech). |
| **Hugging Face Hub** | A public website (huggingface.co) that hosts thousands of free models and datasets. The Python libraries download from it automatically. |
| **Dataset** | A collection of examples. Each example here has a `text` (headline + short description) and a `label` (a number 0–3). |
| **Label / class** | The correct category. In AG News: `0 = World`, `1 = Sports`, `2 = Business`, `3 = Sci/Tech`. |
| **Training set / test set** | Training examples are used to *teach* a model. Test examples are kept aside and only used to *measure* how good the model is. Never train on the test set. |
| **Model** | A program with learned parameters that turns input (text) into output (a label). |
| **Pre-trained model** | A model someone else already trained on huge amounts of text. You reuse it. |
| **Pipeline** | A Hugging Face shortcut that combines everything needed to use a model in one line of code. |
| **Zero-shot** | Classifying without showing the model *any* labelled example of your task. |
| **Embedding** | A list of numbers (a vector) that represents the meaning of a text. Similar texts get similar vectors. |
| **Logistic regression** | A simple, fast classic classifier. Here it learns which embedding patterns belong to which class. |
| **Fine-tuning** | Continuing to train a pre-trained model on your own labelled data so it specialises in your task. |
| **Accuracy** | Share of test examples classified correctly. 0.85 = 85 %. |
| **Macro-F1** | A score that combines precision and recall for each class, then averages over classes equally. It shows whether *every* class works well, not just the large ones. |
| **Confusion matrix** | A table showing, for each true class, how often the model predicted each class. The diagonal holds the correct answers; everything off the diagonal is a mistake. |
| **Inference** | Using a trained model to make predictions. |
| **Batch size** | How many texts the model processes at once. Bigger is faster but needs more memory. |
| **Epoch** | One full pass over the training data during training. |
| **Seed** | A fixed number for the random generator, so "random" shuffles are the same every time you run the notebook (reproducibility). |

---

## 1. Set up your computer

### 1.1 Check your hardware

You need:

| Requirement | Minimum |
|---|---|
| RAM | 8 GB |
| Free disk | 3 GB |
| CPU | Any 4-core laptop CPU from the last ~6 years |
| Python | 3.10 or newer |
| Internet | Needed for the first download only |

**No GPU is needed.** All models in this task are smaller than 350 MB.

### 1.2 Check that Python is installed

Open a terminal:
- **Windows:** press `Win`, type `cmd` or `PowerShell`, press Enter.
- **macOS:** press `Cmd + Space`, type `Terminal`, press Enter.
- **Linux:** open your terminal app.

Type:

```bash
python --version
```

You should see something like `Python 3.11.6`. If it says `3.9` or lower, or "command not found":
- On macOS/Linux, try `python3 --version` and use `python3` instead of `python` everywhere below.
- Otherwise install Python from [python.org](https://www.python.org/downloads/). On Windows, **tick "Add Python to PATH"** during installation.

> **Cannot install anything on your work laptop?** Skip to [1.6 Cloud alternative](#16-cloud-alternative-colab-or-kaggle).

### 1.3 Create a project folder and a virtual environment

A *virtual environment* is a private box of Python libraries for this project only, so nothing you install here interferes with other projects.

```bash
mkdir hf-news-sorter
cd hf-news-sorter
python -m venv hf-env
```

This creates a folder `hf-env` containing an isolated Python.

### 1.4 Activate the virtual environment

You must do this **every time** you open a new terminal for this project.

- **Windows (cmd / PowerShell):**
  ```bash
  hf-env\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  source hf-env/bin/activate
  ```

Your prompt now starts with `(hf-env)`. That is how you know it is active.

> **PowerShell error "running scripts is disabled"?** Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, confirm with `Y`, and try again.

### 1.5 Install the libraries

First install PyTorch, the engine that runs the models. Use the **CPU version**, which is smaller and installs faster than the GPU (CUDA) version:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install everything else:

```bash
pip install transformers accelerate datasets evaluate sentence-transformers scikit-learn pandas matplotlib jupyter
```

What each library does:

| Library | Purpose |
|---|---|
| `torch` | Runs neural networks |
| `transformers` | Hugging Face models and the `pipeline` / `Trainer` API |
| `accelerate` | Required by the `Trainer` API (Step 5); not in the original task's install line |
| `datasets` | Downloads and handles datasets from the Hub |
| `evaluate` | Hugging Face metrics library (optional, we mainly use scikit-learn) |
| `sentence-transformers` | Turns sentences into embeddings |
| `scikit-learn` | Classic machine learning: logistic regression, metrics, confusion matrix |
| `pandas` | Tables (DataFrames) |
| `matplotlib` | Charts |
| `jupyter` | The notebook environment |

Installation takes a few minutes. Warnings in yellow are usually fine. Red **ERROR** lines are not; see [Troubleshooting](#10-troubleshooting).

### 1.6 Cloud alternative: Colab or Kaggle

If you cannot install Python locally:
1. Go to [Google Colab](https://colab.research.google.com) (Google account) or [Kaggle Code](https://www.kaggle.com/code) (Kaggle account).
2. Create a new notebook.
3. In the first cell, run:
   ```python
   !pip install -q sentence-transformers datasets evaluate accelerate
   ```
   (`torch`, `transformers`, `scikit-learn`, `pandas` and `matplotlib` are usually pre-installed.)
4. Continue from [Section 3](#3-step-1--load-and-explore-the-data). The code is identical.

✅ **Checkpoint 1:** `pip list` (in the activated environment) shows `torch`, `transformers`, `datasets` and `sentence-transformers`.

---

## 2. Create your notebook

### 2.1 Start Jupyter

In the activated terminal, inside your project folder:

```bash
jupyter notebook
```

Your browser opens. Click **New → Python 3 (ipykernel)**. Rename the notebook (click "Untitled" at the top) to `news_sorter.ipynb`.

### 2.2 How a notebook works (30-second intro)

- A notebook is made of **cells**. Type code into a cell and press **Shift + Enter** to run it.
- Output appears directly below the cell.
- Variables you create in one cell are available in all later cells.
- `[*]` next to a cell means it is still running. Wait until it shows a number.
- **Kernel → Restart & Run All** reruns everything from scratch. You will use this for the final check.
- You can add a **Markdown cell** (menu: Cell → Cell Type → Markdown) to write notes and answers. Use these for your hypotheses, error explanations and recommendation.

### 2.3 Smoke test

In the first cell, type and run:

```python
from transformers import pipeline
print(pipeline("sentiment-analysis")("Hugging Face runs fine on my laptop!"))
```

The first run downloads a small model (you see progress bars). Expected output, roughly:

```
[{'label': 'POSITIVE', 'score': 0.99...}]
```

A warning that "no model was supplied, defaulted to …" is normal.

✅ **Checkpoint 2:** you see a `POSITIVE` label with a high score.

---

## 3. Step 1 — Load and explore the data

**Time:** ~30 min. **Goal:** get familiar with the data before modelling it.

### 3.1 Load the dataset

New cell:

```python
from datasets import load_dataset

ds = load_dataset("fancyzhx/ag_news")
print(ds)
```

You see two **splits**: `train` (120,000 rows) and `test` (7,600 rows), each with the columns `text` and `label`.

### 3.2 Take small, reproducible subsets

Working with all 120,000 examples on a CPU would take too long. We shuffle (so all classes are mixed) and take the first N:

```python
train = ds["train"].shuffle(seed=42).select(range(2000))
test  = ds["test"].shuffle(seed=42).select(range(500))
labels = train.features["label"].names
print(labels)        # ['World', 'Sports', 'Business', 'Sci/Tech']
print(len(train), len(test))
```

- `shuffle(seed=42)` gives the same random order every time. Everyone in the club using `42` gets the same subsets, so results are comparable.
- `select(range(2000))` keeps rows 0–1999.
- `labels[0]` is `'World'`, `labels[3]` is `'Sci/Tech'`. Use `labels[number]` to turn a label number into its name.

### 3.3 Look at one example

```python
print(train[0])
print("Class name:", labels[train[0]["label"]])
```

Each example is a Python dictionary: `{'text': '...', 'label': 2}`.

### 3.4 Print 5 examples per class

Convert the data to a pandas table, which is easier to filter:

```python
import pandas as pd
pd.set_option("display.max_colwidth", 200)   # show long texts

df_train = train.to_pandas()
df_train["label_name"] = df_train["label"].map(lambda i: labels[i])

for name in labels:
    print(f"\n===== {name} =====")
    for t in df_train[df_train["label_name"] == name]["text"].head(5):
        print("-", t)
```

**Read the texts.** Notice their style: a headline followed by a short description, often with an agency name like "(Reuters)" or "AP".

### 3.5 Check the class balance

"Balanced" means every class has roughly the same number of examples. If one class dominated, accuracy alone could mislead.

```python
print(df_train["label_name"].value_counts())
print(test.to_pandas()["label"].map(lambda i: labels[i]).value_counts())
```

Expect roughly 500 per class in train and 125 per class in test (not exactly, because we sampled randomly).

### 3.6 Write your hypothesis

Add a **Markdown cell** and answer: *Which classes do you expect the models to confuse, and why?*

Hint: think about a headline like *"Apple shares jump after new iPhone launch"*. Is it Business or Sci/Tech? Or *"Olympic committee faces corruption probe"*: Sports or World?

✅ **Checkpoint 3:** you have `train` (2,000), `test` (500) and `labels` (4 names), a class-count table, and a written hypothesis.

---

## 4. Step 2 — Approach A: zero-shot classification

**Time:** ~45 min. **Goal:** see how far you get with **no training at all**.

### 4.1 How zero-shot works

The model `cross-encoder/nli-distilroberta-base` was trained on **Natural Language Inference**: given two sentences, does the second follow from the first? The pipeline turns each candidate label into a sentence ("This example is about sports.") and asks the model how strongly the news text *implies* it. The label with the highest score wins.

So the **wording of your labels matters**. The model only understands them as English words.

### 4.2 Create the pipeline and try one text

```python
from transformers import pipeline

zs = pipeline("zero-shot-classification", model="cross-encoder/nli-distilroberta-base")

candidate_labels = ["world politics", "sports", "business", "science and technology"]

print(test[0]["text"])
print("True label:", labels[test[0]["label"]])
result = zs(test[0]["text"], candidate_labels=candidate_labels)
print(result)
```

The output is a dictionary:
- `result["labels"]`: the candidate labels **sorted from best to worst**
- `result["scores"]`: their scores in the same order
- so `result["labels"][0]` is the prediction.

**Important:** keep `candidate_labels` in the **same order as the dataset labels** (World, Sports, Business, Sci/Tech). Then the position of the predicted label in your list equals the class number: `candidate_labels.index(result["labels"][0])` gives 0–3.

### 4.3 Run on a small batch first (100 texts)

Zero-shot is the slowest approach, because every text is checked against every label. Start small:

```python
import time

texts_100 = test["text"][:100]

start = time.perf_counter()
outputs = zs(texts_100, candidate_labels=candidate_labels, batch_size=16)
elapsed = time.perf_counter() - start
print(f"100 texts took {elapsed:.1f} s")
```

- Passing a **list** of texts (instead of a loop) lets the pipeline batch them, which is much faster.
- `time.perf_counter()` is a precise stopwatch. Subtracting start from end gives the seconds elapsed.

### 4.4 Turn outputs into class numbers and compute accuracy

```python
preds_100 = [candidate_labels.index(o["labels"][0]) for o in outputs]
true_100 = test["label"][:100]

correct = sum(p == t for p, t in zip(preds_100, true_100))
print("Accuracy on 100:", correct / len(true_100))
```

### 4.5 Run on all 500 test texts

If 100 texts took, say, 40 s, then 500 will take ~3–4 minutes. Repeat 4.3–4.4 with `test["text"]` and `test["label"]` (all 500). Save the predictions in a variable such as `preds_A`, and the time per 100 texts (`elapsed / 500 * 100`) in a variable such as `time_A`. You will need both in Step 4.

### 4.6 Label-wording experiment (at least 3 variants)

Write a small **function** so you don't repeat code. A function takes inputs, does work, and returns a result:

```python
def zero_shot_accuracy(cand_labels, texts, true_labels, template=None):
    kwargs = {"candidate_labels": cand_labels, "batch_size": 16}
    if template:
        kwargs["hypothesis_template"] = template
    outs = zs(texts, **kwargs)
    preds = [cand_labels.index(o["labels"][0]) for o in outs]
    acc = sum(p == t for p, t in zip(preds, true_labels)) / len(true_labels)
    return acc, preds
```

Then try several label lists, **always in the order World, Sports, Business, Sci/Tech**:

| Variant | Labels |
|---|---|
| 1 – dataset names | `["World", "Sports", "Business", "Sci/Tech"]` |
| 2 – descriptive | `["world politics", "sports", "business", "science and technology"]` |
| 3 – more specific | `["international news and politics", "sports and athletes", "business, finance and economy", "science, technology and computers"]` |

To save time, run the experiment on 100–200 texts, then use the best variant on all 500. Record the accuracy of each variant in a small table (a pandas DataFrame or a Markdown cell).

### 4.7 Try a custom hypothesis template

The default template is `"This example is {}."`. Try:

```python
acc, _ = zero_shot_accuracy(best_labels, texts_100, true_100,
                            template="This news article is about {}.")
```

The `{}` is replaced by each candidate label. Does accuracy go up or down?

✅ **Checkpoint 4:** you have `preds_A` for all 500 texts, the seconds per 100 texts, and a table with ≥ 3 label variants (+ template result).

---

## 5. Step 3 — Approach B: embeddings + a simple classifier

**Time:** ~60 min. **Goal:** use a little labelled data and a fast classic classifier.

### 5.1 How it works

1. A **sentence-embedding model** (`all-MiniLM-L6-v2`, ~90 MB) turns every text into a vector of **384 numbers** that captures its meaning.
2. **Logistic regression** learns from the 2,000 training vectors which regions of this "meaning space" belong to which class.
3. To classify a new text, embed it and ask the logistic regression.

The embedding model is **not** trained here, only the small classifier on top. That is why it is fast.

### 5.2 Encode the texts

```python
from sentence_transformers import SentenceTransformer
import time

enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

start = time.perf_counter()
X_train = enc.encode(train["text"], batch_size=64, show_progress_bar=True)
print(f"Encoding train took {time.perf_counter() - start:.1f} s")

start = time.perf_counter()
X_test = enc.encode(test["text"], batch_size=64)
encode_test_time = time.perf_counter() - start

print(X_train.shape, X_test.shape)   # (2000, 384) (500, 384)
```

`X_train.shape == (2000, 384)` means 2,000 texts with 384 numbers each.

### 5.3 Train and predict

```python
from sklearn.linear_model import LogisticRegression

y_train = train["label"]
y_test = test["label"]

start = time.perf_counter()
clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
train_time_B = time.perf_counter() - start

preds_B = clf.predict(X_test)
```

- `.fit(X, y)` trains the classifier.
- `max_iter=1000` lets the optimiser run long enough to converge (otherwise you may get a "ConvergenceWarning").
- `.predict(X_test)` returns one class number per test text.

### 5.4 Accuracy

```python
from sklearn.metrics import accuracy_score
print("Accuracy B:", accuracy_score(y_test, preds_B))
```

Expect around **0.85–0.88**.

### 5.5 Learning curve: how much labelled data do you need?

Labelling data costs time and money, so this question matters for the scenario. Train the classifier on only the first 100, then 500, then 2,000 training vectors, and always test on the same 500 test texts:

```python
for n in [100, 500, 2000]:
    clf_n = LogisticRegression(max_iter=1000).fit(X_train[:n], y_train[:n])
    acc = accuracy_score(y_test, clf_n.predict(X_test))
    print(f"{n:>5} examples -> accuracy {acc:.3f}")
```

You don't need to re-encode: the first `n` rows of `X_train` belong to the first `n` training texts. Write down the numbers and answer in a Markdown cell: *How much labelled data do we actually need?*

### 5.6 Measure inference time per 100 texts

For approach B, inference = encoding + predicting:

```python
start = time.perf_counter()
_ = clf.predict(enc.encode(test["text"], batch_size=64))
time_B = (time.perf_counter() - start) / len(test) * 100
print(f"{time_B:.2f} s per 100 texts")
```

✅ **Checkpoint 5:** you have `preds_B`, accuracy ≈ 0.85+, learning-curve numbers for 100/500/2,000, training time and seconds per 100 texts.

---

## 6. Step 4 — Evaluate and compare

**Time:** ~45 min. **Goal:** measure honestly and understand the mistakes.

### 6.1 Accuracy and macro-F1

```python
from sklearn.metrics import accuracy_score, f1_score

for name, preds in [("A – zero-shot", preds_A), ("B – embeddings+LR", preds_B)]:
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    print(f"{name}: accuracy={acc:.3f}  macro-F1={f1:.3f}")
```

`average="macro"` computes F1 per class and averages them equally.

**Optional:** `from sklearn.metrics import classification_report; print(classification_report(y_test, preds_B, target_names=labels))` gives precision, recall and F1 per class.

### 6.2 Confusion matrices

```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(y_test, preds_A, display_labels=labels, ax=axes[0], colorbar=False)
axes[0].set_title("A – zero-shot")
ConfusionMatrixDisplay.from_predictions(y_test, preds_B, display_labels=labels, ax=axes[1], colorbar=False)
axes[1].set_title("B – embeddings + LR")
plt.tight_layout()
plt.show()
```

**How to read it:** rows are the **true** class, columns the **predicted** class. A number in row "Business", column "Sci/Tech" counts Business texts wrongly predicted as Sci/Tech. Big numbers on the diagonal are good.

Compare with your hypothesis from Step 1. Were you right about which classes get confused?

### 6.3 Inference time per 100 texts

You already measured `time_A` (Section 4.5) and `time_B` (Section 5.6). Print them side by side. The zero-shot pipeline is typically much slower than embeddings + LR.

### 6.4 Analyse 5 errors

Find texts where the prediction is wrong:

```python
import numpy as np

wrong = np.where(np.array(preds_B) != np.array(y_test))[0]
print(f"{len(wrong)} errors")
for i in wrong[:10]:
    print(f"[{i}] TRUE={labels[y_test[i]]}  PRED={labels[preds_B[i]]}")
    print("   ", test[int(i)]["text"][:200])
```

Pick 5 (from A, B or both) and write **one line each** in a Markdown cell explaining why the model got it wrong. Typical reasons:
- The text genuinely fits two classes (tech company earnings → Business *and* Sci/Tech).
- Misleading keywords (a "match" or "race" in a political context).
- The headline is too short or vague.
- The dataset label itself looks questionable.

✅ **Checkpoint 6:** accuracy + macro-F1 for A and B, two confusion matrices, timing for both, 5 explained errors.

---

## 7. Step 5 (optional) — Approach C: fine-tune a small transformer

**Time:** ~60–90 min (of which 15–30 min is waiting). **Goal:** see whether adapting a full transformer beats approach B.

### 7.1 How it works

`distilbert-base-uncased` (~260 MB) is a compact pre-trained language model. We add a new **classification head** (4 outputs) and train the whole network for **1 epoch** on our 2,000 examples.

### 7.2 Tokenise

Models don't read words. A **tokenizer** splits text into sub-word pieces and converts them to numbers.

```python
from transformers import AutoTokenizer

model_name = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tok(batch["text"], truncation=True, max_length=64)

train_tok = train.map(tokenize, batched=True)
test_tok = test.map(tokenize, batched=True)
```

- `truncation=True, max_length=64` cuts long texts to 64 tokens. That is enough for headlines and keeps training fast on a CPU.
- `.map(..., batched=True)` applies the function to the whole dataset efficiently.

### 7.3 Load the model

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4,
    id2label={i: l for i, l in enumerate(labels)},
    label2id={l: i for i, l in enumerate(labels)},
)
```

You see a warning: *"Some weights of DistilBertForSequenceClassification were not initialized…"*. **This is expected**: the classification head is brand new and will be trained now.

### 7.4 Configure and train

```python
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import time

args = TrainingArguments(
    output_dir="distilbert-agnews",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,
    logging_steps=25,
    save_strategy="no",
    report_to="none",
    seed=42,
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

- `DataCollatorWithPadding` pads each batch so all texts in it have the same length.
- `save_strategy="no"` skips writing checkpoints to disk.
- `report_to="none"` disables external logging services.
- Every 25 steps you see the **training loss**. It should go down.

Get a coffee: 2,000 examples ÷ 16 per batch = 125 steps, which takes ~15–30 min on a laptop CPU.

### 7.5 Predict and evaluate

```python
start = time.perf_counter()
pred_out = trainer.predict(test_tok)
time_C = (time.perf_counter() - start) / len(test) * 100

preds_C = pred_out.predictions.argmax(axis=-1)
print("Accuracy C:", accuracy_score(y_test, preds_C))
print("Macro-F1 C:", f1_score(y_test, preds_C, average="macro"))
```

`predictions` holds one score per class for every text, and `argmax` picks the highest. Also plot a confusion matrix for C (same code as 6.2, with `preds_C`).

✅ **Checkpoint 7:** `preds_C`, accuracy, macro-F1, training time, seconds per 100 texts, confusion matrix.

---

## 8. Step 6 — Results table and recommendation

### 8.1 Build the results table

Collect the numbers into a pandas DataFrame:

```python
results = pd.DataFrame([
    {"Approach": "A – Zero-shot", "Accuracy": accuracy_score(y_test, preds_A),
     "Macro-F1": f1_score(y_test, preds_A, average="macro"),
     "Sec / 100 texts": time_A, "Training time (s)": 0, "Labelled examples": 0},
    {"Approach": "B – Embeddings + LR", "Accuracy": accuracy_score(y_test, preds_B),
     "Macro-F1": f1_score(y_test, preds_B, average="macro"),
     "Sec / 100 texts": time_B, "Training time (s)": train_time_B, "Labelled examples": 2000},
    # add the C row if you did Step 5
])
results.round(3)
```

For B's training time, it is fair to include the time to encode the training set, since you have to do that to train. Note your choice in the notebook.

### 8.2 Write the recommendation (≤ 150 words)

In a Markdown cell, answer: **Which approach would you deploy for the routing scenario, and why?**

Consider:
- **Accuracy:** how many texts get routed wrongly per day?
- **Speed:** hundreds of texts a day is small, but does it still run on cheap hardware?
- **Labelling effort:** how many labelled examples does it need? (See the learning curve.)
- **Maintenance:** what happens when a new topic is added? Zero-shot only needs a new label word. B needs new examples and a 1-second retrain. C needs a new fine-tune.

**Reference point:** B typically reaches 85–88 %. Zero-shot is usually noticeably lower and very sensitive to label wording. Your numbers may differ; explain why.

---

## 9. Final check and hand-in

1. Menu: **Kernel → Restart & Run All**.
2. Wait until every cell has a number and there are no red errors.
3. Go through the checklist:

- [ ] Notebook runs end to end on a CPU laptop
- [ ] Results table: accuracy, macro-F1, seconds per 100 texts, training time, labelled examples used
- [ ] Confusion matrix for each approach
- [ ] Label-wording experiment with at least 3 variants
- [ ] Learning-curve numbers (100 / 500 / 2,000)
- [ ] 5 analysed errors, one line each
- [ ] Recommendation ≤ 150 words

4. Save (`Ctrl/Cmd + S`) and prepare a **5-minute walkthrough** for the club: scenario → three approaches → results table → one interesting error → recommendation.

### Stretch goals (pick any)

- **Gradio demo:** `pip install gradio`, then wrap your best model in ~10 lines (see Level 4, section 9).
- **Hugging Face Spaces:** publish the Gradio app for free, or `push_to_hub()` your fine-tuned model (needs a free Hugging Face account and access token).
- **Your own data:** label 50–100 texts from your domain (support tickets, product reviews), put them in a CSV with `text,label` columns, load with `load_dataset("csv", data_files="my.csv")` and rerun approach B.
- **Multilingual:** replace the embedding model with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` and try Czech or German headlines.

---

## 10. Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `python` not found | Python not installed or not on PATH | Reinstall with "Add to PATH" ticked, or use `python3` |
| `ModuleNotFoundError: No module named 'transformers'` | Library not installed in *this* environment | Activate `hf-env`, then `pip install …` again. In Jupyter, check that the kernel is the one from `hf-env` (start Jupyter from the activated terminal). |
| First run is slow | Models and dataset are downloading | Normal. They are cached in `~/.cache/huggingface`, so later runs are instant |
| Out of memory / laptop freezes | Batch too big | Lower `batch_size` to 8; use `max_length=64` |
| Zero-shot takes forever | 500 texts × 4 labels = 2,000 model calls | Test on 100 first; close other heavy apps |
| Download blocked (SSL / proxy error) | Corporate proxy | Set `HTTPS_PROXY` (ask IT for the value), or use Colab/Kaggle |
| `ValueError: ... is not in list` in zero-shot | Typo: the predicted label is not exactly in your `candidate_labels` list | Always use the same list variable for the pipeline call and for `.index()` |
| `ConvergenceWarning` from LogisticRegression | Not enough iterations | Use `max_iter=1000` (or 2000) |
| "Some weights … not initialized" | New classification head | Expected, ignore |
| `ImportError: Using the Trainer with PyTorch requires accelerate` | Missing package | `pip install accelerate`, then restart the kernel |
| `TypeError: unexpected keyword 'evaluation_strategy'` | Older tutorial code with newer `transformers` | Use `eval_strategy`, or leave it out as in this guide |
| Accuracy around 0.25 | Predictions are effectively random; often a label-order mismatch | Check that your candidate labels are in the order World, Sports, Business, Sci/Tech |

**Useful reading:** [Hugging Face LLM Course](https://huggingface.co/learn/llm-course), chapters 1–3 cover everything in this task.

---

*Still stuck? → Open **Level 4** for the full reference solution.*
