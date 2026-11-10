# Level 3 — Your First Hugging Face NLP Pipeline: Detailed Beginner Guide

> This level explains **how** to do every step, assuming you have little or no experience with Python or machine learning.
> You write **plain Python scripts** and run them from the terminal. No Jupyter is needed.
> Each step ends with a **✅ Checkpoint**, so you know whether you can move on.
> Still stuck after reading a step? → The matching section in **Level 4** has the complete, working file.

---

## Contents

0. [Key terms explained](#0-key-terms-explained)
1. [Set up your computer](#1-set-up-your-computer)
2. [Create your project](#2-create-your-project)
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
| **Seed** | A fixed number for the random generator, so "random" shuffles are the same every time you run a script (reproducibility). |
| **Script** | A text file ending in `.py` that contains Python code. `python my_script.py` runs it from the first line to the last. |
| **Module / import** | A script can reuse code from another file: `from common import load_data` takes the function `load_data` out of the file `common.py`. |

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

> **Cannot install anything on your work laptop?** Skip to [1.7 Cloud alternative](#17-cloud-alternative-colab-or-kaggle).

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
pip install transformers accelerate datasets evaluate sentence-transformers scikit-learn pandas matplotlib
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

Installation takes a few minutes. Warnings in yellow are usually fine. Red **ERROR** lines are not; see [Troubleshooting](#10-troubleshooting).

### 1.6 Pick a text editor

A script is just a text file, so any plain-text editor works. Good free choices:

- **VS Code** ([code.visualstudio.com](https://code.visualstudio.com)): open the whole folder with **File → Open Folder… → `hf-news-sorter`**. It has a built-in terminal (**Terminal → New Terminal**), so you can edit and run in one window.
- **PyCharm Community**, **Notepad++** (Windows), or the editor already on your computer.

Do **not** use Word or another word processor: they add invisible formatting that Python cannot read.

Two rules that save beginners a lot of time:
- **Save the file (`Ctrl/Cmd + S`) before you run it.** The terminal runs what is on disk, not what is on your screen.
- **Indentation matters in Python.** Lines inside a `def`, `for` or `if` are indented by 4 spaces. Copy the code as shown.

### 1.7 Cloud alternative: Colab or Kaggle

If you cannot install Python locally:
1. Go to [Google Colab](https://colab.research.google.com) (Google account) or [Kaggle Code](https://www.kaggle.com/code) (Kaggle account) and create a new notebook. You only use it as a **remote terminal**; your code still lives in `.py` files.
2. Write the `.py` files on your own computer with any text editor and upload them (Colab: folder icon on the left → upload).
3. Install the missing libraries once:
   ```
   !pip install -q sentence-transformers datasets evaluate accelerate
   ```
   (`torch`, `transformers`, `scikit-learn`, `pandas` and `matplotlib` are usually pre-installed.)
4. Run each script with a leading `!`, for example `!python step1_explore.py`. Everything else in this guide is identical.

✅ **Checkpoint 1:** `pip list` (in the activated environment) shows `torch`, `transformers`, `datasets` and `sentence-transformers`.

---

## 2. Create your project

### 2.1 How working with scripts goes (1-minute intro)

The whole task is one loop, repeated many times:

1. **Edit** a `.py` file in your editor and **save** it.
2. **Run** it in the terminal: `python the_file.py`.
3. **Read** what it prints, then go back to 1.

Things to know:

- A script always runs **from the first line to the last**, from scratch. Nothing is remembered between two runs.
- Because of that, each step **saves its results to files** in an `outputs/` folder (predictions, tables, charts). Later steps load these files instead of repeating slow work.
- A script only shows what you explicitly `print(...)`.
- Charts do not pop up in a window. The scripts **save them as PNG images**; open them from `outputs/` with a double-click.
- To stop a running script, press **`Ctrl + C`** in the terminal.
- If something fails, Python prints a **traceback**. Read it from the bottom: the last line names the error, and the lines above it show the file and line number where it happened.
- Always run scripts **from inside the project folder** (`cd hf-news-sorter`) with `hf-env` activated.
- Press the **Up arrow** in the terminal to get the previous command back.

### 2.2 Smoke test

Create a new file `check_setup.py` in the project folder with this content:

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(classifier("Hugging Face runs fine on my laptop!"))
```

Save it and run it:

```bash
python check_setup.py
```

The first run downloads a small model (you see progress bars). Expected output, roughly:

```
[{'label': 'POSITIVE', 'score': 0.99...}]
```

A warning that "no model was supplied, defaulted to …" is normal.

### 2.3 The project layout you are going to build

```
hf-news-sorter/
├── hf-env/              # virtual environment (already there)
├── check_setup.py       # smoke test (done)
├── common.py            # shared settings + helper functions, used by every step
├── step1_explore.py     # Step 1 – load and explore the data
├── step2_zero_shot.py   # Step 2 – approach A
├── step3_embeddings.py  # Step 3 – approach B
├── step4_evaluate.py    # Step 4 – metrics, confusion matrices, errors
├── step5_finetune.py    # Step 5 – approach C (optional)
├── step6_results.py     # Step 6 – final results table + chart
├── run_all.py           # runs all steps in order
├── REPORT.md            # your written answers
└── outputs/             # created automatically: CSV, PNG, JSON files
```

Why several small scripts instead of one big one? Zero-shot takes minutes and fine-tuning up to half an hour. With separate scripts you can rerun the quick evaluation as often as you like without repeating the slow steps.

### 2.4 Create `REPORT.md`

Your written answers (hypothesis, error explanations, recommendation) go into a plain text file called `REPORT.md`. Create it now with these headings, and fill it in as you go:

```markdown
# News Sorter – Report

## Hypothesis (Step 1)

## Label-wording experiment (Step 2)

## Learning curve (Step 3)

## 5 analysed errors (Step 4)

## Results (Step 6)

## Recommendation (≤ 150 words)
```

`.md` means Markdown: `#` starts a heading, and tables are written with `|` characters. Level 4, section 9 shows a filled-in example.

✅ **Checkpoint 2:** `python check_setup.py` prints a `POSITIVE` label with a high score, and `REPORT.md` exists.

---

## 3. Step 1 — Load and explore the data

**Time:** ~30 min. **Goal:** get familiar with the data before modelling it.
**Files:** `common.py` (new), `step1_explore.py` (new).

### 3.1 Start `common.py`: settings and data loading

Every step needs exactly the same data. Instead of copying the loading code into every script, you write it **once** in `common.py` and import it everywhere.

Create `common.py`:

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
```

What this does:

- The `import` lines at the top load the libraries. Some of them (`json`, `plt`, the `sklearn.metrics` names) are only used by helper functions you add to this file in later steps.
- `matplotlib.use("Agg")` tells matplotlib to draw charts **into files** instead of opening a window.
- `SEED`, `N_TRAIN`, `N_TEST` are settings in one place. Names in capitals are a convention for "constants".
- `OUT = Path("outputs")` and `OUT.mkdir(exist_ok=True)` create the `outputs/` folder if it is not there yet. `OUT / "results.json"` builds the path `outputs/results.json`.
- The two `pd.set_option` lines make pandas print long texts instead of cutting them off.
- `def load_data():` defines a **function**: a named block of code that other scripts can call.
  - `load_dataset("fancyzhx/ag_news")` downloads AG News from the Hub (first run only; then it is cached). It has two **splits**: `train` (120,000 rows) and `test` (7,600 rows), each with the columns `text` and `label`.
  - Working with all 120,000 examples on a CPU would take too long. `shuffle(seed=SEED)` mixes the rows in the same random order every time, and `select(range(2000))` keeps rows 0–1999. Everyone in the club using seed `42` gets the same subsets, so results are comparable.
  - `labels` is the list of class names. `labels[0]` is `'World'`, `labels[3]` is `'Sci/Tech'`. Use `labels[number]` to turn a label number into its name.
  - `y_train` and `y_test` are the correct class numbers as numpy arrays, the format scikit-learn likes.

`common.py` is not meant to be run on its own. It does nothing visible until another script imports it.

### 3.2 Create `step1_explore.py`: load the data and look at one example

```python
"""Step 1 - Load and explore the data."""
import pandas as pd
from common import load_data, OUT

train, test, labels, y_train, y_test = load_data()
print("Labels:", labels)
print("Subset sizes:", len(train), len(test))

print(train[0])
print("Class name:", labels[train[0]["label"]])
```

Run it:

```bash
python step1_explore.py
```

- `from common import load_data, OUT` takes the function and the output folder from your `common.py`. This only works when you run the script from the project folder.
- `load_data()` returns five things, which are unpacked into five variables in one line.
- Each example is a Python dictionary: `{'text': '...', 'label': 2}`.

You should see the four label names, `2000 500`, and one example.

### 3.3 Print 5 examples per class

Add this **at the end** of `step1_explore.py`, save, and run the script again:

```python
# 5 examples per class
df_train = train.to_pandas()
df_train["label_name"] = df_train["label"].map(lambda i: labels[i])
for name in labels:
    print(f"\n===== {name} =====")
    for t in df_train.loc[df_train["label_name"] == name, "text"].head(5):
        print("-", t)
```

- `train.to_pandas()` converts the data to a pandas table (DataFrame), which is easier to filter.
- The `map(lambda i: labels[i])` line adds a column with the class *name* next to the class *number*.
- The loop prints the first 5 texts of each class.

**Read the texts.** Notice their style: a headline followed by a short description, often with an agency name like "(Reuters)" or "AP".

> **Too much text scrolling by?** Send the output to a file and open it in your editor: `python step1_explore.py > outputs/step1_log.txt`

### 3.4 Check the class balance

"Balanced" means every class has roughly the same number of examples. If one class dominated, accuracy alone could mislead.

Add at the end of `step1_explore.py`:

```python
# Class balance of both subsets
balance = pd.DataFrame({
    "train": df_train["label_name"].value_counts(),
    "test": pd.Series(y_test).map(lambda i: labels[i]).value_counts(),
}).loc[labels]
print("\nClass balance:\n" + balance.to_string())
balance.to_csv(OUT / "class_balance.csv")

print("\n-> Now write your hypothesis (which classes will be confused?) into REPORT.md")
```

- `value_counts()` counts how often each class name appears.
- `.loc[labels]` puts the rows in the order World, Sports, Business, Sci/Tech.
- `to_csv(...)` saves the table as `outputs/class_balance.csv`. A CSV file opens in Excel or any text editor.

Expect roughly 500 per class in train and 125 per class in test (not exactly, because we sampled randomly).

### 3.5 Write your hypothesis

Open `REPORT.md` and answer under **Hypothesis**: *Which classes do you expect the models to confuse, and why?*

Hint: think about a headline like *"Apple shares jump after new iPhone launch"*. Is it Business or Sci/Tech? Or *"Olympic committee faces corruption probe"*: Sports or World?

✅ **Checkpoint 3:** `python step1_explore.py` prints the label names, 5 examples per class and the class counts; `outputs/class_balance.csv` exists; `REPORT.md` contains your hypothesis.

---

## 4. Step 2 — Approach A: zero-shot classification

**Time:** ~45 min. **Goal:** see how far you get with **no training at all**.
**Files:** `step2_zero_shot.py` (new), `common.py` (two helper functions added).

### 4.1 How zero-shot works

The model `cross-encoder/nli-distilroberta-base` was trained on **Natural Language Inference**: given two sentences, does the second follow from the first? The pipeline turns each candidate label into a sentence ("This example is about sports.") and asks the model how strongly the news text *implies* it. The label with the highest score wins.

So the **wording of your labels matters**. The model only understands them as English words.

### 4.2 Create the pipeline and try one text

Create `step2_zero_shot.py`:

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

# 1) One example, to see what the pipeline returns
example = zs(test[0]["text"], candidate_labels=["world politics", "sports", "business", "science and technology"])
print("Text:", test[0]["text"])
print("True:", labels[y_test[0]])
print("Pipeline output:", example)
```

**Don't run it yet.** The import line asks `common.py` for `save_preds` and `save_result`, which do not exist so far. Add them now.

### 4.3 Add two helpers to `common.py`: saving predictions and metrics

A script forgets everything when it ends. Step 4 (evaluation) and Step 6 (results table) need this step's predictions and timing, so we save them to files.

Add **at the end** of `common.py`:

```python
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
```

- `save_preds("A", preds)` writes the predictions to `outputs/preds_A.npy` (a numpy file). `load_preds("A")` reads them back, or returns `None` if the file does not exist yet.
- `save_result(...)` computes **accuracy** and **macro-F1**, prints them, and stores one row of the final results table in `outputs/results.json`. It first reads the rows that are already there, so approaches A, B and C end up in the same file. `average="macro"` computes F1 per class and averages them equally.

Now run:

```bash
python step2_zero_shot.py
```

The pipeline output is a dictionary:
- `example["labels"]`: the candidate labels **sorted from best to worst**
- `example["scores"]`: their scores in the same order
- so `example["labels"][0]` is the prediction.

### 4.4 Write a function that classifies many texts and measures the time

You will call the pipeline many times with different labels, so wrap it in a function. In `step2_zero_shot.py`, insert this **after the `zs = pipeline(...)` line and before the `# 1) One example` block**:

```python
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
```

- Passing a **list** of texts (instead of a loop) lets the pipeline batch them, which is much faster.
- `time.perf_counter()` is a precise stopwatch. Subtracting start from end gives the seconds elapsed.
- `kwargs` collects the optional settings; `**kwargs` hands them to the pipeline.
- **Important:** keep the candidate labels in the **same order as the dataset labels** (World, Sports, Business, Sci/Tech). Then the position of the predicted label in your list equals the class number: `cand_labels.index(o["labels"][0])` gives 0–3.

### 4.5 Try it on a small batch first (100 texts)

Zero-shot is the slowest approach, because every text is checked against every label. Start small. Add **at the end** of the script as a temporary trial, and run:

```python
# Trial run on 100 texts (temporary)
trial_labels = ["world politics", "sports", "business", "science and technology"]
preds_100, secs_100 = zero_shot_predict(test["text"][:100], trial_labels)
print(f"100 texts took {secs_100:.1f} s")
print("Accuracy on 100:", accuracy_score(y_test[:100], preds_100))
```

`accuracy_score(true, predicted)` is the share of positions where both agree.

Note how long 100 texts take. If it is, say, 40 s, then 500 texts will take ~3–4 minutes. Remember: every run of the script starts from scratch, so keep the slow parts small while you are still developing.

Once this works, **delete the trial block**. The experiment below replaces it.

### 4.6 Label-wording experiment (at least 3 variants) + a custom template

Try several label lists, **always in the order World, Sports, Business, Sci/Tech**, and combine each with two **hypothesis templates**. The default template is `"This example is {}."`; the `{}` is replaced by each candidate label.

Add at the end of the script:

```python
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
```

- `label_variants` and `templates` are dictionaries: a short name on the left, the value on the right.
- The two nested `for` loops run every label variant with every template: 4 × 2 = 8 runs on `N_EXP` texts each. To save time, the experiment uses only the first 200 test texts (set `N_EXP = 100` at the top if your laptop is slow).
- Each run adds one row to `rows`; `print(rows[-1])` shows progress while you wait.
- The table is sorted from best to worst and saved as `outputs/zero_shot_wording.csv`.

Feel free to add your own label variants. Does the custom template make accuracy go up or down?

### 4.7 Run the best variant on all 500 test texts and save the results

Add at the end of the script:

```python
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

- `wording_df.iloc[0]` is the first row of the sorted table, the best variant.
- `secs_A / len(test) * 100` converts the time for 500 texts into **seconds per 100 texts**, the unit used in the results table.
- Zero-shot needs no training and no labelled examples, hence `train_time_s=0, n_labelled=0`.

Run the complete script (expect several minutes):

```bash
python step2_zero_shot.py
```

Copy the experiment table into `REPORT.md` under **Label-wording experiment**.

✅ **Checkpoint 4:** the script ends with a line like `A - Zero-shot: accuracy=…`, and `outputs/` contains `zero_shot_wording.csv` (≥ 3 label variants + template results), `preds_A.npy` and `results.json`.

---

## 5. Step 3 — Approach B: embeddings + a simple classifier

**Time:** ~60 min. **Goal:** use a little labelled data and a fast classic classifier.
**File:** `step3_embeddings.py` (new).

### 5.1 How it works

1. A **sentence-embedding model** (`all-MiniLM-L6-v2`, ~90 MB) turns every text into a vector of **384 numbers** that captures its meaning.
2. **Logistic regression** learns from the 2,000 training vectors which regions of this "meaning space" belong to which class.
3. To classify a new text, embed it and ask the logistic regression.

The embedding model is **not** trained here, only the small classifier on top. That is why it is fast.

### 5.2 Encode the texts

Create `step3_embeddings.py`:

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
```

Run it with `python step3_embeddings.py`. `X_train.shape == (2000, 384)` means 2,000 texts with 384 numbers each.

(`joblib` was installed together with scikit-learn. You use it in 5.6.)

### 5.3 Train, predict and measure

Add at the end of the script:

```python
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
```

- `.fit(X, y)` trains the classifier.
- `max_iter=1000` lets the optimiser run long enough to converge (otherwise you may get a "ConvergenceWarning").
- `.predict(...)` returns one class number per test text.
- For approach B, **inference = encoding + predicting**, so the stopwatch covers both.
- **Training time** includes the time to encode the training set, since you have to do that before you can train. That keeps the comparison with the other approaches fair.

Run the script. `save_result` prints accuracy and macro-F1. Expect an accuracy of around **0.85–0.88**.

### 5.4 Learning curve: how much labelled data do you need?

Labelling data costs time and money, so this question matters for the scenario. Train the classifier on only the first 100, then 500, then 2,000 training vectors, and always test on the same 500 test texts.

Add at the end of the script:

```python
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
```

You don't need to re-encode: the first `n` rows of `X_train` belong to the first `n` training texts.

### 5.5 Save the learning curve as a chart

In a script, a chart is **saved to a file** instead of shown on screen. Add at the end:

```python
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
```

- `plt.subplots(...)` creates an empty figure (`fig`) with one drawing area (`ax`).
- `ax.plot(...)` draws the line; `set_xscale("log")` spreads 100 / 500 / 2,000 evenly along the x-axis.
- `fig.savefig(...)` writes the PNG file. Open `outputs/learning_curve.png` with a double-click.

### 5.6 Keep the trained classifier (for the stretch goal)

Add as the last lines:

```python
# 5) Keep the trained classifier for the Gradio demo (stretch goal)
joblib.dump(clf, OUT / "news_sorter_lr.joblib")
```

`joblib.dump` saves the trained classifier to a file, so a demo app can load it later without retraining.

Run the complete script, then copy the learning-curve numbers into `REPORT.md` and answer there: *How much labelled data do we actually need?*

✅ **Checkpoint 5:** the script prints `B - Embeddings + LR: accuracy=…` (≈ 0.85+) and the learning-curve table; `outputs/` now also contains `preds_B.npy`, `learning_curve.csv` and `learning_curve.png`; `results.json` has a row for B.

---

## 6. Step 4 — Evaluate and compare

**Time:** ~45 min. **Goal:** measure honestly and understand the mistakes.
**Files:** `common.py` (two helper functions added), `step4_evaluate.py` (new).

This script is fast: it loads the predictions that steps 2 and 3 saved, so no model has to run again.

### 6.1 Add two helpers to `common.py`: confusion matrix and error table

Add at the end of `common.py`:

```python
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

- `save_confusion` draws a confusion matrix and saves it as a PNG. `plt.close(fig)` frees the memory afterwards.
- `errors_table` finds every position where the prediction differs from the true label (`np.where(...)`) and returns a table with the index, true class, predicted class and text.

`common.py` is now complete. It should match Level 4, section 1.

### 6.2 Create `step4_evaluate.py`

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

Run it:

```bash
python step4_evaluate.py
```

The loop does the same three things for approach A and then for approach B. If the predictions of one approach are missing, it tells you which step to run first and carries on.

### 6.3 Accuracy and macro-F1

`classification_report` prints **precision, recall and F1 for every class**, plus two summary lines:
- the `accuracy` row is the overall accuracy,
- the `macro avg` row, column `f1-score`, is the **macro-F1**.

These are the same numbers `save_result` already printed in steps 2 and 3. Which class has the lowest F1?

### 6.4 Confusion matrices

Open `outputs/confusion_A.png` and `outputs/confusion_B.png`.

**How to read them:** rows are the **true** class, columns the **predicted** class. A number in row "Business", column "Sci/Tech" counts Business texts wrongly predicted as Sci/Tech. Big numbers on the diagonal are good.

Compare with your hypothesis from Step 1. Were you right about which classes get confused?

### 6.5 Inference time per 100 texts

Steps 2 and 3 already measured this and stored it in `outputs/results.json`. Open that file in your editor and compare the `"Sec / 100 texts"` values. The zero-shot pipeline is typically much slower than embeddings + LR.

### 6.6 Analyse 5 errors

The script printed the first 8 errors of each approach and saved **all** of them to `outputs/errors_A.csv` and `outputs/errors_B.csv`. Open the CSV files in Excel or your editor.

Pick 5 errors (from A, B or both) and write **one line each** in `REPORT.md` explaining why the model got it wrong. Typical reasons:
- The text genuinely fits two classes (tech company earnings → Business *and* Sci/Tech).
- Misleading keywords (a "match" or "race" in a political context).
- The headline is too short or vague.
- The dataset label itself looks questionable.

✅ **Checkpoint 6:** per-class reports for A and B printed, `confusion_A.png` and `confusion_B.png` saved, timing for both known, 5 explained errors in `REPORT.md`.

---

## 7. Step 5 (optional) — Approach C: fine-tune a small transformer

**Time:** ~60–90 min (of which 15–30 min is waiting). **Goal:** see whether adapting a full transformer beats approach B.
**File:** `step5_finetune.py` (new).

### 7.1 How it works

`distilbert-base-uncased` (~260 MB) is a compact pre-trained language model. We add a new **classification head** (4 outputs) and train the whole network for **1 epoch** on our 2,000 examples.

Because training takes a while and a script always runs from top to bottom, **type in the whole script first** (7.2–7.5), check it for typos, and only then run it.

### 7.2 Tokenise

Models don't read words. A **tokenizer** splits text into sub-word pieces and converts them to numbers.

Create `step5_finetune.py`:

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
```

- `truncation=True, max_length=64` cuts long texts to 64 tokens. That is enough for headlines and keeps training fast on a CPU.
- `.map(..., batched=True)` applies the function to the whole dataset efficiently.

### 7.3 Load the model

Add at the end:

```python
# 2) Model with a new 4-class head ("Some weights ... not initialized" warning is expected)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label={i: l for i, l in enumerate(labels)},
    label2id={l: i for i, l in enumerate(labels)},
)
```

When you run the script you will see a warning: *"Some weights of DistilBertForSequenceClassification were not initialized…"*. **This is expected**: the classification head is brand new and will be trained now.

### 7.4 Configure and train

Add at the end:

```python
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
```

- `DataCollatorWithPadding` pads each batch so all texts in it have the same length.
- `save_strategy="no"` skips writing checkpoints to disk.
- `report_to="none"` disables external logging services.
- Every 25 steps the terminal shows the **training loss**. It should go down.

### 7.5 Predict, evaluate and save

Add at the end:

```python
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

- `predictions` holds one score per class for every text, and `argmax` picks the highest.
- The rest reuses your helpers from `common.py`: metrics into `results.json`, a confusion matrix as PNG, all errors as CSV.
- The last lines store the fine-tuned model in `outputs/distilbert-agnews`, so you never have to train it again just to use it.

Now run it:

```bash
python step5_finetune.py
```

Get a coffee: 2,000 examples ÷ 16 per batch = 125 steps, which takes ~15–30 min on a laptop CPU. Leave the terminal open and don't let the laptop go to sleep.

✅ **Checkpoint 7:** the script prints `C - Fine-tuned DistilBERT: accuracy=…` and the training time; `outputs/` contains `preds_C.npy`, `confusion_C.png` and `errors_C.csv`; `results.json` has a row for C.

---

## 8. Step 6 — Results table and recommendation

**File:** `step6_results.py` (new).

### 8.1 Build the results table

Every approach stored its numbers in `outputs/results.json`. This script only reads that file and turns it into a table and a chart.

Create `step6_results.py`:

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

- `raise SystemExit(...)` stops the script with a clear message if there are no results yet.
- `pd.DataFrame(...).T` turns the JSON data into a table with **one row per approach** (`.T` swaps rows and columns).
- `.plot.bar(...)` draws a bar chart of accuracy and macro-F1, saved as `outputs/comparison.png`.

Run it:

```bash
python step6_results.py
```

Copy the table into `REPORT.md` under **Results**. A Markdown table looks like this:

```markdown
| Approach | Accuracy | Macro-F1 | Sec / 100 texts | Training time (s) | Labelled examples |
|---|---|---|---|---|---|
| A - Zero-shot | … | … | … | 0 | 0 |
| B - Embeddings + LR | … | … | … | … | 2000 |
```

### 8.2 Write the recommendation (≤ 150 words)

In `REPORT.md`, answer: **Which approach would you deploy for the routing scenario, and why?**

Consider:
- **Accuracy:** how many texts get routed wrongly per day?
- **Speed:** hundreds of texts a day is small, but does it still run on cheap hardware?
- **Labelling effort:** how many labelled examples does it need? (See the learning curve.)
- **Maintenance:** what happens when a new topic is added? Zero-shot only needs a new label word. B needs new examples and a 1-second retrain. C needs a new fine-tune.

**Reference point:** B typically reaches 85–88 %. Zero-shot is usually noticeably lower and very sensitive to label wording. Your numbers may differ; explain why.

✅ **Checkpoint 8:** `outputs/results_table.csv` and `outputs/comparison.png` exist, and `REPORT.md` contains the table and your recommendation.

---

## 9. Final check and hand-in

### 9.1 Write `run_all.py`

So far you ran each script by hand, some of them many times. The final check proves that everything works **from a clean state, in the right order**. `run_all.py` does that with one command.

Create `run_all.py`:

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

- `sys.argv` holds the words you typed after `python run_all.py`. That is how the script notices `--with-finetune`.
- The `unlink` loop deletes old results and predictions, so the final table only contains numbers from this run.
- `subprocess.run([sys.executable, script], check=True)` starts each script exactly as `python script.py` would, using the same Python (the one from `hf-env`). `check=True` stops everything at the first error.

### 9.2 Run the final check

```bash
python run_all.py
```

or, if you did Step 5:

```bash
python run_all.py --with-finetune
```

It must finish with `All steps finished in … min.` and no traceback. Because the fresh run overwrites the files in `outputs/`, check that the numbers in `REPORT.md` still match and update them if needed.

### 9.3 Checklist

- [ ] `python run_all.py` runs end to end on a CPU laptop without errors
- [ ] Results table: accuracy, macro-F1, seconds per 100 texts, training time, labelled examples used (`outputs/results_table.csv` → `REPORT.md`)
- [ ] Confusion matrix (PNG) for each approach (`outputs/confusion_*.png`)
- [ ] Label-wording experiment with at least 3 variants (`outputs/zero_shot_wording.csv` → `REPORT.md`)
- [ ] Learning-curve numbers for 100 / 500 / 2,000 (`outputs/learning_curve.csv` → `REPORT.md`)
- [ ] 5 analysed errors, one line each (in `REPORT.md`)
- [ ] Recommendation ≤ 150 words (in `REPORT.md`)

### 9.4 Hand in

Zip the project folder **without `hf-env/`** (it is large and can be recreated at any time), including `REPORT.md` and `outputs/`. Prepare a **5-minute walkthrough** for the club: scenario → three approaches → results table → one interesting error → recommendation.

### Stretch goals (pick any)

Each stretch goal is one more small script. Level 4, section 11 has complete versions.

- **Gradio demo (`app.py`):** `pip install gradio`, then wrap your best model in ~15 lines. Start it with `python app.py` and open the printed address in your browser.
- **Hugging Face Spaces:** publish the Gradio app for free, or `push_to_hub()` your fine-tuned model (needs a free Hugging Face account and access token).
- **Your own data (`own_data.py`):** label 50–100 texts from your domain (support tickets, product reviews), put them in a CSV with `text,label` columns, load with `load_dataset("csv", data_files="my.csv")` and rerun approach B.
- **Multilingual (`multilingual.py`):** replace the embedding model with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` and try Czech or German headlines.

---

## 10. Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `python` not found | Python not installed or not on PATH | Reinstall with "Add to PATH" ticked, or use `python3` |
| `can't open file '…step1_explore.py'` | The terminal is in a different folder, or the file name has a typo | `cd` into `hf-news-sorter`; list the files with `dir` (Windows) or `ls` |
| `ModuleNotFoundError: No module named 'transformers'` | Library not installed in *this* environment | Activate `hf-env` (the prompt must start with `(hf-env)`), then `pip install …` again |
| `ModuleNotFoundError: No module named 'common'` | The script was started from another folder | Run it from the project folder, where `common.py` lives |
| `ImportError: cannot import name 'save_preds' from 'common'` | The helper function is not in `common.py` yet, or the file was not saved | Add the function (Sections 4.3 and 6.1) and save `common.py` |
| `IndentationError` / `SyntaxError` | Spaces lost while copying, or a missing bracket or quote | Compare the line named in the traceback with the guide; use 4 spaces per level |
| My change has no effect | The file was not saved, or you ran a different script | Save (`Ctrl/Cmd + S`) and run again |
| No chart window appears | Expected: scripts save charts as PNG | Open the files in `outputs/` |
| `step4` / `step6` says results not found | An earlier step has not run yet | Run steps 2 and 3 first, or simply `python run_all.py` |
| First run is slow | Models and dataset are downloading | Normal. They are cached in `~/.cache/huggingface`, so later runs are much faster |
| Out of memory / laptop freezes | Batch too big | Lower `batch_size` to 8; use `max_length=64` |
| Zero-shot takes forever | Every text is checked against every label, for every variant | Set `N_EXP = 100` in `step2_zero_shot.py`, or remove a template; close other heavy apps |
| Download blocked (SSL / proxy error) | Corporate proxy | Set `HTTPS_PROXY` in the terminal before running (ask IT for the value), or use Colab/Kaggle |
| `UnicodeEncodeError` when printing on Windows | The terminal cannot display a special character | Run `set PYTHONUTF8=1` (cmd) or `$env:PYTHONUTF8=1` (PowerShell) before the script |
| `ValueError: ... is not in list` in zero-shot | The predicted label is not exactly in your candidate list | Always pass the same list to the pipeline and to `.index()`, as `zero_shot_predict` does |
| `ConvergenceWarning` from LogisticRegression | Not enough iterations | Use `max_iter=1000` (or 2000) |
| "Some weights … not initialized" | New classification head | Expected, ignore |
| `ImportError: Using the Trainer with PyTorch requires accelerate` | Missing package | `pip install accelerate`, then run the script again |
| `TypeError: unexpected keyword 'evaluation_strategy'` | Older tutorial code with newer `transformers` | Use `eval_strategy`, or leave it out as in this guide |
| Accuracy around 0.25 | Predictions are effectively random; often a label-order mismatch | Check that your candidate labels are in the order World, Sports, Business, Sci/Tech |

**Useful reading:** [Hugging Face LLM Course](https://huggingface.co/learn/llm-course), chapters 1–3 cover everything in this task.

---

*Still stuck? → Open **Level 4** for the full reference solution.*
