# What `run_all.py` does — step by step

This document follows `run_all.py` from the moment you type the command until the last file is written. It goes into the `news_sorter` package wherever the script calls it, because `run_all.py` itself is only about 60 lines: it reads the command line, builds the objects and starts the pipeline. All the real work happens inside the package.

**The goal of the experiment:** sort short news texts from the **AG News** dataset into four topics (**World, Sports, Business, Sci/Tech**) and compare three Hugging Face approaches by quality (accuracy, macro‑F1) and cost (speed, training time, number of labelled examples):

| Key | Approach | Training? |
|---|---|---|
| **A** | Zero-shot classification (NLI model `cross-encoder/nli-distilroberta-base`) | none |
| **B** | Sentence embeddings (`all-MiniLM-L6-v2`) + logistic regression | ~seconds |
| **C** | Fine-tuned DistilBERT (`distilbert-base-uncased`), optional | 15–30 min on a CPU |

---

## 0. How to start it

```bash
python run_all.py                    # approaches A + B (~15 min on a CPU)
python run_all.py --with-finetune    # A + B + C (adds ~15–30 min)
python run_all.py --n-train 100 --n-test 40 --n-wording 10 --output-dir outputs_smoke   # quick smoke test
```

When Python runs the file directly, the last lines are executed:

```python
if __name__ == "__main__":
    main()
```

This guard means the experiment starts **only** when the file is run directly (`python run_all.py`), not when it is imported from other code.

---

## 1. Imports (happen before `main()` runs)

```python
import argparse
from pathlib import Path
from news_sorter import Config, NewsSorterPipeline
from news_sorter.classifiers import EmbeddingClassifier, NewsClassifier, ZeroShotClassifier
from news_sorter.config import DEFAULT_OUTPUT_DIR
```

What happens behind the scenes:

1. **`news_sorter/__init__.py`** runs and imports `Config`, `NewsData` and `NewsSorterPipeline`. Through these imports, the modules `config`, `data`, `pipeline`, `evaluation`, `exploration`, `results` and `timing` are loaded, together with the libraries they need (`datasets`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`).
2. **`news_sorter/config.py`** computes `DEFAULT_OUTPUT_DIR` = the `outputs` folder next to the package (`.../oo/outputs`). It is derived from the file's own location, so the script writes to the same place no matter which directory you start it from.
3. **`news_sorter/classifiers/__init__.py`** imports `NewsClassifier` (the abstract base), `ZeroShotClassifier` (which loads `transformers`) and `EmbeddingClassifier` (which loads `sentence_transformers`).
   - `FineTunedClassifier` is **deliberately not imported here**. It pulls in the heavy training stack (`Trainer`, `accelerate`), which the default run does not need.
4. Every module that draws plots calls `matplotlib.use("Agg")`, so charts are written straight to PNG files and no window or display is needed.

---

## 2. `main()` → `parse_args()`: reading the command line

```python
defaults = Config()
parser = argparse.ArgumentParser(...)
```

1. A default `Config()` object is created only to read the default values from it. That way the defaults live in a single place (`config.py`) and the help text shows them.
2. Five options are defined:

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--with-finetune` | flag | off | also run approach C |
| `--n-train` | int | 2000 | number of training examples |
| `--n-test` | int | 500 | number of test examples |
| `--n-wording` | int | 200 | test texts used in the zero-shot label-wording experiment |
| `--output-dir` | path | `.../oo/outputs` | folder for all produced files |

3. `parser.parse_args()` reads `sys.argv` and returns a `Namespace` object (`args.n_train`, `args.with_finetune`, …). If an option is wrong (for example `--n-train abc`), argparse prints an error and the script stops here.

---

## 3. `main()`: building the configuration

```python
config = Config(
    n_train=args.n_train,
    n_test=args.n_test,
    n_wording=args.n_wording,
    output_dir=args.output_dir.resolve(),
)
```

- `Config` is a **frozen dataclass**, which means it is read-only after it is created. No part of the code can change a setting in the middle of a run.
- `.resolve()` turns the output path into an absolute path.
- Everything not given on the command line keeps its default value:

| Setting | Value |
|---|---|
| `seed` | 42 (same shuffle → every approach sees the same texts) |
| `dataset_name` | `fancyzhx/ag_news` |
| `zero_shot_model` | `cross-encoder/nli-distilroberta-base` |
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` (~90 MB) |
| `learning_curve_sizes` | (100, 500, 2000) |
| `finetune_model` | `distilbert-base-uncased` (~260 MB) |
| `finetune_max_length` | 64 tokens |
| `finetune_epochs` | 1 |

---

## 4. `main()` → `build_classifiers()`: creating the classifiers

```python
classifiers = [ZeroShotClassifier(config), EmbeddingClassifier(config)]
if with_finetune:
    from news_sorter.classifiers.fine_tuned import FineTunedClassifier
    classifiers.append(FineTunedClassifier(config))
```

> ⚠️ **Important detail:** creating the objects already **loads the models**. This happens *before* the pipeline starts, so on the first run you will see model downloads before the "Step 1" headline appears. Models are cached by Hugging Face afterwards, so later runs skip the download.

What each constructor does:

**4a. `ZeroShotClassifier(config)` (approach A)**
1. The base class `NewsClassifier.__init__` stores `config` and sets `train_time_s = 0.0` and `n_labelled = 0`.
2. `batch_size = 16`.
3. `pipeline("zero-shot-classification", model="cross-encoder/nli-distilroberta-base")` downloads or loads the NLI model and its tokenizer.
4. A starting prompt is set, `"2 descriptive"` (`world politics`, `sports`, `business`, `science and technology`), so `predict()` would work even without `fit()`.

**4b. `EmbeddingClassifier(config)` (approach B)**
1. Base-class initialisation, as above.
2. `batch_size = 64`.
3. `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")` downloads or loads the encoder.
4. A fresh, untrained `LogisticRegression(max_iter=1000)` is created.
5. `_X_train = None`. The training embeddings will be cached here later.

**4c. `FineTunedClassifier(config)` (approach C, only with `--with-finetune`)**
1. Only now is the module imported, together with `Trainer`, `TrainingArguments` and the rest.
2. Base-class initialisation.
3. `AutoTokenizer.from_pretrained("distilbert-base-uncased")` loads the tokenizer. The model itself is created later, in `fit()`.
4. `_trainer = None`.

The result is a list of classifier objects that all share the same interface (`NewsClassifier`).

---

## 5. `main()`: creating the pipeline and calling `run()`

```python
NewsSorterPipeline(config, build_classifiers(config, args.with_finetune)).run()
```

`NewsSorterPipeline.__init__` only stores the config and the list of classifiers. The pipeline **does not know which approaches exist**. It receives ready-made objects from outside (*dependency injection*) and talks to them only through the `NewsClassifier` interface (*polymorphism*).

Everything from here on happens in `NewsSorterPipeline.run()`.

---

## 6. `run()` – preparation

1. `output_dir.mkdir(parents=True, exist_ok=True)` creates the output folder (and any missing parent folders) if it does not exist yet.
2. `_remove_stale_predictions()` deletes every `preds_*.npy` file left from earlier runs. Without this, for example, `preds_C.npy` from an old run with fine-tuning would stay next to fresh A/B files and look current.
   - Note: only the `preds_*.npy` files are deleted. Other files of an approach that is not run this time (for example `confusion_C.png`, `errors_C.csv`) stay in the folder.
3. `with Timer() as total:` starts a stopwatch for the whole experiment. `Timer` uses `time.perf_counter()`, the most precise clock for measuring durations, and it records the time even if an error occurs.

---

## 7. Step 1 – load and explore the data

A headline is printed (`>>> Step 1 - load and explore the data`). `flush=True` makes sure it appears immediately, even when the output is redirected to a file.

### 7a. `NewsData.load(config)`
1. `load_dataset("fancyzhx/ag_news")` downloads AG News from the Hugging Face Hub, or reads it from the local cache. The full dataset has 120,000 training and 7,600 test texts.
2. `dataset["train"].shuffle(seed=42).select(range(2000))` gives a reproducible random sample of 2,000 training texts.
3. `dataset["test"].shuffle(seed=42).select(range(500))` gives a reproducible random sample of 500 test texts.
4. The class names are read from the `ClassLabel` feature: `['World', 'Sports', 'Business', 'Sci/Tech']`. The position in this list is the class id, 0–3.
5. `y_train` and `y_test` are stored as NumPy arrays of class ids.
6. A **frozen** `NewsData` object is returned. This one object is shared by all classifiers, which guarantees that they are trained and tested on exactly the same texts.

### 7b. `DataExplorer(data, output_dir).report()`
1. It prints the labels and the subset sizes (`2000 500`).
2. `_print_examples(5)` prints the first 5 training texts for each of the four classes.
3. `class_balance()` builds a table with the number of examples per class in train and test, in the official class order.
4. The table is printed and saved to **`outputs/class_balance.csv`**.
5. It prints a reminder to write your hypothesis (which classes will be confused?) into `REPORT.md`.

Why the balance check matters: with balanced classes, plain accuracy is a trustworthy metric.

---

## 8. Preparing the shared measuring tools

```python
evaluator = Evaluator(data, output_dir)
report = ResultsReport(output_dir)
```

- **`Evaluator`** is a single evaluator shared by all approaches, so all of them are measured in exactly the same way.
- **`ResultsReport`** starts with an empty list of results and later **overwrites** the result files, so old results can never leak into the table.

---

## 9. Steps 2–5 – the same loop for every classifier

```python
for classifier in self.classifiers:
    self._headline(classifier.name)
    classifier.fit(data)                                    # (1) prepare / train
    preds, sec_per_100 = classifier.timed_predict(data.test_texts)   # (2) predict + time it
    report.add(evaluator.evaluate(classifier, preds, sec_per_100))   # (3) evaluate
    classifier.run_experiments(data)                        # (4) extra analysis
    classifier.save()                                       # (5) save the model
```

The order is **A → B → (C)**. Each approach goes through the same 5 actions below.

### 9.1 Shared parts (from the base class `NewsClassifier`)

**`timed_predict(texts)`**
1. It starts a `Timer`.
2. It calls the subclass's `predict(texts)` on all 500 test texts.
3. It returns `(predictions, seconds / number_of_texts × 100)`, the speed in **seconds per 100 texts**. This keeps runs with different test sizes comparable. The time is measured end to end; for B it includes computing the embeddings.

**`evaluator.evaluate(classifier, preds, sec_per_100)`**
1. It computes **accuracy** (the share of correct predictions) and **macro‑F1** (F1 per class, then averaged, so every class counts equally).
2. It builds an `ApproachResult` with key, name, accuracy, macro‑F1, speed, training time and number of labelled examples.
3. It prints a summary line and then a `classification_report` (precision / recall / F1 per class).
4. It saves the raw predictions to **`preds_X.npy`**.
5. `_save_confusion_matrix` draws a confusion matrix (rows = true class, columns = predicted class, diagonal = correct) and saves it to **`confusion_X.png`**.
6. `_save_errors` builds a table of all wrong predictions (`idx`, `true`, `pred`, `text`), saves it to **`errors_X.csv`** and prints the first 8 errors, each text shortened to 160 characters.
7. It returns the `ApproachResult`. `report.add(...)` stores it for the final table.

`run_experiments()` and `save()` do nothing by default. A subclass overrides them only if it has something to do.

### 9.2 Approach A – `ZeroShotClassifier`

**How it works:** for each candidate label, the pipeline builds a hypothesis sentence (e.g. *"This example is sports."*) and asks the NLI model whether the news text **entails** it. The label with the highest entailment score wins. Each text costs 4 model calls (one per label), so this is the slowest approach at inference time.

**`fit(data)`** (no training, only prompt selection)
1. `_show_example` runs the pipeline on the first test text with the starting prompt and prints the text, its true class and the raw pipeline output (a dict with labels sorted by score), so you can see what the output looks like.
2. `wording_experiment(data)` takes the first `n_wording` = 200 test texts and tries **4 label wordings × 2 templates = 8 combinations**:
   - Label wordings: `1 dataset names` (World, Sports…), `2 descriptive` (world politics, sports, business, science and technology), `3 specific` (international news and politics…), `4 one-word` (politics, sports, economy, technology)
   - Templates: `default` (*"This example is {}."*) and `news` (*"This news article is about {}."*)
   - For each combination it measures accuracy and speed and prints the row.
3. The table is sorted by accuracy, best first. The sort is *stable*, so when two combinations tie, the one tried first wins and the choice is deterministic.
4. The table is printed and saved to **`zero_shot_wording.csv`**.
5. The best row becomes the active `self.prompt`.
6. `train_time_s = 0`, `n_labelled = 0`.

> ⚠️ Known caveat (left in on purpose): the best prompt is chosen on *test* texts, so A's final score is slightly optimistic. A stricter setup would use a separate validation set.

**`predict(texts)`** → `_predict_with(texts, best_prompt)`
1. It calls the pipeline with `candidate_labels`, `batch_size=16` and, if one is set, the `hypothesis_template`.
2. From each output it takes the top label (`output["labels"][0]`) and converts it to a class id by its **position in our label list**. That is why the labels must stay in dataset order: World, Sports, Business, Sci/Tech.

Then come the shared steps: `timed_predict` → `evaluate` (`preds_A.npy`, `confusion_A.png`, `errors_A.csv`).

**`run_experiments()`** and **`save()`** are not overridden, so nothing happens. The zero-shot model is not saved because it is not changed.

### 9.3 Approach B – `EmbeddingClassifier`

**How it works:** a pre-trained encoder (not trained here) turns each text into a vector of **384 numbers**. Texts with a similar meaning get similar vectors. Only a small logistic regression is trained on top of these vectors.

**`fit(data)`**
1. `encode(train_texts, show_progress_bar=True)` turns all 2,000 training texts into vectors, in batches of 64, with a progress bar. The result is stored in `_X_train`.
2. It prints `Train embedding shape: (2000, 384)`.
3. It trains the logistic regression on `(_X_train, y_train)`.
4. `train_time_s` = encoding time + fitting time. Encoding counts as training time, because without it there is nothing to train on.
5. `n_labelled = 2000`.

**`predict(texts)`**: it encodes the test texts and classifies the vectors with the trained logistic regression.

Then come the shared steps: `timed_predict` → `evaluate` (`preds_B.npy`, `confusion_B.png`, `errors_B.csv`).

**`run_experiments(data)`**, the learning curve ("how many texts do we need to label by hand?"):
1. It checks that `fit()` has run.
2. It encodes the test texts once more (`X_test`).
3. `_curve_sizes` gives the sizes from `learning_curve_sizes` that are smaller than `n_train`, plus `n_train` itself. With the defaults that is **100, 500, 2000**.
4. For each size *n*, it trains a **new** logistic regression on the first *n* cached training vectors (the data is already shuffled, so this is a random sample) and measures accuracy and macro‑F1 on the test set. The texts are not encoded again, because the cached `_X_train` is reused.
5. The table is printed and saved to **`learning_curve.csv`**.
6. `_plot_learning_curve` draws accuracy against the number of examples (log scale on the X axis) and saves it to **`learning_curve.png`**.

**`save()`**: it saves only the small logistic regression to **`news_sorter_lr.joblib`**, for example for a later Gradio demo. The encoder can always be downloaded again from the Hub by its name.

### 9.4 Approach C – `FineTunedClassifier` (only with `--with-finetune`)

**How it works:** unlike B, *all* weights of the transformer are updated during training, so the model specialises in this task. It usually gives the best accuracy, but by far the longest training.

**`fit(data)`**
1. `AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4, id2label=…, label2id=…)` puts a new 4-class head on the pre-trained DistilBERT body. The head starts with random weights, so the warning *"Some weights … were not initialized"* is expected. The class names are stored in the model, so it reports "Sports" instead of "LABEL_1".
2. `TrainingArguments`: 1 epoch, batch 16 (training) / 32 (evaluation), learning rate `5e-5` (kept small so the pre-trained weights are adapted, not destroyed), weight decay 0.01, a log entry every 25 steps, no intermediate checkpoints, no external loggers, seed 42.
3. `_tokenize(data.train)` turns the texts into tokens (`input_ids`, `attention_mask`) and cuts them at 64 tokens.
4. A `Trainer` is created with `DataCollatorWithPadding`, which pads each batch only to the length of its longest text. This is faster than padding everything to a global maximum.
5. `trainer.train()` runs the training inside a `Timer`, then prints `Training took X min`.
6. `train_time_s` = measured time, `n_labelled = 2000`.

**`predict(texts)`**
1. It wraps the texts in a `Dataset` and tokenizes them.
2. `trainer.predict(...)` returns **logits**, one score per class for each text.
3. `argmax` picks the class with the highest score.

Then come the shared steps: `timed_predict` → `evaluate` (`preds_C.npy`, `confusion_C.png`, `errors_C.csv`).

**`run_experiments()`** is not overridden, so nothing happens.

**`save()`**: it saves the model and the tokenizer together to the **`outputs/distilbert-agnews/`** folder, ready for reuse or `push_to_hub`.

---

## 10. Step 6 – final comparison

A headline is printed (`>>> Final comparison`), then `report.save()` runs:

1. If there are no results, it raises a `RuntimeError`.
2. `to_frame()` builds the table, one row per approach, with rounded columns: `Accuracy`, `Macro-F1`, `Sec / 100 texts`, `Training time (s)`, `Labelled examples`. The rows are sorted by name (A, B, C).
3. It prints the table.
4. It saves **`results.json`** (machine-readable) and **`results_table.csv`** (for `REPORT.md`).
5. `_plot_comparison` draws a bar chart of Accuracy and Macro‑F1 for each approach. The Y axis runs from 0 to 1 on purpose, because a cut-off axis would make small differences look dramatic. The chart is saved as **`comparison.png`**.
6. It prints where the files were saved and a reminder to copy the table into `REPORT.md` and write a recommendation (≤ 150 words).

---

## 11. The end

The `with Timer() as total:` block ends, and the script prints:

```
All steps finished in X.X min.
```

`run()` returns the `ResultsReport` object, `main()` ends, and so does the script.

---

## 12. Summary of produced files (`outputs/`)

| File | Created in | Content |
|---|---|---|
| `class_balance.csv` | Step 1 | number of examples per class (train/test) |
| `zero_shot_wording.csv` | A – `fit` | accuracy and speed of 8 prompt combinations |
| `preds_A.npy`, `preds_B.npy`, (`preds_C.npy`) | evaluation | raw predictions |
| `confusion_A.png`, `confusion_B.png`, (`confusion_C.png`) | evaluation | confusion matrix |
| `errors_A.csv`, `errors_B.csv`, (`errors_C.csv`) | evaluation | all misclassified texts |
| `learning_curve.csv`, `learning_curve.png` | B – `run_experiments` | accuracy for 100 / 500 / 2000 examples |
| `news_sorter_lr.joblib` | B – `save` | trained logistic regression |
| `distilbert-agnews/` | C – `save` | fine-tuned model + tokenizer |
| `results.json`, `results_table.csv`, `comparison.png` | Step 6 | final comparison |

Results of the last run in the folder (A + B, default settings):

| Approach | Accuracy | Macro-F1 | Sec / 100 texts | Training (s) | Labelled examples |
|---|---|---|---|---|---|
| A – Zero-shot | 0.706 | 0.688 | 19.2 | 0.0 | 0 |
| B – Embeddings + LR | 0.850 | 0.852 | 1.63 | 29.2 | 2000 |

---

## 13. Flow diagram

```
python run_all.py
 └─ main()
     ├─ parse_args()                      → args
     ├─ Config(...)                       → frozen settings
     ├─ build_classifiers()               → [A, B, (C)]   (models are loaded here!)
     └─ NewsSorterPipeline(config, clfs).run()
         ├─ mkdir outputs/, delete preds_*.npy
         ├─ Timer start
         ├─ Step 1: NewsData.load() → DataExplorer.report()     → class_balance.csv
         ├─ Evaluator + empty ResultsReport
         ├─ for each classifier:
         │    fit → timed_predict → evaluate → run_experiments → save
         │    (A: prompt choice | B: training LR + learning curve | C: fine-tuning)
         ├─ Step 6: ResultsReport.save()  → results.json / .csv / comparison.png
         └─ "All steps finished in X min."
```
