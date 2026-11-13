# Level 2 — Your First Hugging Face NLP Pipeline: Structured Step-by-Step Plan

> This level tells you **what** to do in each step and **what result** to produce.
> Need more help on *how*? → [**Level 3**](03_Detailed_Beginner_Guide.md). Need working code? → [**Level 4**](04_Reference_Solution.md).

---

## Goal

Build a "News Sorter" that assigns news headlines to one of four topics (**World, Sports, Business, Sci/Tech**). Compare three Hugging Face approaches on the same test set and recommend the best one for an automatic routing scenario.

**Dataset:** [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) has 120,000 training and 7,600 test texts in 4 balanced classes. You will only use small subsets.

**Working style:** plain Python scripts run from the terminal (`python step1_explore.py`), with no Jupyter. Plots are saved as PNG files, tables as CSV files, and your written answers go into `REPORT.md`.

---

## Step 0 — Prepare your environment and project (≈ 20 min)

| Requirement | Minimum |
|---|---|
| RAM | 8 GB |
| Free disk | 3 GB (models + dataset cache) |
| CPU | Any 4-core laptop CPU from the last ~6 years |
| Python | 3.10 or newer |
| Internet | Needed for the first download only |

**To do:**
1. Create a project folder `hf-news-sorter` and a Python virtual environment inside it.
2. Install the **CPU build** of PyTorch, then: `transformers`, `accelerate` (needed by the `Trainer` in Step 5), `datasets`, `evaluate`, `sentence-transformers`, `scikit-learn`, `pandas`, `matplotlib`.
3. Pick a text editor or IDE (e.g. VS Code, PyCharm, Notepad++, or any plain editor).
4. Write a tiny script `check_setup.py` that runs a one-line sentiment-analysis `pipeline`, and run it with `python check_setup.py`.
5. Plan the project layout. A suggested structure:

```
hf-news-sorter/
├── common.py            # shared settings: seed, subset sizes, data loading, helpers
├── step1_explore.py
├── step2_zero_shot.py
├── step3_embeddings.py
├── step4_evaluate.py
├── step5_finetune.py    # optional
├── step6_results.py
├── run_all.py           # runs all steps in order
├── REPORT.md            # your written answers
└── outputs/             # CSV, PNG and saved predictions
```

**Design tip:** put data loading (same seed, same subsets) in `common.py` so every script uses identical data. Let each step **save its predictions and metrics to `outputs/`**, so later steps (evaluation, results table) can reuse them without rerunning slow steps.

*Alternative:* Google Colab or Kaggle if you cannot install Python locally. Upload your `.py` files there and run `!python step1_explore.py`.

**Done when:** `python check_setup.py` prints a label and a score.

---

## Step 1 — Load and explore the data (≈ 30 min) → `step1_explore.py`

**To do:**
1. Load `fancyzhx/ag_news` with the `datasets` library.
2. Create reproducible subsets: shuffle with `seed=42`, take **2,000 training** and **500 test** examples.
3. Get the label names from the dataset features.
4. Print **5 examples per class** and read them.
5. Check the **class balance** of both subsets (convert to pandas), print it and save it as CSV.
6. Write your hypothesis into `REPORT.md`: *which classes do you expect to be confused, and why?*

**Done when:** the script prints the examples and class counts, and `REPORT.md` contains your hypothesis.

---

## Step 2 — Approach A: zero-shot classification (≈ 45 min) → `step2_zero_shot.py`

No training. A model trained on natural language inference (NLI) decides which candidate label "fits" the text best.

**To do:**
1. Create a `zero-shot-classification` pipeline with model `cross-encoder/nli-distilroberta-base`.
2. Try it on a single test text with candidate labels such as `"world politics", "sports", "business", "science and technology"`, and print the result.
3. Run it on **all 500** test texts (pass a list, use `batch_size=16`) and **measure the time**.
4. Convert predicted label strings back to class IDs (0–3) and compute accuracy.
5. **Label-wording experiment:** try at least **3 label variants** (e.g. `"Sci/Tech"` vs `"science and technology"`) and record the accuracy of each.
6. Try a custom `hypothesis_template`, e.g. `"This news article is about {}."`
7. Save: the experiment table (CSV), the 500 predictions, and the metrics for the results table.

**Tip:** run the experiment on 100–200 texts, and only the best variant on all 500. Zero-shot is the slowest approach.

**Done when:** you have a saved table of label variants with their accuracies, plus saved predictions and run time for 500 texts.

---

## Step 3 — Approach B: embeddings + simple classifier (≈ 60 min) → `step3_embeddings.py`

Turn each text into a numeric vector (an embedding), then train a logistic regression on those vectors.

**To do:**
1. Load the sentence-embedding model `sentence-transformers/all-MiniLM-L6-v2` (~90 MB).
2. Encode the training and test texts (`batch_size=64`). Time the encoding.
3. Train `LogisticRegression(max_iter=1000)` on the training embeddings and labels.
4. Predict on the test set, compute accuracy and measure inference time per 100 texts.
5. **Learning curve:** repeat training with **100, 500 and 2,000** examples. Record accuracy for each and save a CSV (and optionally a PNG chart).
6. Save the predictions and metrics. Optionally save the trained classifier (`joblib`) for a demo app.

**Done when:** you have test accuracy, a saved learning-curve table, and you can answer *"How much labelled data do we actually need?"*

---

## Step 4 — Evaluate and compare (≈ 45 min) → `step4_evaluate.py`

**To do:**
1. Load the saved predictions of A and B.
2. Compute **accuracy** and **macro-F1** (and a per-class report) with `sklearn.metrics` or `evaluate`.
3. Create a **confusion matrix** for each approach (`ConfusionMatrixDisplay`) and **save it as a PNG** (`plt.savefig`).
4. Compare **inference time per 100 texts** for each approach.
5. Save all misclassified examples to CSV. Choose **5** and explain in one line each in `REPORT.md` why the model got them wrong.
6. Compare the confusion matrices with your hypothesis from Step 1.

**Done when:** you have metrics, two confusion-matrix PNGs, timing numbers and 5 analysed errors in `REPORT.md`.

---

## Step 5 (optional) — Approach C: fine-tune a small transformer (≈ 60–90 min) → `step5_finetune.py`

**To do:**
1. Load `distilbert-base-uncased` (~260 MB) with a 4-class classification head.
2. Tokenise the 2,000 training texts with `max_length=64`.
3. Fine-tune for **1 epoch** with the `Trainer` API, `per_device_train_batch_size=16` (≈ 15–30 min on a CPU).
4. Predict on the 500 test texts, compute the same metrics and save a confusion-matrix PNG.
5. Save predictions and metrics so the result appears in the comparison table. Optionally save the model.

**Note:** a warning that "some weights were not initialized" is expected, because the classification head is new.

---

## Step 6 — Results and recommendation (≈ 20 min) → `step6_results.py` + `run_all.py`

**To do:**
1. Read all saved metrics and print/save a results table with one row per approach:

| Approach | Accuracy | Macro-F1 | Seconds / 100 texts | Training time | Labelled examples used |
|---|---|---|---|---|---|
| A – Zero-shot | | | | 0 | 0 |
| B – Embeddings + LR | | | | | 2,000 |
| C – Fine-tuned DistilBERT | | | | | 2,000 |

2. Copy the table into `REPORT.md` and write a recommendation (**≤ 150 words**): which approach would you deploy for the routing scenario, and why? Consider accuracy, speed, labelling effort and maintenance.
3. Write `run_all.py`, which runs all step scripts in order and stops at the first error. Clear old results first.
4. Run `python run_all.py` from a clean state to confirm that everything runs end to end.

**Reference point:** approach B typically reaches **85–88 %** accuracy with 2,000 examples. Zero-shot is usually noticeably lower and very sensitive to label wording. Your numbers will differ, and explaining the difference is the point.

---

## Acceptance checklist

- [ ] `python run_all.py` runs end to end on a CPU laptop without errors
- [ ] Results table: accuracy, macro-F1, seconds per 100 texts, training time, labelled examples used
- [ ] Confusion matrix (PNG) for each approach
- [ ] Label-wording experiment with at least 3 variants
- [ ] Learning-curve numbers (100 / 500 / 2,000 examples)
- [ ] 5 analysed errors, one line each (in `REPORT.md`)
- [ ] Recommendation ≤ 150 words (in `REPORT.md`)

**Hand in:** the project folder (without the virtual environment), including `REPORT.md` and `outputs/`, plus a 5-minute walkthrough at the next club session.

---

## Stretch goals (pick any)

- Wrap your best model in a **Gradio** app (`app.py`, ~15 lines) and demo it live.
- Publish the app on **Hugging Face Spaces**, or push your fine-tuned model to the Hub with `push_to_hub()`.
- Use your **own texts**: 50–100 labelled examples from your domain (support tickets, reviews) in a CSV, and rerun approach B.
- Try a **multilingual** embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) on Czech or German headlines.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'common'` | Run scripts from the project folder |
| `No module named 'transformers'` | Activate the virtual environment first |
| First run is slow | Models download once to `~/.cache/huggingface`; later runs are instant |
| Out of memory | Lower `batch_size` to 8, or shorten texts with `max_length=64` |
| Zero-shot takes too long | Test on 100 texts first, then scale to 500 |
| Plots don't appear | Expected: scripts save plots as PNG in `outputs/`; open them from there |
| Corporate proxy blocks downloads | Set `HTTPS_PROXY`, or use Colab/Kaggle |
| "Some weights not initialized" when fine-tuning | Expected: the classification head is new and gets trained |
| `ImportError: ... requires accelerate` when fine-tuning | `pip install accelerate` |

---

*Need more detail on how to do a step? → [**Level 3**](03_Detailed_Beginner_Guide.md).*
