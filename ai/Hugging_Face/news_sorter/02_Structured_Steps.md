# Level 2 — Your First Hugging Face NLP Pipeline: Structured Step-by-Step Plan

> This level tells you **what** to do in each step and **what result** to produce.
> Need more help on *how*? → **Level 3**. Need working code? → **Level 4**.

---

## Goal

Build a "News Sorter" that assigns news headlines to one of four topics (**World, Sports, Business, Sci/Tech**). Compare three Hugging Face approaches on the same test set and recommend the best one for an automatic routing scenario.

**Dataset:** [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) has 120,000 training and 7,600 test texts in 4 balanced classes. You will only use small subsets.

---

## Step 0 — Prepare your environment (≈ 20 min)

| Requirement | Minimum |
|---|---|
| RAM | 8 GB |
| Free disk | 3 GB (models + dataset cache) |
| CPU | Any 4-core laptop CPU from the last ~6 years |
| Python | 3.10 or newer |
| Internet | Needed for the first download only |

**To do:**
1. Create a Python virtual environment.
2. Install the **CPU build** of PyTorch, then: `transformers`, `accelerate` (needed by the `Trainer` in Step 5), `datasets`, `evaluate`, `sentence-transformers`, `scikit-learn`, `pandas`, `matplotlib`, `jupyter`.
3. Start a Jupyter notebook and run a one-line sentiment-analysis `pipeline` to confirm that everything works.

*Alternative:* Google Colab or Kaggle notebooks if you cannot install Python locally.

**Done when:** the sentiment pipeline prints a label and a score.

---

## Step 1 — Load and explore the data (≈ 30 min)

**To do:**
1. Load `fancyzhx/ag_news` with the `datasets` library.
2. Create reproducible subsets: shuffle with `seed=42`, take **2,000 training** and **500 test** examples.
3. Get the label names from the dataset features.
4. Print **5 examples per class** and read them.
5. Check the **class balance** of both subsets (convert to pandas).
6. Write down your hypothesis: *which classes do you expect to be confused, and why?*

**Done when:** you have `train`, `test` and `labels` variables, a class count per subset, and a written hypothesis.

---

## Step 2 — Approach A: zero-shot classification (≈ 45 min)

No training. A model trained on natural language inference (NLI) decides which candidate label "fits" the text best.

**To do:**
1. Create a `zero-shot-classification` pipeline with model `cross-encoder/nli-distilroberta-base`.
2. Try it on a single test text with candidate labels such as `"world politics", "sports", "business", "science and technology"`.
3. Run it on **all 500** test texts (pass a list, use `batch_size=16`) and **measure the time**.
4. Convert predicted label strings back to class IDs (0–3) and compute accuracy.
5. **Label-wording experiment:** try at least **3 label variants** (e.g. `"Sci/Tech"` vs `"science and technology"`) and record the accuracy of each.
6. Try a custom `hypothesis_template`, e.g. `"This news article is about {}."`

**Tip:** test on 100 texts first; zero-shot is the slowest approach.

**Done when:** you have a table of label variants with their accuracies and the run time for 500 texts.

---

## Step 3 — Approach B: embeddings + simple classifier (≈ 60 min)

Turn each text into a numeric vector (an embedding), then train a logistic regression on those vectors.

**To do:**
1. Load the sentence-embedding model `sentence-transformers/all-MiniLM-L6-v2` (~90 MB).
2. Encode the training and test texts (`batch_size=64`). Time the encoding.
3. Train `LogisticRegression(max_iter=1000)` on the training embeddings and labels.
4. Predict on the test set and compute accuracy.
5. **Learning curve:** repeat training with **100, 500 and 2,000** examples. Record accuracy for each.

**Done when:** you have test accuracy and a learning-curve table, and you can answer *"How much labelled data do we actually need?"*

---

## Step 4 — Evaluate and compare (≈ 45 min)

**To do:**
1. Compute **accuracy** and **macro-F1** for approaches A and B (`sklearn.metrics` or `evaluate`).
2. Plot a **confusion matrix** for each approach (`ConfusionMatrixDisplay`).
3. Measure **inference time per 100 texts** for each approach.
4. Choose **5 misclassified examples** and explain in one line each why the model got them wrong.
5. Compare the confusion matrices with your hypothesis from Step 1.

**Done when:** you have metrics, two confusion matrices, timing numbers and 5 analysed errors.

---

## Step 5 (optional) — Approach C: fine-tune a small transformer (≈ 60–90 min)

**To do:**
1. Load `distilbert-base-uncased` (~260 MB) with a 4-class classification head.
2. Tokenise the 2,000 training texts with `max_length=64`.
3. Fine-tune for **1 epoch** with the `Trainer` API, `per_device_train_batch_size=16` (≈ 15–30 min on a CPU).
4. Predict on the 500 test texts and compute the same metrics as in Step 4.
5. Add the result to your comparison table.

**Note:** a warning that "some weights were not initialized" is expected, because the classification head is new.

---

## Step 6 — Results and recommendation (≈ 20 min)

**To do:**
1. Build a results table with one row per approach:

| Approach | Accuracy | Macro-F1 | Seconds / 100 texts | Training time | Labelled examples used |
|---|---|---|---|---|---|
| A – Zero-shot | | | | 0 | 0 |
| B – Embeddings + LR | | | | | 2,000 |
| C – Fine-tuned DistilBERT | | | | | 2,000 |

2. Write a recommendation (**≤ 150 words**): which approach would you deploy for the routing scenario, and why? Consider accuracy, speed, labelling effort and maintenance.
3. Restart the kernel and **Run all** to confirm that the notebook runs end to end.

**Reference point:** approach B typically reaches **85–88 %** accuracy with 2,000 examples. Zero-shot is usually noticeably lower and very sensitive to label wording. Your numbers will differ, and explaining the difference is the point.

---

## Acceptance checklist

- [ ] Notebook runs end to end on a CPU laptop (restart kernel → Run all)
- [ ] Results table: accuracy, macro-F1, seconds per 100 texts, training time, labelled examples used
- [ ] Confusion matrix for each approach
- [ ] Label-wording experiment with at least 3 variants
- [ ] Learning-curve numbers (100 / 500 / 2,000 examples)
- [ ] 5 analysed errors, one line each
- [ ] Recommendation ≤ 150 words

---

## Stretch goals (pick any)

- Wrap your best model in a **Gradio** app (~10 lines) and demo it live.
- Publish the app on **Hugging Face Spaces**, or push your fine-tuned model to the Hub with `push_to_hub()`.
- Use your **own texts**: 50–100 labelled examples from your domain (support tickets, reviews), and rerun approach B.
- Try a **multilingual** embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) on Czech or German headlines.

## Troubleshooting

| Problem | Fix |
|---|---|
| First run is slow | Models download once to `~/.cache/huggingface`; later runs are instant |
| Out of memory | Lower `batch_size` to 8, or shorten texts with `max_length=64` |
| Zero-shot takes too long | Test on 100 texts first, then scale to 500 |
| Corporate proxy blocks downloads | Set `HTTPS_PROXY`, or use Colab/Kaggle |
| "Some weights not initialized" when fine-tuning | Expected: the classification head is new and gets trained |
| `ImportError: ... requires accelerate` when fine-tuning | `pip install accelerate` |

---

*Need more detail on how to do a step? → **Level 3**.*
