# Level 1 — Your First Hugging Face NLP Pipeline: The Challenge

> **How to use the four levels**
> - **Level 1 (this file):** the challenge in a nutshell. Try to solve it on your own.
> - **Level 2:** a structured step-by-step plan, if you want guidance on *what* to do.
> - **Level 3:** a very detailed walkthrough for beginners, covering *how* to do each step.
> - **Level 4:** a complete reference solution in Python.
>
> Stuck? Go one level deeper, get unblocked, and come back.

---

## The scenario

Your team receives hundreds of short news texts every day and wants them **routed automatically** to one of four topics:

**World · Sports · Business · Sci/Tech**

Your job is to build a small **"News Sorter"** and find out which Hugging Face approach gives the **best accuracy for the least effort**.

## What you will do

Using the public **AG News** dataset (`fancyzhx/ag_news` on the Hugging Face Hub), compare three ways of classifying text on the same test set:

1. **Zero-shot classification:** use a pre-trained model as it is, with no training.
2. **Embeddings + a simple classifier:** turn texts into vectors and train a tiny model on top.
3. **Fine-tuning a small transformer (optional):** adapt a pre-trained model to the task.

Then measure the approaches fairly (accuracy, macro-F1, confusion matrix, speed), look at where they fail, and recommend one.

## Constraints

- Everything runs on an ordinary **CPU-only laptop**, with no GPU.
- Use **small subsets** of the data, for example 2,000 training and 500 test examples.
- Expected effort: about **4–6 hours**.
- Work in **one Jupyter notebook** (Google Colab or Kaggle also work).

## What you hand in

- A notebook that runs top to bottom without errors.
- A results table comparing the approaches.
- A short recommendation (≤ 150 words): which approach would you deploy, and why?
- A 5-minute walkthrough at the next club session.

## What you will learn

Loading datasets from the Hub, using the `pipeline` API, working with sentence embeddings, evaluating models honestly, and (optionally) fine-tuning with the `Trainer` API.

**Useful reading:** [Hugging Face LLM Course](https://huggingface.co/learn/llm-course), chapters 1–3.

---

*Not sure where to start? → Open **Level 2**.*
