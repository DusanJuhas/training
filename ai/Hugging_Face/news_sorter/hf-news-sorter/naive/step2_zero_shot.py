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
