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
