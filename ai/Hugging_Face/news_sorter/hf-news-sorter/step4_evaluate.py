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
