"""Step 1 - Load and explore the data."""
import pandas as pd
from common import load_data, OUT

train, test, labels, y_train, y_test = load_data()
print("Labels:", labels)
print("Subset sizes:", len(train), len(test))

# 5 examples per class
df_train = train.to_pandas()
df_train["label_name"] = df_train["label"].map(lambda i: labels[i])
for name in labels:
    print(f"\n===== {name} =====")
    for t in df_train.loc[df_train["label_name"] == name, "text"].head(5):
        print("-", t)

# Class balance of both subsets
balance = pd.DataFrame({
    "train": df_train["label_name"].value_counts(),
    "test": pd.Series(y_test).map(lambda i: labels[i]).value_counts(),
}).loc[labels]
print("\nClass balance:\n" + balance.to_string())
balance.to_csv(OUT / "class_balance.csv")

print("\n-> Now write your hypothesis (which classes will be confused?) into REPORT.md")
