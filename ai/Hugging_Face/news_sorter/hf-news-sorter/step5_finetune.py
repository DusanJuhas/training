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

# 2) Model with a new 4-class head ("Some weights ... not initialized" warning is expected)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label={i: l for i, l in enumerate(labels)},
    label2id={l: i for i, l in enumerate(labels)},
)

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
