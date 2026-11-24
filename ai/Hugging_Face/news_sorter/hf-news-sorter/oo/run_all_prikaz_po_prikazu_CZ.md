# `run_all.py --with-finetune` – příkaz po příkazu

Dokument prochází běh

```powershell
python run_all.py --with-finetune
```

v pořadí, v jakém se příkazy skutečně provádějí. U každého volání se sestupuje do souborů v [news_sorter/](news_sorter/) a [news_sorter/classifiers/](news_sorter/classifiers/). Odkazy ve tvaru `soubor:řádek` vedou přímo na kód.

Výchozí hodnoty (z [config.py](news_sorter/config.py)), se kterými běh počítá: `n_train=2000`, `n_test=500`, `n_wording=200`, `seed=42`, výstupní složka `oo/outputs`.

---

## 1. Import modulů (ještě před `main()`)

### 1.1 [run_all.py:17-22](run_all.py:17)

| Příkaz | Co dělá |
|---|---|
| `import argparse` | Standardní knihovna pro čtení přepínačů z příkazové řádky. |
| `from pathlib import Path` | Třída pro práci s cestami k souborům. |
| `from news_sorter import Config, NewsSorterPipeline` | Spustí [news_sorter/\_\_init\_\_.py](news_sorter/__init__.py) – viz 1.2. |
| `from news_sorter.classifiers import EmbeddingClassifier, NewsClassifier, ZeroShotClassifier` | Spustí [classifiers/\_\_init\_\_.py](news_sorter/classifiers/__init__.py) – viz 1.3. |
| `from news_sorter.config import DEFAULT_OUTPUT_DIR` | Převezme výchozí výstupní složku (už je spočtená, modul je načtený). |

### 1.2 [news_sorter/\_\_init\_\_.py:38-42](news_sorter/__init__.py:38)

| Příkaz | Co dělá |
|---|---|
| `from news_sorter.config import Config` | Načte [config.py](news_sorter/config.py): `DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"` spočte absolutní cestu `oo/outputs` (nezávisle na tom, odkud se skript spouští); `@dataclass(frozen=True) class Config` definuje neměnnou konfiguraci. |
| `from news_sorter.data import NewsData` | Načte [data.py](news_sorter/data.py) – importuje `numpy` a `datasets` (Hugging Face) a definuje třídu `NewsData`. |
| `from news_sorter.pipeline import NewsSorterPipeline` | Načte [pipeline.py](news_sorter/pipeline.py), který dále importuje: `classifiers.base` (abstraktní třída `NewsClassifier`), `evaluation` (`matplotlib.use("Agg")` – grafy jen do souborů, bez okna; `sklearn.metrics`, `pandas`), `exploration`, `results` a `timing` (třída `Timer`). |
| `__all__ = [...]` | Určuje, co balíček veřejně nabízí. |

### 1.3 [classifiers/\_\_init\_\_.py:10-14](news_sorter/classifiers/__init__.py:10)

| Příkaz | Co dělá |
|---|---|
| `from ...base import NewsClassifier` | Už načteno (bod 1.2). |
| `from ...embedding import EmbeddingClassifier` | Importuje `joblib`, `matplotlib`, `sentence_transformers.SentenceTransformer`, `sklearn.LogisticRegression`. |
| `from ...zero_shot import ZeroShotClassifier` | Importuje `transformers.pipeline`; definuje `Prompt`, slovník `LABEL_VARIANTS` (4 varianty názvů tříd) a `TEMPLATES` (2 šablony hypotézy). |

`FineTunedClassifier` se zde záměrně **neimportuje** – tahá těžký tréninkový stack; načte se až v bodě 4.

### 1.4 [run_all.py:66-67](run_all.py:66)

`if __name__ == "__main__": main()` – soubor je spuštěn přímo, takže se zavolá `main()`.

---

## 2. `main()` → `parse_args()` – [run_all.py:25-39](run_all.py:25)

| Příkaz | Co dělá |
|---|---|
| `args = parse_args()` | Zavolá funkci pro čtení přepínačů. |
| `defaults = Config()` | Vytvoří konfiguraci s výchozími hodnotami jen kvůli převzetí defaultů. |
| `parser = argparse.ArgumentParser(...)` | Vytvoří parser s popisem programu. |
| `add_argument("--with-finetune", action="store_true")` | Přepínač bez hodnoty; když je uveden, je `True`. **V našem případě `True`.** |
| `add_argument("--n-train", ...)` | Počet trénovacích příkladů, default 2000. |
| `add_argument("--n-test", ...)` | Počet testovacích příkladů, default 500. |
| `add_argument("--n-wording", ...)` | Počet testovacích textů pro experiment se zněním štítků, default 200. |
| `add_argument("--output-dir", type=Path, ...)` | Výstupní složka, default `DEFAULT_OUTPUT_DIR`. |
| `return parser.parse_args()` | Přečte `sys.argv` a vrátí objekt `Namespace` (`args.with_finetune=True`, ostatní výchozí). |

## 3. `main()` – sestavení konfigurace – [run_all.py:55-60](run_all.py:55)

`config = Config(n_train=..., n_test=..., n_wording=..., output_dir=args.output_dir.resolve())` – vytvoří finální neměnnou konfiguraci; `.resolve()` převede cestu na absolutní. Ostatní hodnoty (modely, seed, epochy) zůstanou výchozí.

## 4. `main()` → `build_classifiers(config, True)` – [run_all.py:42-50](run_all.py:42)

| Příkaz | Co dělá |
|---|---|
| `ZeroShotClassifier(config)` | Konstruktor [zero_shot.py:83-89](news_sorter/classifiers/zero_shot.py:83): `super().__init__(config)` ([base.py:46-49](news_sorter/classifiers/base.py:46)) uloží config a nastaví `train_time_s=0.0`, `n_labelled=0`; `self.batch_size=16`; `pipeline("zero-shot-classification", model="cross-encoder/nli-distilroberta-base")` stáhne (nebo načte z cache) NLI model; `self.prompt = Prompt("2 descriptive", ...)` nastaví výchozí prompt. |
| `EmbeddingClassifier(config)` | Konstruktor [embedding.py:42-49](news_sorter/classifiers/embedding.py:42): `super().__init__`; `batch_size=64`; `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")` načte enkodér (~90 MB); `_new_classifier()` vytvoří netrénovanou `LogisticRegression(max_iter=1000)`; `_X_train=None`. |
| `if with_finetune:` | Podmínka je splněna. |
| `from news_sorter.classifiers.fine_tuned import FineTunedClassifier` | Až teď se importuje [fine_tuned.py](news_sorter/classifiers/fine_tuned.py) a s ním `AutoModelForSequenceClassification`, `AutoTokenizer`, `DataCollatorWithPadding`, `Trainer`, `TrainingArguments`. |
| `classifiers.append(FineTunedClassifier(config))` | Konstruktor [fine_tuned.py:33-38](news_sorter/classifiers/fine_tuned.py:33): `super().__init__`; `AutoTokenizer.from_pretrained("distilbert-base-uncased")` načte tokenizér; `_trainer=None`. |
| `return classifiers` | Seznam `[A, B, C]` v pořadí, v jakém poběží. |

## 5. `main()` → `NewsSorterPipeline(config, classifiers).run()` – [run_all.py:61](run_all.py:61)

Konstruktor [pipeline.py:24-26](news_sorter/pipeline.py:24) jen uloží config a seznam klasifikátorů. Pipeline nezná konkrétní přístupy, pracuje jen s rozhraním `NewsClassifier`.

---

## 6. `run()` – příprava – [pipeline.py:34-38](news_sorter/pipeline.py:34)

| Příkaz | Co dělá |
|---|---|
| `output_dir = self.config.output_dir` | Zkratka na výstupní složku. |
| `output_dir.mkdir(parents=True, exist_ok=True)` | Vytvoří složku (i rodičovské), pokud neexistuje. |
| `self._remove_stale_predictions()` | [pipeline.py:62-69](news_sorter/pipeline.py:62): `glob("preds_*.npy")` + `path.unlink()` smaže predikce z dřívějších běhů. |
| `with Timer() as total:` | [timing.py](news_sorter/timing.py): `__enter__` uloží `time.perf_counter()`; `__exit__` na konci bloku spočte `seconds`. Měří celý běh. |

## 7. Krok 1 – načtení a průzkum dat – [pipeline.py:40-42](news_sorter/pipeline.py:40)

| Příkaz | Co dělá |
|---|---|
| `self._headline("Step 1 - ...")` | [pipeline.py:71-75](news_sorter/pipeline.py:71): vytiskne nadpis mezi řádky `=`; `flush=True` ho vypíše hned. |
| `data = NewsData.load(self.config)` | Viz 7.1. |
| `DataExplorer(data, output_dir).report()` | Viz 7.2. |

### 7.1 `NewsData.load` – [data.py:35-52](news_sorter/data.py:35)

| Příkaz | Co dělá |
|---|---|
| `load_dataset("fancyzhx/ag_news")` | Stáhne/načte z cache AG News (120 000 train, 7 600 test). |
| `dataset["train"].shuffle(seed=42).select(range(2000))` | Reprodukovatelně zamíchá a vezme prvních 2000 trénovacích textů. |
| `dataset["test"].shuffle(seed=42).select(range(500))` | Totéž pro 500 testovacích textů. |
| `labels=train.features["label"].names` | Názvy tříd `['World', 'Sports', 'Business', 'Sci/Tech']` (index = id třídy). |
| `y_train=np.array(train["label"])`, `y_test=...` | Správné třídy jako NumPy pole. |
| `return cls(...)` | Vytvoří neměnný objekt `NewsData`, sdílený všemi přístupy. |

Pomocné vlastnosti: `train_texts` / `test_texts` vrací texty jako Python seznam; `label_names(ids)` převádí id na názvy.

### 7.2 `DataExplorer.report` – [exploration.py:22-32](news_sorter/exploration.py:22)

| Příkaz | Co dělá |
|---|---|
| `print("Labels:", ...)`, `print("Subset sizes:", ...)` | Vypíše třídy a velikosti (2000, 500). |
| `self._print_examples(5)` | [exploration.py:49-55](news_sorter/exploration.py:49): sestaví `DataFrame` (text, label) a pro každou třídu vytiskne 5 ukázkových textů. |
| `balance = self.class_balance()` | [exploration.py:34-47](news_sorter/exploration.py:34): `value_counts()` spočte počty příkladů na třídu v train i test; `.loc[data.labels]` je seřadí v oficiálním pořadí. |
| `print(...)`, `balance.to_csv(".../class_balance.csv")` | Vypíše tabulku a uloží ji do **`class_balance.csv`**. |
| `print("-> Now write your hypothesis ...")` | Výzva zapsat hypotézu do REPORT.md. |

## 8. Sdílené nástroje – [pipeline.py:45-46](news_sorter/pipeline.py:45)

| Příkaz | Co dělá |
|---|---|
| `evaluator = Evaluator(data, output_dir)` | [evaluation.py:62-64](news_sorter/evaluation.py:62): jeden hodnotitel pro všechny přístupy (stejné měření = férové srovnání). |
| `report = ResultsReport(output_dir)` | [results.py:25-27](news_sorter/results.py:25): prázdný seznam `_results` pro výsledky. |

---

## 9. Smyčka přes klasifikátory – [pipeline.py:47-53](news_sorter/pipeline.py:47)

Pro každý klasifikátor (A, B, C) se provede stejných 6 příkazů:

```python
self._headline(classifier.name)                               # nadpis
classifier.fit(data)                                          # příprava / trénink
preds, sec_per_100 = classifier.timed_predict(data.test_texts)  # predikce + měření
report.add(evaluator.evaluate(classifier, preds, sec_per_100))  # metriky + soubory
classifier.run_experiments(data)                              # doplňková analýza
classifier.save()                                             # uložení modelu
```

Společné metody ze základní třídy [base.py](news_sorter/classifiers/base.py):

- **`timed_predict(texts)`** ([base.py:87-100](news_sorter/classifiers/base.py:87)): v bloku `with Timer()` zavolá `self.predict(texts)` (polymorfně – každá podtřída má vlastní) a vrátí `(preds, timer.seconds / len(texts) * 100)`, tj. sekundy na 100 textů.
- **`run_experiments`** a **`save`** – výchozí implementace nedělají nic; přepisují je jen přístupy, které mají co dělat.

Společné hodnocení **`Evaluator.evaluate(...)`** ([evaluation.py:66-103](news_sorter/evaluation.py:66)), provede se pro A, B i C:

| Příkaz | Co dělá |
|---|---|
| `accuracy_score(y_true, preds)` | Podíl správně zařazených textů. |
| `f1_score(..., average="macro")` | F1 spočtené pro každou třídu zvlášť a zprůměrované. |
| `ApproachResult(...)` | Neměnný záznam: klíč, název, accuracy, macro-F1, rychlost, doba tréninku, počet označených příkladů. |
| `print(...)` ×2 | Vypíše název a souhrnná čísla. |
| `classification_report(...)` | Vytiskne precision / recall / F1 pro každou třídu. |
| `np.save(".../preds_X.npy", preds)` | Uloží surové predikce → **`preds_A/B/C.npy`**. |
| `self._save_confusion_matrix(...)` | [evaluation.py:123-144](news_sorter/evaluation.py:123): `plt.subplots`, `ConfusionMatrixDisplay.from_predictions` (řádky = skutečná třída, sloupce = predikce), `savefig` → **`confusion_X.png`**, `plt.close`. |
| `self._save_errors(...)` | [evaluation.py:146-153](news_sorter/evaluation.py:146): `errors_table()` (`np.where(preds != y_test)` najde chyby a sestaví `DataFrame` idx/true/pred/text) → `to_csv` **`errors_X.csv`**; vytiskne prvních 8 chyb. |
| `return result` | Výsledek jde do `report.add()` ([results.py:29-31](news_sorter/results.py:29)) – připojí se do seznamu. |

### 9.1 Přístup A – `ZeroShotClassifier` ([zero_shot.py](news_sorter/classifiers/zero_shot.py))

**`fit(data)`** ([zero_shot.py:91-110](news_sorter/classifiers/zero_shot.py:91)) – nic netrénuje, jen vybírá nejlepší prompt:

| Příkaz | Co dělá |
|---|---|
| `self._show_example(data)` | [zero_shot.py:160-166](news_sorter/classifiers/zero_shot.py:160): pustí pipeline na první testovací text a vytiskne text, správnou třídu a surový výstup pipeline. |
| `wording = self.wording_experiment(data)` | [zero_shot.py:116-144](news_sorter/classifiers/zero_shot.py:116): vezme prvních 200 testovacích textů; pro každou z 4 variant štítků × 2 šablon (= 8 kombinací) vytvoří `Prompt`, v `Timer` zavolá `_predict_with`, spočte accuracy a rychlost, řádek vytiskne. Nakonec `sort_values("Accuracy", kind="stable")` – nejlepší první, při shodě vyhrává dřívější kombinace. |
| `wording.to_csv(".../zero_shot_wording.csv")` | Uloží výsledky experimentu → **`zero_shot_wording.csv`**. |
| `best = wording.iloc[0]` + `self.prompt = Prompt(...)` | Nastaví vítěznou kombinaci štítků a šablony jako aktivní prompt. |
| `train_time_s = 0.0`, `n_labelled = 0` | Zero-shot nepotřebuje trénink ani označená data. (Pozn.: prompt se vybírá na testovacích datech, skóre A je proto mírně optimistické.) |

**`_predict_with(texts, prompt)`** ([zero_shot.py:146-158](news_sorter/classifiers/zero_shot.py:146)) – jádro predikce A:

| Příkaz | Co dělá |
|---|---|
| `kwargs = {"candidate_labels": labels, "batch_size": 16}` | Parametry pipeline. |
| `if prompt.template: kwargs["hypothesis_template"] = ...` | Vlastní šablonu přidá jen pokud není výchozí (`"This example is {}."`). |
| `outputs = self._pipeline(list(texts), **kwargs)` | Pro každý text a každý ze 4 štítků se NLI model ptá, zda text „implikuje“ hypotézu → 4 průchody modelem na text. |
| `np.array([labels.index(output["labels"][0]) ...])` | Vítězný štítek (první v seřazeném výstupu) převede na id třídy podle pozice v seznamu – proto musí štítky dodržet pořadí datasetu. |

**`timed_predict`** → `predict(texts)` ([zero_shot.py:112-114](news_sorter/classifiers/zero_shot.py:112)) = `_predict_with(texts, self.prompt)` na všech 500 testovacích textech.

**`evaluate`** → `preds_A.npy`, `confusion_A.png`, `errors_A.csv`.
**`run_experiments`**, **`save`** → nic (výchozí z `NewsClassifier`).

### 9.2 Přístup B – `EmbeddingClassifier` ([embedding.py](news_sorter/classifiers/embedding.py))

**`fit(data)`** ([embedding.py:51-63](news_sorter/classifiers/embedding.py:51)):

| Příkaz | Co dělá |
|---|---|
| `with Timer() as encode_timer: self._X_train = self.encode(train_texts, show_progress_bar=True)` | [embedding.py:69-71](news_sorter/classifiers/embedding.py:69): `SentenceTransformer.encode` převede 2000 textů na vektory 384 čísel (po dávkách 64, s progress barem). |
| `print("Train embedding shape:", ...)` | Vypíše `(2000, 384)`. |
| `with Timer() as fit_timer: self._classifier.fit(X_train, y_train)` | Natrénuje logistickou regresi na vektorech (enkodér se netrénuje). |
| `train_time_s = encode + fit`, `n_labelled = 2000` | Do doby tréninku se počítá i kódování trénovacích textů. |

**`timed_predict`** → `predict(texts)` ([embedding.py:65-67](news_sorter/classifiers/embedding.py:65)): `self._classifier.predict(self.encode(texts))` – zakóduje 500 testovacích textů a klasifikuje je (měří se včetně kódování).

**`evaluate`** → `preds_B.npy`, `confusion_B.png`, `errors_B.csv`.

**`run_experiments(data)`** – křivka učení ([embedding.py:73-101](news_sorter/classifiers/embedding.py:73)):

| Příkaz | Co dělá |
|---|---|
| `if self._X_train is None: raise ...` | Kontrola, že `fit()` proběhl. |
| `X_test = self.encode(data.test_texts)` | Zakóduje testovací texty (jednou pro všechny body křivky). |
| `for n in self._curve_sizes(2000)` | [embedding.py:122-128](news_sorter/classifiers/embedding.py:122): velikosti `{100, 500} ∪ {2000}` → `[100, 500, 2000]`. |
| `self._new_classifier().fit(X_train[:n], y_train[:n])` | Pro každé `n` natrénuje novou LR na prvních `n` (už zakódovaných) příkladech. |
| `classifier.predict(X_test)` + `accuracy_score`, `f1_score` | Změří kvalitu pro danou velikost a uloží řádek. |
| `curve.to_csv(".../learning_curve.csv")` | → **`learning_curve.csv`**. |
| `self._plot_learning_curve(curve)` | [embedding.py:130-142](news_sorter/classifiers/embedding.py:130): čárový graf accuracy vs. počet příkladů (logaritmická osa x) → **`learning_curve.png`**. |

**`save()`** ([embedding.py:103-111](news_sorter/classifiers/embedding.py:103)): `joblib.dump(self._classifier, ...)` uloží jen malou logistickou regresi → **`news_sorter_lr.joblib`** (enkodér lze kdykoli stáhnout z Hubu podle názvu).

### 9.3 Přístup C – `FineTunedClassifier` ([fine_tuned.py](news_sorter/classifiers/fine_tuned.py)) – běží díky `--with-finetune`

**`fit(data)`** ([fine_tuned.py:40-80](news_sorter/classifiers/fine_tuned.py:40)):

| Příkaz | Co dělá |
|---|---|
| `AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4, id2label=..., label2id=...)` | Načte předtrénovaný DistilBERT (~260 MB) a nasadí na něj novou klasifikační hlavu se 4 výstupy (náhodné váhy → očekávané varování „Some weights ... were not initialized“). Názvy tříd se uloží do modelu. |
| `TrainingArguments(...)` | Nastavení tréninku: cesta `outputs/distilbert-checkpoints`, 1 epocha, batch 16 (trénink) / 32 (predikce), `learning_rate=5e-5`, `weight_decay=0.01`, log každých 25 kroků, `save_strategy="no"` (bez mezilehlých checkpointů), `report_to="none"`, `seed=42`. |
| `self._tokenize(data.train)` | [fine_tuned.py:101-111](news_sorter/classifiers/fine_tuned.py:101): `dataset.map(tokenize, batched=True)` přidá sloupce `input_ids` a `attention_mask`; texty delší než 64 tokenů ořízne. |
| `DataCollatorWithPadding(self._tokenizer)` | Doplňuje (padding) každou dávku jen na délku jejího nejdelšího textu. |
| `Trainer(model=..., args=..., train_dataset=..., data_collator=...)` | Sestaví tréninkový objekt. |
| `with Timer() as timer: self._trainer.train()` | Fine-tuning: aktualizuje **všechny** váhy modelu; 2000 / 16 = 125 kroků, na CPU cca 15–30 min. |
| `print(f"Training took ... min")` | Vypíše dobu tréninku. |
| `train_time_s = timer.seconds`, `n_labelled = 2000` | Uloží náklady přístupu. |

**`timed_predict`** → `predict(texts)` ([fine_tuned.py:82-90](news_sorter/classifiers/fine_tuned.py:82)):

| Příkaz | Co dělá |
|---|---|
| `if self._trainer is None: raise ...` | Kontrola, že model je natrénovaný. |
| `Dataset.from_dict({"text": list(texts)})` + `self._tokenize(...)` | Z 500 testovacích textů udělá tokenizovaný `Dataset`. |
| `self._trainer.predict(dataset).predictions` | Model vrátí pro každý text 4 skóre (logity). |
| `logits.argmax(axis=-1)` | Predikcí je třída s nejvyšším skóre. |

**`evaluate`** → `preds_C.npy`, `confusion_C.png`, `errors_C.csv`.
**`run_experiments`** → nic.
**`save()`** ([fine_tuned.py:92-99](news_sorter/classifiers/fine_tuned.py:92)): `self._trainer.save_model(...)` + `self._tokenizer.save_pretrained(...)` uloží model i tokenizér do složky **`distilbert-agnews/`** (připraveno pro opětovné použití nebo `push_to_hub`).

---

## 10. Krok 6 – finální srovnání – [pipeline.py:56-57](news_sorter/pipeline.py:56)

| Příkaz | Co dělá |
|---|---|
| `self._headline("Final comparison")` | Nadpis. |
| `report.save()` | [results.py:39-53](news_sorter/results.py:39) – viz níže. |
| `if not self._results: raise RuntimeError(...)` | Ochrana proti prázdnému reportu. |
| `table = self.to_frame()` | [results.py:33-37](news_sorter/results.py:33): pro každý výsledek `to_row()` ([evaluation.py:44-52](news_sorter/evaluation.py:44)) – zaokrouhlené sloupce Accuracy, Macro-F1, Sec / 100 texts, Training time (s), Labelled examples; `DataFrame` seřazený podle názvu (A, B, C). |
| `print(table.to_string())` | Vytiskne srovnávací tabulku. |
| `json_path.write_text(json.dumps(...))` | → **`results.json`**. |
| `table.to_csv(...)` | → **`results_table.csv`**. |
| `self._plot_comparison(table)` | [results.py:55-65](news_sorter/results.py:55): sloupcový graf Accuracy a Macro-F1 pro každý přístup, osa y celá 0–1 (aby malé rozdíly nevypadaly dramaticky) → **`comparison.png`**. |
| `print(...)` ×2 | Potvrzení uložení a výzva zkopírovat tabulku do REPORT.md a napsat doporučení. |

## 11. Konec – [pipeline.py:59-60](news_sorter/pipeline.py:59)

Blok `with Timer() as total` skončí (`__exit__` spočte čas), vytiskne se `All steps finished in X min.` a `run()` vrátí `report`. `main()` návratovou hodnotu nepoužije a program skončí.

---

## Přehled vytvořených souborů v `outputs/`

| Soubor | Vzniká v kroku |
|---|---|
| `class_balance.csv` | 7.2 – průzkum dat |
| `zero_shot_wording.csv` | 9.1 – A `fit()` |
| `preds_A.npy`, `confusion_A.png`, `errors_A.csv` | 9.1 – hodnocení A |
| `preds_B.npy`, `confusion_B.png`, `errors_B.csv` | 9.2 – hodnocení B |
| `learning_curve.csv`, `learning_curve.png` | 9.2 – B `run_experiments()` |
| `news_sorter_lr.joblib` | 9.2 – B `save()` |
| `preds_C.npy`, `confusion_C.png`, `errors_C.csv` | 9.3 – hodnocení C |
| `distilbert-agnews/` | 9.3 – C `save()` |
| `distilbert-checkpoints/` | 9.3 – pracovní složka `Trainer` (checkpointy se neukládají) |
| `results.json`, `results_table.csv`, `comparison.png` | 10 – finální srovnání |

## Kde se tráví čas

| Část | Počet průchodů modelem (přibližně) |
|---|---|
| A – experiment se zněním štítků | 8 kombinací × 200 textů × 4 štítky = 6 400 |
| A – finální predikce | 500 textů × 4 štítky = 2 000 |
| B – kódování | 2 000 (train) + 500 (predikce) + 500 (křivka učení) |
| C – fine-tuning | 125 tréninkových kroků po 16 textech (nejpomalejší část) + 500 textů predikce |
