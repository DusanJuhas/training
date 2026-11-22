# Co dělá `run_all.py` – krok za krokem

Tento dokument sleduje `run_all.py` od chvíle, kdy napíšete příkaz, až po zápis posledního souboru. Všude, kde skript volá balíček `news_sorter`, jde dokument i do něj, protože samotný `run_all.py` má jen asi 60 řádků: načte příkazovou řádku, vytvoří objekty a spustí pipeline. Veškerá skutečná práce probíhá uvnitř balíčku.

**Cíl experimentu:** zařadit krátké zprávy z datasetu **AG News** do čtyř témat (**World, Sports, Business, Sci/Tech**) a porovnat tři přístupy Hugging Face podle kvality (accuracy, macro‑F1) a nákladů (rychlost, doba trénování, počet označených příkladů):

| Klíč | Přístup | Trénování? |
|---|---|---|
| **A** | Zero-shot klasifikace (NLI model `cross-encoder/nli-distilroberta-base`) | žádné |
| **B** | Větné embeddingy (`all-MiniLM-L6-v2`) + logistická regrese | ~sekundy |
| **C** | Fine-tuned DistilBERT (`distilbert-base-uncased`), volitelné | 15–30 min na CPU |

---

## 0. Jak se spouští

```bash
python run_all.py                    # přístupy A + B (~15 min na CPU)
python run_all.py --with-finetune    # A + B + C (přidá ~15–30 min)
python run_all.py --n-train 100 --n-test 40 --n-wording 10 --output-dir outputs_smoke   # rychlý smoke test
```

Když Python spustí soubor přímo, provedou se poslední řádky:

```python
if __name__ == "__main__":
    main()
```

Tato podmínka zajistí, že experiment se spustí **jen** tehdy, když se soubor spouští přímo (`python run_all.py`), ne když ho importuje jiný kód.

---

## 1. Importy (proběhnou ještě před spuštěním `main()`)

```python
import argparse
from pathlib import Path
from news_sorter import Config, NewsSorterPipeline
from news_sorter.classifiers import EmbeddingClassifier, NewsClassifier, ZeroShotClassifier
from news_sorter.config import DEFAULT_OUTPUT_DIR
```

Co se děje na pozadí:

1. Spustí se **`news_sorter/__init__.py`**, který importuje `Config`, `NewsData` a `NewsSorterPipeline`. Přes tyto importy se načtou moduly `config`, `data`, `pipeline`, `evaluation`, `exploration`, `results` a `timing` i knihovny, které potřebují (`datasets`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`).
2. **`news_sorter/config.py`** spočítá `DEFAULT_OUTPUT_DIR` = složku `outputs` vedle balíčku (`.../oo/outputs`). Cesta se odvozuje od umístění samotného souboru, takže skript zapisuje na stejné místo bez ohledu na to, z jaké složky ho spustíte.
3. **`news_sorter/classifiers/__init__.py`** importuje `NewsClassifier` (abstraktní základní třídu), `ZeroShotClassifier` (načte `transformers`) a `EmbeddingClassifier` (načte `sentence_transformers`).
   - `FineTunedClassifier` se tu **záměrně neimportuje**. Táhne s sebou těžký tréninkový stack (`Trainer`, `accelerate`), který výchozí běh nepotřebuje.
4. Každý modul, který kreslí grafy, volá `matplotlib.use("Agg")`, takže se grafy zapisují rovnou do PNG souborů a není potřeba žádné okno ani displej.

---

## 2. `main()` → `parse_args()`: čtení příkazové řádky

```python
defaults = Config()
parser = argparse.ArgumentParser(...)
```

1. Vytvoří se výchozí objekt `Config()`, jen aby se z něj přečetly výchozí hodnoty. Výchozí hodnoty tak žijí na jediném místě (`config.py`) a nápověda je zobrazuje.
2. Definuje se pět voleb:

| Volba | Typ | Výchozí | Význam |
|---|---|---|---|
| `--with-finetune` | přepínač | vypnuto | spustí i přístup C |
| `--n-train` | int | 2000 | počet trénovacích příkladů |
| `--n-test` | int | 500 | počet testovacích příkladů |
| `--n-wording` | int | 200 | testovací texty pro experiment s formulací štítků u zero-shot |
| `--output-dir` | cesta | `.../oo/outputs` | složka pro všechny vytvořené soubory |

3. `parser.parse_args()` přečte `sys.argv` a vrátí objekt `Namespace` (`args.n_train`, `args.with_finetune`, …). Při chybné volbě (např. `--n-train abc`) argparse vypíše chybu a skript tady skončí.

---

## 3. `main()`: sestavení konfigurace

```python
config = Config(
    n_train=args.n_train,
    n_test=args.n_test,
    n_wording=args.n_wording,
    output_dir=args.output_dir.resolve(),
)
```

- `Config` je **zmrazená (frozen) dataclass**, tedy po vytvoření jen pro čtení. Žádná část kódu nemůže změnit nastavení uprostřed běhu.
- `.resolve()` převede výstupní cestu na absolutní.
- Vše, co nebylo zadáno na příkazové řádce, si ponechá výchozí hodnotu:

| Nastavení | Hodnota |
|---|---|
| `seed` | 42 (stejné zamíchání → každý přístup vidí stejné texty) |
| `dataset_name` | `fancyzhx/ag_news` |
| `zero_shot_model` | `cross-encoder/nli-distilroberta-base` |
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` (~90 MB) |
| `learning_curve_sizes` | (100, 500, 2000) |
| `finetune_model` | `distilbert-base-uncased` (~260 MB) |
| `finetune_max_length` | 64 tokenů |
| `finetune_epochs` | 1 |

---

## 4. `main()` → `build_classifiers()`: vytvoření klasifikátorů

```python
classifiers = [ZeroShotClassifier(config), EmbeddingClassifier(config)]
if with_finetune:
    from news_sorter.classifiers.fine_tuned import FineTunedClassifier
    classifiers.append(FineTunedClassifier(config))
```

> ⚠️ **Důležitý detail:** už samotné vytvoření objektů **načítá modely**. Děje se to *před* spuštěním pipeline, takže při prvním běhu uvidíte stahování modelů ještě před nadpisem „Step 1“. Hugging Face si pak modely ukládá do cache, takže další běhy stahování přeskočí.

Co dělá každý konstruktor:

**4a. `ZeroShotClassifier(config)` (přístup A)**
1. Základní třída `NewsClassifier.__init__` uloží `config` a nastaví `train_time_s = 0.0` a `n_labelled = 0`.
2. `batch_size = 16`.
3. `pipeline("zero-shot-classification", model="cross-encoder/nli-distilroberta-base")` stáhne nebo načte NLI model a jeho tokenizér.
4. Nastaví se počáteční prompt `"2 descriptive"` (`world politics`, `sports`, `business`, `science and technology`), aby `predict()` fungoval i bez `fit()`.

**4b. `EmbeddingClassifier(config)` (přístup B)**
1. Inicializace základní třídy jako výše.
2. `batch_size = 64`.
3. `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")` stáhne nebo načte enkodér.
4. Vytvoří se nová, nenatrénovaná `LogisticRegression(max_iter=1000)`.
5. `_X_train = None`. Sem se později uloží trénovací embeddingy.

**4c. `FineTunedClassifier(config)` (přístup C, jen s `--with-finetune`)**
1. Teprve teď se importuje modul i s `Trainer`, `TrainingArguments` a dalšími.
2. Inicializace základní třídy.
3. `AutoTokenizer.from_pretrained("distilbert-base-uncased")` načte tokenizér. Samotný model se vytvoří až později, ve `fit()`.
4. `_trainer = None`.

Výsledkem je seznam objektů klasifikátorů, které všechny sdílejí stejné rozhraní (`NewsClassifier`).

---

## 5. `main()`: vytvoření pipeline a volání `run()`

```python
NewsSorterPipeline(config, build_classifiers(config, args.with_finetune)).run()
```

`NewsSorterPipeline.__init__` jen uloží konfiguraci a seznam klasifikátorů. Pipeline **neví, jaké přístupy existují**. Dostane hotové objekty zvenku (*dependency injection*) a komunikuje s nimi jen přes rozhraní `NewsClassifier` (*polymorfismus*).

Vše další se odehrává v `NewsSorterPipeline.run()`.

---

## 6. `run()` – příprava

1. `output_dir.mkdir(parents=True, exist_ok=True)` vytvoří výstupní složku (i chybějící nadřazené složky), pokud ještě neexistuje.
2. `_remove_stale_predictions()` smaže všechny soubory `preds_*.npy` z dřívějších běhů. Bez toho by například `preds_C.npy` ze starého běhu s fine-tuningem zůstal vedle čerstvých souborů A/B a vypadal by jako aktuální.
   - Poznámka: mažou se jen soubory `preds_*.npy`. Ostatní soubory přístupu, který tentokrát neběží (např. `confusion_C.png`, `errors_C.csv`), ve složce zůstanou.
3. `with Timer() as total:` spustí stopky pro celý experiment. `Timer` používá `time.perf_counter()`, nejpřesnější hodiny pro měření doby trvání, a čas zaznamená i v případě chyby.

---

## 7. Krok 1 – načtení a průzkum dat

Vypíše se nadpis (`>>> Step 1 - load and explore the data`). `flush=True` zajistí, že se objeví okamžitě, i když je výstup přesměrovaný do souboru.

### 7a. `NewsData.load(config)`
1. `load_dataset("fancyzhx/ag_news")` stáhne AG News z Hugging Face Hubu, nebo ho načte z lokální cache. Celý dataset má 120 000 trénovacích a 7 600 testovacích textů.
2. `dataset["train"].shuffle(seed=42).select(range(2000))` dá reprodukovatelný náhodný vzorek 2 000 trénovacích textů.
3. `dataset["test"].shuffle(seed=42).select(range(500))` dá reprodukovatelný náhodný vzorek 500 testovacích textů.
4. Názvy tříd se přečtou z vlastnosti `ClassLabel`: `['World', 'Sports', 'Business', 'Sci/Tech']`. Pozice v tomto seznamu je id třídy 0–3.
5. `y_train` a `y_test` se uloží jako NumPy pole s id tříd.
6. Vrátí se **zmrazený** objekt `NewsData`. Tento jeden objekt sdílejí všechny klasifikátory, což zaručuje, že se trénují i testují na úplně stejných textech.

### 7b. `DataExplorer(data, output_dir).report()`
1. Vypíše štítky a velikosti podmnožin (`2000 500`).
2. `_print_examples(5)` vypíše prvních 5 trénovacích textů pro každou ze čtyř tříd.
3. `class_balance()` sestaví tabulku s počtem příkladů na třídu v train a test, v oficiálním pořadí tříd.
4. Tabulka se vypíše a uloží do **`outputs/class_balance.csv`**.
5. Vypíše připomínku, abyste do `REPORT.md` zapsali svou hypotézu (které třídy se budou plést?).

Proč na kontrole vyváženosti záleží: při vyvážených třídách je prostá accuracy důvěryhodná metrika.

---

## 8. Příprava sdílených měřicích nástrojů

```python
evaluator = Evaluator(data, output_dir)
report = ResultsReport(output_dir)
```

- **`Evaluator`** je jeden hodnotitel sdílený všemi přístupy, takže všechny se měří úplně stejně.
- **`ResultsReport`** začíná s prázdným seznamem výsledků a později výsledkové soubory **přepíše**, takže staré výsledky se do tabulky nikdy nedostanou.

---

## 9. Kroky 2–5 – stejná smyčka pro každý klasifikátor

```python
for classifier in self.classifiers:
    self._headline(classifier.name)
    classifier.fit(data)                                    # (1) příprava / trénink
    preds, sec_per_100 = classifier.timed_predict(data.test_texts)   # (2) predikce + měření času
    report.add(evaluator.evaluate(classifier, preds, sec_per_100))   # (3) vyhodnocení
    classifier.run_experiments(data)                        # (4) doplňková analýza
    classifier.save()                                       # (5) uložení modelu
```

Pořadí je **A → B → (C)**. Každý přístup projde stejnými 5 akcemi níže.

### 9.1 Společné části (ze základní třídy `NewsClassifier`)

**`timed_predict(texts)`**
1. Spustí `Timer`.
2. Zavolá `predict(texts)` podtřídy na všech 500 testovacích textech.
3. Vrátí `(predikce, sekundy / počet_textů × 100)`, tedy rychlost v **sekundách na 100 textů**. Díky tomu jsou běhy s různou velikostí testu srovnatelné. Čas se měří od začátku do konce; u B zahrnuje i výpočet embeddingů.

**`evaluator.evaluate(classifier, preds, sec_per_100)`**
1. Spočítá **accuracy** (podíl správných predikcí) a **macro‑F1** (F1 pro každou třídu, pak průměr, takže každá třída má stejnou váhu).
2. Sestaví `ApproachResult` s klíčem, názvem, accuracy, macro‑F1, rychlostí, dobou tréninku a počtem označených příkladů.
3. Vypíše souhrnný řádek a pak `classification_report` (precision / recall / F1 pro každou třídu).
4. Uloží surové predikce do **`preds_X.npy`**.
5. `_save_confusion_matrix` nakreslí matici záměn (řádky = skutečná třída, sloupce = predikovaná třída, diagonála = správně) a uloží ji do **`confusion_X.png`**.
6. `_save_errors` sestaví tabulku všech chybných predikcí (`idx`, `true`, `pred`, `text`), uloží ji do **`errors_X.csv`** a vypíše prvních 8 chyb, každý text zkrácený na 160 znaků.
7. Vrátí `ApproachResult`. `report.add(...)` ho uloží pro finální tabulku.

`run_experiments()` a `save()` ve výchozím stavu nedělají nic. Podtřída je přepíše, jen když má co dělat.

### 9.2 Přístup A – `ZeroShotClassifier`

**Jak funguje:** pro každý kandidátní štítek pipeline sestaví větu-hypotézu (např. *„This example is sports.“*) a zeptá se NLI modelu, zda z textu zprávy **vyplývá** (entailment). Vyhraje štítek s nejvyšším skóre. Každý text stojí 4 volání modelu (jedno na štítek), proto je tento přístup při inferenci nejpomalejší.

**`fit(data)`** (žádný trénink, jen výběr promptu)
1. `_show_example` spustí pipeline na první testovací text s počátečním promptem a vypíše text, jeho skutečnou třídu a surový výstup pipeline (slovník se štítky seřazenými podle skóre), abyste viděli, jak výstup vypadá.
2. `wording_experiment(data)` vezme prvních `n_wording` = 200 testovacích textů a vyzkouší **4 formulace štítků × 2 šablony = 8 kombinací**:
   - Formulace štítků: `1 dataset names` (World, Sports…), `2 descriptive` (world politics, sports, business, science and technology), `3 specific` (international news and politics…), `4 one-word` (politics, sports, economy, technology)
   - Šablony: `default` (*„This example is {}.“*) a `news` (*„This news article is about {}.“*)
   - U každé kombinace změří accuracy a rychlost a vypíše řádek.
3. Tabulka se seřadí podle accuracy, nejlepší první. Řazení je *stabilní*, takže při shodě vyhraje kombinace zkoušená dřív a výběr je deterministický.
4. Tabulka se vypíše a uloží do **`zero_shot_wording.csv`**.
5. Nejlepší řádek se stane aktivním `self.prompt`.
6. `train_time_s = 0`, `n_labelled = 0`.

> ⚠️ Známá výhrada (ponechaná záměrně): nejlepší prompt se vybírá na *testovacích* textech, takže finální skóre A je mírně optimistické. Přísnější postup by použil samostatnou validační sadu.

**`predict(texts)`** → `_predict_with(texts, nejlepší_prompt)`
1. Zavolá pipeline s `candidate_labels`, `batch_size=16` a případně s `hypothesis_template`, pokud je nastavená.
2. Z každého výstupu vezme nejlepší štítek (`output["labels"][0]`) a převede ho na id třídy podle jeho **pozice v našem seznamu štítků**. Proto musí štítky zůstat v pořadí datasetu: World, Sports, Business, Sci/Tech.

Pak následují společné kroky: `timed_predict` → `evaluate` (`preds_A.npy`, `confusion_A.png`, `errors_A.csv`).

**`run_experiments()`** a **`save()`** nejsou přepsané, takže se nic nestane. Zero-shot model se neukládá, protože se nemění.

### 9.3 Přístup B – `EmbeddingClassifier`

**Jak funguje:** předtrénovaný enkodér (zde se netrénuje) převede každý text na vektor **384 čísel**. Texty s podobným významem dostanou podobné vektory. Nad těmito vektory se trénuje jen malá logistická regrese.

**`fit(data)`**
1. `encode(train_texts, show_progress_bar=True)` převede všech 2 000 trénovacích textů na vektory, po dávkách 64, s ukazatelem průběhu. Výsledek se uloží do `_X_train`.
2. Vypíše `Train embedding shape: (2000, 384)`.
3. Natrénuje logistickou regresi na `(_X_train, y_train)`.
4. `train_time_s` = doba kódování + doba trénování. Kódování se počítá do doby tréninku, protože bez něj by nebylo na čem trénovat.
5. `n_labelled = 2000`.

**`predict(texts)`**: zakóduje testovací texty a klasifikuje vektory natrénovanou logistickou regresí.

Pak následují společné kroky: `timed_predict` → `evaluate` (`preds_B.npy`, `confusion_B.png`, `errors_B.csv`).

**`run_experiments(data)`**, učící křivka („kolik textů musíme ručně označit?“):
1. Ověří, že proběhl `fit()`.
2. Zakóduje testovací texty ještě jednou (`X_test`).
3. `_curve_sizes` dá velikosti z `learning_curve_sizes`, které jsou menší než `n_train`, plus samotné `n_train`. Při výchozím nastavení to je **100, 500, 2000**.
4. Pro každou velikost *n* natrénuje **novou** logistickou regresi na prvních *n* uložených trénovacích vektorech (data jsou už zamíchaná, takže jde o náhodný vzorek) a změří accuracy a macro‑F1 na testu. Texty se znovu nekódují, protože se použije uložený `_X_train`.
5. Tabulka se vypíše a uloží do **`learning_curve.csv`**.
6. `_plot_learning_curve` nakreslí accuracy v závislosti na počtu příkladů (osa X v logaritmickém měřítku) a uloží ji do **`learning_curve.png`**.

**`save()`**: uloží jen malou logistickou regresi do **`news_sorter_lr.joblib`**, např. pro pozdější Gradio demo. Enkodér lze z Hubu kdykoli znovu stáhnout podle názvu.

### 9.4 Přístup C – `FineTunedClassifier` (jen s `--with-finetune`)

**Jak funguje:** na rozdíl od B se při tréninku aktualizují *všechny* váhy transformeru, takže se model specializuje na tuto úlohu. Obvykle dává nejlepší accuracy, ale zdaleka nejdelší trénink.

**`fit(data)`**
1. `AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4, id2label=…, label2id=…)` nasadí na předtrénované tělo DistilBERT novou hlavu pro 4 třídy. Hlava začíná s náhodnými váhami, proto je varování *„Some weights … were not initialized“* očekávané. Názvy tříd se uloží do modelu, takže hlásí „Sports“ místo „LABEL_1“.
2. `TrainingArguments`: 1 epocha, dávka 16 (trénink) / 32 (vyhodnocení), learning rate `5e-5` (záměrně malý, aby se předtrénované váhy přizpůsobily, ne zničily), weight decay 0,01, záznam do logu každých 25 kroků, žádné průběžné checkpointy, žádné externí loggery, seed 42.
3. `_tokenize(data.train)` převede texty na tokeny (`input_ids`, `attention_mask`) a ořízne je na 64 tokenů.
4. Vytvoří se `Trainer` s `DataCollatorWithPadding`, který doplňuje každou dávku jen na délku jejího nejdelšího textu. To je rychlejší než doplňovat vše na globální maximum.
5. `trainer.train()` proběhne uvnitř `Timer`, pak se vypíše `Training took X min`.
6. `train_time_s` = naměřený čas, `n_labelled = 2000`.

**`predict(texts)`**
1. Zabalí texty do `Dataset` a tokenizuje je.
2. `trainer.predict(...)` vrátí **logity**, jedno skóre na třídu pro každý text.
3. `argmax` vybere třídu s nejvyšším skóre.

Pak následují společné kroky: `timed_predict` → `evaluate` (`preds_C.npy`, `confusion_C.png`, `errors_C.csv`).

**`run_experiments()`** není přepsaný, takže se nic nestane.

**`save()`**: uloží model i tokenizér společně do složky **`outputs/distilbert-agnews/`**, připravené k opětovnému použití nebo `push_to_hub`.

---

## 10. Krok 6 – finální srovnání

Vypíše se nadpis (`>>> Final comparison`), pak proběhne `report.save()`:

1. Pokud nejsou žádné výsledky, vyhodí `RuntimeError`.
2. `to_frame()` sestaví tabulku, jeden řádek na přístup, se zaokrouhlenými sloupci: `Accuracy`, `Macro-F1`, `Sec / 100 texts`, `Training time (s)`, `Labelled examples`. Řádky jsou seřazené podle názvu (A, B, C).
3. Vypíše tabulku.
4. Uloží **`results.json`** (strojově čitelný) a **`results_table.csv`** (pro `REPORT.md`).
5. `_plot_comparison` nakreslí sloupcový graf Accuracy a Macro‑F1 pro každý přístup. Osa Y jde záměrně od 0 do 1, protože oříznutá osa by malé rozdíly zveličila. Graf se uloží jako **`comparison.png`**.
6. Vypíše, kam se soubory uložily, a připomínku, abyste tabulku zkopírovali do `REPORT.md` a napsali doporučení (≤ 150 slov).

---

## 11. Konec

Blok `with Timer() as total:` skončí a skript vypíše:

```
All steps finished in X.X min.
```

`run()` vrátí objekt `ResultsReport`, `main()` skončí a s ním i skript.

---

## 12. Přehled vytvořených souborů (`outputs/`)

| Soubor | Vzniká v | Obsah |
|---|---|---|
| `class_balance.csv` | Krok 1 | počet příkladů na třídu (train/test) |
| `zero_shot_wording.csv` | A – `fit` | accuracy a rychlost 8 kombinací promptů |
| `preds_A.npy`, `preds_B.npy`, (`preds_C.npy`) | vyhodnocení | surové predikce |
| `confusion_A.png`, `confusion_B.png`, (`confusion_C.png`) | vyhodnocení | matice záměn |
| `errors_A.csv`, `errors_B.csv`, (`errors_C.csv`) | vyhodnocení | všechny chybně zařazené texty |
| `learning_curve.csv`, `learning_curve.png` | B – `run_experiments` | accuracy pro 100 / 500 / 2000 příkladů |
| `news_sorter_lr.joblib` | B – `save` | natrénovaná logistická regrese |
| `distilbert-agnews/` | C – `save` | fine-tuned model + tokenizér |
| `results.json`, `results_table.csv`, `comparison.png` | Krok 6 | finální srovnání |

Výsledky posledního běhu ve složce (A + B, výchozí nastavení):

| Přístup | Accuracy | Macro-F1 | s / 100 textů | Trénink (s) | Označené příklady |
|---|---|---|---|---|---|
| A – Zero-shot | 0,706 | 0,688 | 19,2 | 0,0 | 0 |
| B – Embeddings + LR | 0,850 | 0,852 | 1,63 | 29,2 | 2000 |

---

## 13. Diagram průběhu

```
python run_all.py
 └─ main()
     ├─ parse_args()                      → args
     ├─ Config(...)                       → zmrazené nastavení
     ├─ build_classifiers()               → [A, B, (C)]   (tady se načítají modely!)
     └─ NewsSorterPipeline(config, clfs).run()
         ├─ mkdir outputs/, smazání preds_*.npy
         ├─ start Timeru
         ├─ Krok 1: NewsData.load() → DataExplorer.report()     → class_balance.csv
         ├─ Evaluator + prázdný ResultsReport
         ├─ pro každý klasifikátor:
         │    fit → timed_predict → evaluate → run_experiments → save
         │    (A: výběr promptu | B: trénink LR + učící křivka | C: fine-tuning)
         ├─ Krok 6: ResultsReport.save()  → results.json / .csv / comparison.png
         └─ „All steps finished in X min.“
```
