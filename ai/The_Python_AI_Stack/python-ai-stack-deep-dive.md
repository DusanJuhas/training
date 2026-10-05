# The Python AI Stack: A Deep Dive

*A companion to the "The Python AI Stack" deck. It covers 34 libraries in 10 categories.*
*Status as of 5 October 2026.*

For each library this document covers:

- its **unique selling point (USP)**
- how **popular** it is
- its **current version**
- **typical use cases**
- **links** to the project and its ecosystem
- **training materials**

Each category ends with a **comparison** of its libraries and guidance on when to pick which.

---

## How to read the numbers

- **Versions and release dates** come straight from the PyPI JSON API, checked on 5 October 2026.
- **GitHub stars and PyPI downloads per month** come from three sources, so they are snapshots of different ages:
  - The *best-of-ml-python* list, with data from around October 2025. Libraries marked ¹.
  - The *best-of-python* and *best-of-web-python* lists, with data from September/October 2026. Libraries marked ².
  - Two 2026 articles on agent frameworks. Libraries marked ³.

  Where none of these sources gave a figure, the tables say "n/a" and the text describes popularity in words. No figures were estimated.
- **Download counts include indirect installs.** For example, FastAPI and NumPy are pulled in as dependencies of many other packages, so their counts are inflated relative to direct use.
- **Popularity ≠ quality.** Stars measure attention. Downloads measure how embedded a library is. A smaller library can still be the right choice for your case.

## Quick reference: all 34 libraries

| Category | Library | PyPI package | Latest version | Released | GitHub ⭐ | PyPI downloads / month |
|---|---|---|---|---|---|---|
| Data processing | NumPy | `numpy` | 2.5.3 | 2026-09-06 | n/a | n/a |
| | Pandas | `pandas` | 3.0.6 | 2026-09-17 | 50K² | n/a |
| | Polars | `polars` | 1.44.2 | 2026-09-09 | 40K² | n/a |
| Machine learning | Scikit-learn | `scikit-learn` | 1.9.1 | 2026-09-10 | 64K¹ | 140M¹ |
| | XGBoost | `xgboost` | 3.4.1 | 2026-08-15 | 28K¹ | 31M¹ |
| | LightGBM | `lightgbm` | 4.7.0 | 2026-07-18 | 18K¹ | 11M¹ |
| Deep learning | PyTorch | `torch` | 2.14.1 | 2026-09-30 | 94K¹ | 70M¹ |
| | TensorFlow | `tensorflow` | 2.21.0 | 2026-03-06 | 200K¹ | 26M¹ |
| | JAX | `jax` | 0.11.2 | 2026-09-17 | 34K¹ | 12M¹ |
| | Keras | `keras` | 3.15.1 | 2026-07-29 | 64K¹ | 19M¹ |
| Experiment tracking | MLflow | `mlflow` | 3.16.1 | 2026-09-16 | 23K¹ | 26M¹ |
| | Weights & Biases | `wandb` | 0.30.0 | 2026-09-09 | 10K¹ | 20M¹ |
| | Comet | `comet_ml` | 3.58.7 | 2026-09-24 | n/a (Opik: 15K¹) | n/a |
| Visualization | Matplotlib | `matplotlib` | 3.11.2 | 2026-09-11 | 22K¹ | 120M¹ |
| | Seaborn | `seaborn` | 0.13.2 | 2024-01-25 | 14K¹ | 31M¹ |
| | Plotly | `plotly` | 7.1.0 | 2026-09-15 | 18K¹ | 37M¹ |
| | Altair | `altair` | 6.3.0 | 2026-09-15 | 10K¹ | 37M¹ |
| Model serving | FastAPI | `fastapi` | 0.142.2 | 2026-09-30 | 100K² | 350M² |
| | BentoML | `bentoml` | 1.4.39 | 2026-05-07 | 8.2K¹ | 180K¹ |
| | Gradio | `gradio` | 6.29.1 | 2026-10-02 | 40K¹ | 11M¹ |
| | Streamlit | `streamlit` | 1.65.0 | 2026-10-02 | 46K² | 19M¹ |
| Orchestration | Apache Airflow | `apache-airflow` | 3.3.2 | 2026-09-17 | 48K² | n/a |
| | Prefect | `prefect` | 3.8.7 | 2026-09-27 | 24K² | n/a |
| | Kubeflow Pipelines | `kfp` | 2.17.0 | 2026-07-09 | n/a | n/a |
| | Dagster | `dagster` | 1.13.25 | 2026-10-01 | 16K² | n/a |
| Data validation | Great Expectations | `great_expectations` | 1.23.2 | 2026-09-25 | n/a | n/a |
| | Evidently | `evidently` | 0.7.23 | 2026-09-11 | n/a | n/a |
| | Deepchecks | `deepchecks` | 0.19.1 | 2024-12-15 | n/a | n/a |
| Privacy and security | Presidio | `presidio-analyzer` | 2.2.364 | 2026-07-22 | n/a | n/a |
| | PySyft | `syft` | 0.10.0 | 2026-08-26 | 9.8K¹ | 32K¹ |
| AI agents | LangGraph | `langgraph` | 1.2.12 | 2026-09-21 | ~31K³ | n/a |
| | LangChain | `langchain` | 1.4.3 | 2026-09-28 | ~123K³ | n/a |
| | PydanticAI | `pydantic-ai` | 2.54.0 | 2026-10-03 | ~14K³ | n/a |
| | CrewAI | `crewai` | 1.15.23 | 2026-09-28 | ~42K–51K³ | n/a |

---

## 1. Data processing

### NumPy
- **USP:** NumPy is the universal n-dimensional array, the shared memory format of scientific Python. Its vectorized operations run in C, which makes them orders of magnitude faster than Python loops.
- **Popularity:** NumPy is effectively a dependency of the whole ecosystem: Pandas, scikit-learn, Matplotlib, SciPy and the deep-learning frameworks all build on it or interoperate with it. It is among the most-downloaded packages on PyPI.
- **Current version:** 2.5.3 (September 2026).
  - The 2.x line (since 2024) cleaned up the API and changed type-promotion rules, so very old code may need small fixes.
- **Typical use cases:**
  - numerical computing and linear algebra
  - random sampling and simulations
  - image arrays
  - feature matrices for ML
  - the backbone of custom algorithms
- **Links:**
  - Website: [numpy.org](https://numpy.org)
  - Docs: [numpy.org/doc/stable](https://numpy.org/doc/stable/)
  - GitHub: [numpy/numpy](https://github.com/numpy/numpy)
  - Ecosystem: [SciPy](https://scipy.org), [CuPy](https://cupy.dev) (NumPy on GPUs), [Numba](https://numba.pydata.org) (JIT compilation)
- **Training:**
  - [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html)
  - [NumPy tutorials](https://numpy.org/numpy-tutorials/)
  - [Learn page](https://numpy.org/learn/)

### Pandas
- **USP:** The DataFrame, a labeled, tabular data structure with a huge API for reading, cleaning, reshaping, joining and aggregating data. It is the lingua franca of data analysis in Python.
- **Popularity:** About 50K GitHub stars². It is used by virtually every data scientist and taught in nearly every data course.
- **Current version:** 3.0.6 (September 2026).
  - Pandas 3.0 made Copy-on-Write the default behavior and introduced a dedicated string data type.
  - Code that relied on chained assignment may need updating.
- **Typical use cases:**
  - exploratory data analysis
  - data cleaning
  - time series
  - preparing features for scikit-learn
  - Excel/CSV/SQL ETL scripts
- **Links:**
  - Website: [pandas.pydata.org](https://pandas.pydata.org)
  - Docs: [pandas.pydata.org/docs](https://pandas.pydata.org/docs/)
  - GitHub: [pandas-dev/pandas](https://github.com/pandas-dev/pandas)
  - [Ecosystem page](https://pandas.pydata.org/community/ecosystem.html)
- **Training:**
  - [Official getting-started tutorials](https://pandas.pydata.org/docs/getting_started/intro_tutorials/)
  - [Kaggle Learn: Pandas](https://www.kaggle.com/learn/pandas)
  - Wes McKinney, [*Python for Data Analysis*, 3rd ed.](https://wesmckinney.com/book/), free online

### Polars
- **USP:** Polars is a DataFrame engine written in Rust on Apache Arrow.
  - It is multi-threaded by default.
  - It offers a lazy API with a query optimizer: predicate and projection pushdown, plus streaming for larger-than-memory data.
  - Its expression syntax is consistent and composable.
- **Popularity:** About 40K GitHub stars². It is the fastest-growing DataFrame library of recent years.
- **Current version:** 1.44.2 (September 2026). The 1.x API is stable.
- **Typical use cases:**
  - large-file ETL (CSV/Parquet in the GBs)
  - feature pipelines where Pandas is too slow or memory-hungry
  - data engineering jobs on a single machine
- **Links:**
  - Website: [pola.rs](https://pola.rs)
  - Docs: [docs.pola.rs](https://docs.pola.rs)
  - GitHub: [pola-rs/polars](https://github.com/pola-rs/polars)
  - Ecosystem: interoperates with Arrow, DuckDB, Pandas and NumPy. Polars Cloud is for distributed runs.
- **Training:**
  - [Polars User Guide](https://docs.pola.rs/user-guide/), which includes a "coming from Pandas" migration guide

### Comparison: data processing

| | NumPy | Pandas | Polars |
|---|---|---|---|
| Core abstraction | n-dim homogeneous array | labeled DataFrame / Series | DataFrame + expressions |
| Execution | eager, single-threaded C (BLAS for linear algebra) | eager, mostly single-threaded | eager **or lazy**, multi-threaded Rust |
| Strength | numerical math, linear algebra | breadth of API, ecosystem, familiarity | speed, memory efficiency, query optimization |
| Weakness | no labels, no mixed types | slower and memory-hungry on big data | smaller ecosystem, different syntax |
| Learning curve | low | low–medium | medium (new mental model) |

**When to choose:**
- **NumPy** for numerical arrays and math.
- **Pandas** as the default for analysis and anything that has to plug into the broader ecosystem.
- **Polars** when data is large or pipelines are slow. It is increasingly the default for new data-engineering code.

The three combine well: NumPy underneath, Polars for heavy lifting, Pandas at the edges where libraries expect it.

---

## 2. Machine learning

### Scikit-learn
- **USP:** One consistent `fit / predict / transform` API across hundreds of classical algorithms, plus pipelines, preprocessing, cross-validation and metrics. It sets the de facto interface standard that other libraries imitate.
- **Popularity:** 64K stars and about 140M downloads per month¹. It is the most-used classical ML library.
- **Current version:** 1.9.1 (September 2026).
- **Typical use cases:**
  - classification and regression baselines
  - clustering
  - dimensionality reduction
  - feature engineering pipelines
  - model selection and hyperparameter search
- **Links:**
  - Website and docs: [scikit-learn.org](https://scikit-learn.org)
  - GitHub: [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn)
  - Ecosystem: imbalanced-learn, scikit-learn-contrib. skops is for model sharing.
- **Training:**
  - [Official tutorials](https://scikit-learn.org/stable/tutorial/index.html)
  - Free [Inria scikit-learn MOOC](https://inria.github.io/scikit-learn-mooc/)
  - [Kaggle Learn: Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)

### XGBoost
- **USP:** A highly optimized, regularized gradient-boosted decision tree library. It has a long track record of winning tabular-data competitions, plus:
  - native handling of missing values
  - GPU training
  - distributed training on Dask and Spark
- **Popularity:** 28K stars and about 31M downloads per month¹.
- **Current version:** 3.4.1 (August 2026).
- **Typical use cases:**
  - credit scoring
  - churn and fraud prediction
  - ranking (learning-to-rank)
  - any structured/tabular prediction problem
- **Links:**
  - Website: [xgboost.ai](https://xgboost.ai)
  - Docs: [xgboost.readthedocs.io](https://xgboost.readthedocs.io)
  - GitHub: [dmlc/xgboost](https://github.com/dmlc/xgboost)
- **Training:**
  - [Official tutorials](https://xgboost.readthedocs.io/en/stable/tutorials/index.html)
  - [Kaggle Learn: Intermediate ML](https://www.kaggle.com/learn/intermediate-machine-learning), which includes a chapter on XGBoost

### LightGBM
- **USP:** Microsoft's gradient boosting framework. Histogram-based split finding and leaf-wise tree growth make it typically the fastest and most memory-efficient booster on large datasets. It also handles categorical features natively.
- **Popularity:** 18K stars and about 11M downloads per month¹.
- **Current version:** 4.7.0 (July 2026).
- **Typical use cases:**
  - large tabular datasets (millions of rows)
  - ranking
  - fast iteration in Kaggle-style experimentation
- **Links:**
  - Docs: [lightgbm.readthedocs.io](https://lightgbm.readthedocs.io)
  - GitHub: [microsoft/LightGBM](https://github.com/microsoft/LightGBM)
- **Training:**
  - [Python quick start](https://lightgbm.readthedocs.io/en/latest/Python-Intro.html)
  - [Parameter-tuning guide](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)

### Comparison: machine learning

| | Scikit-learn | XGBoost | LightGBM |
|---|---|---|---|
| Scope | broad toolkit (dozens of algorithms + tooling) | one algorithm family, deeply optimized | one algorithm family, deeply optimized |
| Best at | baselines, preprocessing, evaluation | accuracy and robustness on tabular data | speed on large tabular data |
| GPU | no (mostly CPU) | yes | yes (more limited) |
| Categorical features | via encoders | native (newer versions) | native |
| Works with sklearn API | it *is* the API | yes (`XGBClassifier`) | yes (`LGBMClassifier`) |

**When to choose:**
- Start every project with **scikit-learn** for the pipeline, a baseline and evaluation.
- For the final tabular model, try **XGBoost** and **LightGBM** (and CatBoost, an alternative outside this list). Which one wins is dataset-specific.
- LightGBM is usually faster to train. XGBoost is often slightly more robust out of the box.

---

## 3. Deep learning

### PyTorch
- **USP:** Eager, Pythonic tensors with autograd. You debug models like normal Python code.
  - `torch.compile` adds graph-level speed-ups.
  - It is the framework most research and most open-source LLMs are built on.
- **Popularity:** 94K stars and about 70M downloads per month¹. It is the dominant deep-learning framework in research and for generative AI.
- **Current version:** 2.14.1 (September 2026).
- **Typical use cases:**
  - research prototypes
  - LLM training and fine-tuning
  - computer vision
  - speech
  - anything in the Hugging Face ecosystem
- **Links:**
  - Website: [pytorch.org](https://pytorch.org)
  - Docs: [docs.pytorch.org](https://docs.pytorch.org)
  - GitHub: [pytorch/pytorch](https://github.com/pytorch/pytorch)
  - Ecosystem: [PyTorch Ecosystem](https://pytorch.org/ecosystem/), [Lightning](https://lightning.ai), [Hugging Face Transformers](https://huggingface.co/docs/transformers), TorchVision, TorchAudio
- **Training:**
  - [Official PyTorch tutorials](https://docs.pytorch.org/tutorials/)
  - [Learn PyTorch for Deep Learning](https://www.learnpytorch.io) (free online book)
  - [fast.ai Practical Deep Learning](https://course.fast.ai)

### TensorFlow
- **USP:** Google's end-to-end production platform:
  - `tf.data` pipelines
  - distributed training
  - TFX for MLOps
  - deployment to servers (TF Serving), mobile/edge ([LiteRT](https://ai.google.dev/edge/litert), formerly TF Lite) and browsers (TensorFlow.js)
- **Popularity:** It has the most GitHub stars of any ML library (200K¹), a legacy of its early lead, and about 26M downloads per month¹. Research and new-project mindshare have shifted toward PyTorch and JAX, but TensorFlow remains widespread in production and on edge devices.
- **Current version:** 2.21.0 (March 2026). Releases are now less frequent than for PyTorch or JAX.
- **Typical use cases:**
  - maintaining existing production systems
  - mobile and embedded ML
  - TFX pipelines
  - browser ML with TF.js
- **Links:**
  - Website: [tensorflow.org](https://www.tensorflow.org)
  - GitHub: [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)
  - Ecosystem: [TFX](https://www.tensorflow.org/tfx), [TensorFlow.js](https://www.tensorflow.org/js), TensorBoard
- **Training:**
  - [TensorFlow tutorials](https://www.tensorflow.org/tutorials)
  - [TensorFlow learning resources](https://www.tensorflow.org/resources/learn-ml)

### JAX
- **USP:** NumPy-compatible arrays plus composable function transformations:
  - `grad` for derivatives
  - `jit` for XLA compilation
  - `vmap` for auto-vectorization
  - sharding for multi-device parallelism

  JAX is functional and very fast on TPUs and GPUs. Google DeepMind and many frontier labs use it.
- **Popularity:** 34K stars and about 12M downloads per month¹. Its user base is smaller but heavily weighted toward advanced research and large-scale training.
- **Current version:** 0.11.2 (September 2026). It is still 0.x, and the API evolves.
- **Typical use cases:**
  - large-scale model training on TPU
  - scientific computing and differentiable simulation
  - research needing custom gradients or vectorization
- **Links:**
  - Docs: [docs.jax.dev](https://docs.jax.dev)
  - GitHub: [jax-ml/jax](https://github.com/jax-ml/jax)
  - Ecosystem: [Flax](https://flax.readthedocs.io) (neural networks), [Optax](https://optax.readthedocs.io) (optimizers), Orbax (checkpointing)
- **Training:**
  - [JAX tutorials](https://docs.jax.dev/en/latest/tutorials.html)
  - Flax NNX getting-started guides

### Keras
- **USP:** The friendliest high-level deep-learning API. Since Keras 3 it is **multi-backend**: the same model code runs on TensorFlow, JAX or PyTorch.
- **Popularity:** 64K stars and about 19M downloads per month¹.
- **Current version:** 3.15.1 (July 2026).
- **Typical use cases:**
  - teaching and learning deep learning
  - fast prototyping
  - standard architectures with KerasHub pretrained models
  - teams wanting backend portability
- **Links:**
  - Website: [keras.io](https://keras.io)
  - GitHub: [keras-team/keras](https://github.com/keras-team/keras)
  - Ecosystem: [KerasHub](https://keras.io/keras_hub/), KerasTuner
- **Training:**
  - [Getting started](https://keras.io/getting_started/)
  - [Code examples](https://keras.io/examples/), hundreds of complete worked examples
  - François Chollet, *Deep Learning with Python* (Manning)

### Comparison: deep learning

| | PyTorch | TensorFlow | JAX | Keras |
|---|---|---|---|---|
| Level | low/mid | low/mid/high | low (functional) | high |
| Style | eager, object-oriented | graph + eager | pure functions + transforms | declarative layers |
| Research share | dominant | declining | strong at frontier labs | moderate |
| Production/deployment | strong (TorchServe, ExecuTorch, ONNX) | strongest on edge/mobile | via XLA / export | via the chosen backend |
| Hardware sweet spot | NVIDIA GPUs | GPUs, TPUs, mobile | TPUs, large GPU clusters | any (inherits backend) |
| Learning curve | medium | medium–high | high | low |

**When to choose:**
- **PyTorch** is the default for new work and anything built on Hugging Face.
- **JAX** suits TPU-scale training or functional, research-heavy code.
- **TensorFlow** is mainly for existing systems and edge deployment.
- **Keras** is for learning, quick prototypes, or when you want to switch backends later.

---

## 4. Experiment tracking

### MLflow
- **USP:** The open-source, vendor-neutral standard for the ML lifecycle:
  - experiment tracking
  - model registry
  - model packaging and serving
  - since MLflow 3, GenAI tracing and LLM evaluation

  You can self-host it for free, or use it managed (built into Databricks and offered by major clouds).
- **Popularity:** 23K stars and about 26M downloads per month¹. It is the most widely deployed tracking tool in enterprises.
- **Current version:** 3.16.1 (September 2026).
- **Typical use cases:**
  - tracking and comparing runs
  - a central model registry with stage/alias promotion
  - autologging for sklearn, XGBoost and PyTorch
  - LLM app tracing
- **Links:**
  - Website: [mlflow.org](https://mlflow.org)
  - Docs: [mlflow.org/docs/latest](https://mlflow.org/docs/latest/)
  - GitHub: [mlflow/mlflow](https://github.com/mlflow/mlflow)
- **Training:**
  - Getting-started tutorials in the [official docs](https://mlflow.org/docs/latest/)
  - Databricks Academy courses on MLflow

### Weights & Biases (W&B)
- **USP:** The most polished hosted experience:
  - live interactive dashboards
  - hyperparameter **Sweeps**
  - **Artifacts** for dataset and model lineage
  - collaborative **Reports**
  - **Weave** for tracing and evaluating LLM apps

  It is very popular in deep-learning research teams.
- **Popularity:** 10K stars and about 20M downloads per month¹. Many open-source training frameworks have built-in W&B logging.
- **Current version:** `wandb` 0.30.0 (September 2026).
- **Typical use cases:**
  - deep-learning and LLM training runs
  - hyperparameter search
  - sharing results across a research team
- **Links:**
  - Website: [wandb.ai](https://wandb.ai)
  - Docs: [docs.wandb.ai](https://docs.wandb.ai)
  - GitHub: [wandb/wandb](https://github.com/wandb/wandb)
- **Training:**
  - [W&B AI Academy](https://wandb.ai/site/courses), free courses on MLOps, LLMs and evaluation

### Comet
- **USP:** An end-to-end experiment management platform:
  - tracking code, hyperparameters, metrics and models
  - run comparison and a model registry
  - production model monitoring

  Its open-source sibling **Opik** (15K stars¹) focuses on LLM tracing and evaluation.
- **Popularity:** No GitHub star figure is available for the `comet_ml` SDK itself. Opik is the company's open-source flagship. Comet is a commercial alternative with an established enterprise customer base, smaller than MLflow or W&B in community size.
- **Current version:** `comet_ml` 3.58.7 (September 2026).
- **Typical use cases:**
  - teams that want a managed (SaaS or on-prem) tracking platform with monitoring included
  - LLM evaluation with Opik
- **Links:**
  - Website: [comet.com](https://www.comet.com)
  - Docs: [comet.com/docs/v2](https://www.comet.com/docs/v2/)
  - Opik: [github.com/comet-ml/opik](https://github.com/comet-ml/opik)
- **Training:**
  - Quickstarts and integration guides in the [Comet docs](https://www.comet.com/docs/v2/)
  - Opik documentation tutorials

### Comparison: experiment tracking

| | MLflow | Weights & Biases | Comet |
|---|---|---|---|
| License / model | open source (Apache 2.0), self-host or managed | proprietary SaaS (client is open source); self-hosted option for enterprises | proprietary SaaS / on-prem; Opik is open source |
| Strength | registry, deployment, vendor neutrality, Databricks integration | UX, dashboards, sweeps, research collaboration | all-in-one platform incl. production monitoring |
| GenAI support | MLflow Tracing and evaluation | Weave | Opik |
| Cost | free to self-host | free tier, paid teams | free tier, paid teams |

**When to choose:**
- **MLflow** if you need open source, self-hosting or Databricks.
- **W&B** for deep-learning research teams that value visualization and collaboration.
- **Comet** if you want one commercial vendor covering tracking through monitoring, or if you are standardizing on Opik for LLM evaluation.

---

## 5. Visualization

### Matplotlib
- **USP:** Total, low-level control over every pixel of a figure, plus publication-quality output (PNG, SVG, PDF, LaTeX text). It is the rendering engine under Pandas `.plot()` and Seaborn.
- **Popularity:** 22K stars and about 120M downloads per month¹. It is the most-installed plotting library.
- **Current version:** 3.11.2 (September 2026).
- **Typical use cases:**
  - scientific papers
  - custom or unusual charts
  - static reports
  - figures generated in batch jobs
- **Links:**
  - Website: [matplotlib.org](https://matplotlib.org)
  - [Gallery](https://matplotlib.org/stable/gallery/)
  - GitHub: [matplotlib/matplotlib](https://github.com/matplotlib/matplotlib)
- **Training:**
  - [Official tutorials](https://matplotlib.org/stable/tutorials/index.html)
  - [Cheat sheets](https://matplotlib.org/cheatsheets/)

### Seaborn
- **USP:** Statistical plots in one line: distributions, regressions, categorical comparisons and pair plots. It works directly on DataFrames and has attractive defaults.
- **Popularity:** 14K stars and about 31M downloads per month¹.
- **Current version:** 0.13.2 (January 2024).
  - The project is mature, and releases are infrequent.
  - The newer `seaborn.objects` interface offers a grammar-of-graphics style.
- **Typical use cases:**
  - exploratory data analysis
  - checking distributions and correlations
  - quick, good-looking statistical charts
- **Links:**
  - Website: [seaborn.pydata.org](https://seaborn.pydata.org)
  - GitHub: [mwaskom/seaborn](https://github.com/mwaskom/seaborn)
- **Training:**
  - [Official tutorial](https://seaborn.pydata.org/tutorial.html)
  - [Kaggle Learn: Data Visualization](https://www.kaggle.com/learn/data-visualization), which is Seaborn-based

### Plotly
- **USP:** Interactive, browser-rendered charts with zoom, hover and pan.
  - Plotly Express makes most charts a one-liner.
  - It includes 3D and maps.
  - The same figures power **Dash**, a framework for analytical web apps.
- **Popularity:** 18K stars and about 37M downloads per month¹.
- **Current version:** 7.1.0 (September 2026).
- **Typical use cases:**
  - interactive notebooks
  - dashboards (Dash)
  - shareable HTML reports
  - geospatial and 3D plots
- **Links:**
  - Website: [plotly.com/python](https://plotly.com/python/)
  - GitHub: [plotly/plotly.py](https://github.com/plotly/plotly.py)
  - [Dash](https://dash.plotly.com)
- **Training:**
  - [Plotly Express guide](https://plotly.com/python/plotly-express/)
  - [Dash tutorial](https://dash.plotly.com/tutorial)

### Altair
- **USP:** Declarative statistical visualization based on the **Vega-Lite** grammar.
  - You map columns to encodings, and Altair handles scales, legends and interactivity, including linked selections across charts.
  - The resulting chart specs are concise and reproducible.
- **Popularity:** 10K stars and about 37M downloads per month¹. The download count is boosted because Streamlit depends on Altair.
- **Current version:** 6.3.0 (September 2026).
- **Typical use cases:**
  - exploratory analysis with linked interactive views
  - charts in Streamlit and Jupyter
  - teaching the grammar of graphics
- **Links:**
  - Website and docs: [altair-viz.github.io](https://altair-viz.github.io)
  - GitHub: [vega/altair](https://github.com/vega/altair)
  - [Vega-Lite](https://vega.github.io/vega-lite/)
- **Training:**
  - [Altair user guide](https://altair-viz.github.io/user_guide/data.html)
  - University of Washington [Visualization Curriculum](https://uwdata.github.io/visualization-curriculum/), free and built on Altair

### Comparison: visualization

| | Matplotlib | Seaborn | Plotly | Altair |
|---|---|---|---|---|
| Paradigm | imperative, low-level | high-level statistical | high-level, interactive | declarative grammar |
| Output | static (PNG/SVG/PDF) | static (via Matplotlib) | interactive HTML/JS | interactive HTML/JS (Vega) |
| Customization | unlimited | good (falls back to Matplotlib) | very good | good within the grammar |
| Large data | good | good | moderate (WebGL helps) | limited in browser (aggregate first) |
| Best for | papers, bespoke figures | quick EDA | dashboards, sharing | linked exploratory views |

**When to choose:**
- **Seaborn** for fast EDA.
- **Matplotlib** when you need full control or print quality.
- **Plotly** when the audience will interact with the chart or it goes into a dashboard.
- **Altair** if you like declarative specs and linked selections, especially inside Streamlit.

---

## 6. Model serving

### FastAPI
- **USP:** Production-grade REST APIs from Python type hints:
  - automatic request validation via Pydantic
  - auto-generated OpenAPI/Swagger docs
  - async support
  - high performance on Starlette/Uvicorn
- **Popularity:** 100K stars and about 350M downloads per month² (downloads are heavily boosted by indirect installs). It is the most popular modern Python web framework.
- **Current version:** 0.142.2 (September 2026). It is still 0.x but widely used in production.
- **Typical use cases:**
  - inference REST endpoints
  - backend APIs for ML products
  - microservices
  - wrapping LLM calls behind an API
- **Links:**
  - Docs: [fastapi.tiangolo.com](https://fastapi.tiangolo.com)
  - GitHub: [fastapi/fastapi](https://github.com/fastapi/fastapi)
  - Ecosystem: [Pydantic](https://docs.pydantic.dev), Starlette, Uvicorn, [SQLModel](https://sqlmodel.tiangolo.com)
- **Training:**
  - The official [Tutorial – User Guide](https://fastapi.tiangolo.com/tutorial/), widely regarded as one of the best framework tutorials

### BentoML
- **USP:** Purpose-built for **ML** serving. On top of what a generic web framework offers, it adds:
  - adaptive (dynamic) batching
  - multi-model composition
  - GPU resource configuration
  - one-command containerization into "Bentos"

  It deploys anywhere or to the managed BentoCloud.
- **Popularity:** 8.2K stars and about 180K downloads per month¹. It is a specialist tool with a smaller but focused user base.
- **Current version:** 1.4.39 (May 2026).
- **Typical use cases:**
  - high-throughput model inference
  - serving open-source LLMs and multi-model pipelines
  - standardizing model packaging across teams
- **Links:**
  - Website: [bentoml.com](https://www.bentoml.com)
  - Docs: [docs.bentoml.com](https://docs.bentoml.com)
  - GitHub: [bentoml/BentoML](https://github.com/bentoml/BentoML)
- **Training:**
  - [BentoML docs](https://docs.bentoml.com): quickstarts and example projects, including LLM serving

### Gradio
- **USP:** The fastest way to put a web UI on a model. A few lines give you inputs, outputs and a shareable public link, and it is the native app format of **Hugging Face Spaces**. It also exposes your app as an API and can act as an MCP server.
- **Popularity:** 40K stars and about 11M downloads per month¹.
- **Current version:** 6.29.1 (October 2026).
- **Typical use cases:**
  - ML demos
  - internal testing UIs
  - chatbots
  - showcasing models on Hugging Face
- **Links:**
  - Website: [gradio.app](https://www.gradio.app)
  - GitHub: [gradio-app/gradio](https://github.com/gradio-app/gradio)
  - [Hugging Face Spaces](https://huggingface.co/spaces)
- **Training:**
  - [Gradio Quickstart and Guides](https://www.gradio.app/guides/quickstart)
  - Gradio chapter in the [Hugging Face course](https://huggingface.co/learn)

### Streamlit
- **USP:** Turns a plain Python script into an interactive data app. There are no callbacks: the script reruns top to bottom on every interaction. It has rich widgets, caching and multipage apps, and is owned by Snowflake, which offers native hosting in Snowflake.
- **Popularity:** 46K stars² and about 19M downloads per month¹.
- **Current version:** 1.65.0 (October 2026).
- **Typical use cases:**
  - internal data apps and dashboards
  - model exploration tools
  - LLM chat front-ends
  - prototypes for stakeholders
- **Links:**
  - Website: [streamlit.io](https://streamlit.io)
  - Docs: [docs.streamlit.io](https://docs.streamlit.io)
  - GitHub: [streamlit/streamlit](https://github.com/streamlit/streamlit)
  - [Gallery](https://streamlit.io/gallery)
- **Training:**
  - [Get started tutorial](https://docs.streamlit.io/get-started)
  - [30 Days of Streamlit](https://30days.streamlit.app)

### Comparison: model serving

| | FastAPI | BentoML | Gradio | Streamlit |
|---|---|---|---|---|
| What you build | API (no UI) | inference service (API) | model demo UI (+ API) | data app / dashboard |
| Audience | other programs | other programs | end users trying a model | business users, analysts |
| ML-specific features | none (generic) | batching, GPU, packaging | ML input/output components | caching, charts |
| Scalability | high (with workers / k8s) | high (designed for it) | demo to moderate | moderate |
| Learning curve | low–medium | medium | very low | very low |

**When to choose:**
- **FastAPI** for a general API around a model or LLM.
- **BentoML** when inference throughput, batching or GPU efficiency matter.
- **Gradio** to demo a model, especially on Hugging Face.
- **Streamlit** for data apps and dashboards that business users explore.

A common combination is a FastAPI or BentoML backend with a Streamlit or Gradio front-end.

---

<!-- CONTINUE -->
