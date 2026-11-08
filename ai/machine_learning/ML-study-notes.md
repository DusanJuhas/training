# Machine Learning – Developer Study Notes

Classic Machine Learning: Concepts • Examples • Interview Preparation

**Scope:** classic (non-deep-learning) ML on tabular data: the ML workflow, data handling, the main supervised and unsupervised algorithms, and how to evaluate them. Gradient boosting and neural networks are mentioned but not covered in depth.

## Contents
1. What is Machine Learning?
2. Types of Machine Learning
3. Data Collection and Understanding
4. Data Preprocessing and Model Validation
5. Exploratory Data Analysis (EDA)
6. Machine Learning Algorithms – Overview
7. Common Machine Learning Algorithms by Learning Type
8. Regression Algorithms
9. Classification Algorithms
10. Clustering Algorithms
11. Linear Regression
12. Logistic Regression
13. Decision Tree
14. Random Forest
15. Support Vector Machine (SVM)
16. K-Nearest Neighbors (KNN)
17. K-Means Clustering
18. Principal Component Analysis (PCA)

Diagrams are described in *italics*; formulas are in LaTeX.

---

## 1. What is Machine Learning?

### 1.1 Definition
Machine Learning (ML) is a subset of Artificial Intelligence that enables computers to learn patterns from data without being explicitly programmed.

### 1.2 Key Points
- ML uses algorithms to learn from historical data.
- It automatically finds patterns and makes predictions or decisions.
- It tends to improve with more *relevant, good-quality* data. More noisy, biased or irrelevant data does not help, and a model that is too simple stops improving (high bias). Data quality matters as much as quantity.
- Widely used in real-world applications.

### 1.3 How Machine Learning Works
*Diagram:* Input Data → Machine Learning Algorithm → Learn Patterns → Prediction / Decision

### 1.4 Examples
- Email spam detection
- House price prediction
- Image recognition
- Recommendation systems (YouTube, Netflix)

### 1.5 Types of Problems
- Classification (e.g., spam or not)
- Regression (e.g., predict house price)
- Clustering (e.g., group similar customers)
- Anomaly Detection (e.g., fraud detection)

---

## 2. Types of Machine Learning

Machine Learning is commonly divided into the following types, based on the type of data and the learning approach.

### 2.1 Supervised Learning
- Uses labeled data (input and output both are known).
- Learns a mapping from input to output.
- Used for classification and regression problems.
- Examples: Email spam detection, house price prediction.

*Diagram:* Labeled Data → Learning Algorithm → Model → Prediction

### 2.2 Unsupervised Learning
- Uses unlabeled data (only input data is available).
- Finds hidden patterns or structure in the data.
- Used for clustering, dimensionality reduction, etc.
- Examples: Customer segmentation, grouping similar products.

*Diagram:* Unlabeled Data → Learning Algorithm → Patterns / Clusters → Insights

### 2.3 Reinforcement Learning
- Learns by interacting with an environment.
- Receives rewards or penalties for actions.
- Goal is to learn a policy that maximizes total (cumulative) reward.
- Examples: Game playing (chess, Go), robotics. Some components of autonomous driving use RL, although production driving systems rely mostly on supervised and imitation learning.

*Diagram:* Environment → Agent → Action → (Reward back to Agent / Environment loop)

### 2.4 Semi-Supervised Learning
- Uses a combination of labeled and unlabeled data.
- Useful when labeled data is limited and unlabeled data is large.
- Combines the advantages of supervised and unsupervised learning.
- Examples: Image classification, text classification.

### 2.5 Self-Supervised Learning
- Generates labels automatically from the data itself (e.g., predict the next word, fill in a masked word, predict a hidden part of an image).
- Needs no human labeling, so it can use huge unlabeled datasets.
- It is how modern large language models (GPT, Claude, BERT) and many vision models are pre-trained; they are then fine-tuned for specific tasks.

---

## 3. Data Collection and Understanding

Data collection is the process of gathering relevant data from different sources and understanding its structure, quality and meaning before building a machine learning model.

### 3.1 Sources of Data
- Internal data (company database, application data)
- External data (public datasets, APIs, websites)
- Manual collection (surveys, forms, sensors)
- Open source datasets (Kaggle, UCI, government portals)

*Diagram:* Data Sources → Databases / APIs / Websites / Files (CSV, Excel) / Sensors/Devices

### 3.2 Types of Data
- Structured Data: Tabular format (rows and columns)
- Unstructured Data: Text, images, audio, video
- Semi-structured Data: JSON, XML, logs
- Labeled Data: Contains target/output variable
- Unlabeled Data: No target/output variable

### 3.3 Data Collection Process
Identify Problem → Find Data Sources → Check Legal & Privacy Constraints → Collect Data → Label Data (if needed) → Store Data (CSV/Database) → Check Data Quality → Explore Data

### 3.4 Common Data Formats and Storage

| Format / Storage | Description | Example Use |
|---|---|---|
| CSV | Simple text format, widely used | Tabular data (small to medium) |
| Excel | Easy to use, good for small datasets | Business data, manual collection |
| JSON | Common in web and API data | API responses, semi-structured data |
| Relational database tables (queried with SQL) | Structured tables with a defined schema | Large structured data |
| Parquet | Columnar, compressed binary format, fast to read selected columns | Big data and data lakes |

### 3.5 Understanding the Data
- Look at the shape of the data (rows and columns).
- Check the data types of each column.
- Identify missing values.
- Understand the meaning of each feature.
- Find basic statistics (mean, median, min, max).
- Visualize the data to find patterns and outliers.

### 3.6 Data Quality and Risk Checks
- **Legal and privacy:** check licensing and terms of use (especially for web scraping) and personal-data rules such as GDPR.
- **Labeling:** labels are often expensive and noisy. Check label quality and consistency between annotators.
- **Sampling bias:** make sure the data represents the population the model will be used on.
- **Data leakage:** make sure no feature contains information that will not be available at prediction time (e.g., future values or a proxy of the target).

### 3.7 Tools for Data Collection and Exploration
- Python (pandas, numpy)
- Jupyter Notebook
- SQL (for database data)
- Excel (small datasets)
- APIs (requests library)
- Web scraping (BeautifulSoup, Scrapy)
- Visualization (matplotlib, seaborn)

*Diagram:* Data Exploration Tools → pandas (Data analysis), matplotlib/seaborn (Visualization), Jupyter Notebook (Experimentation), SQL (Querying data), Excel (Basic analysis)

---

## 4. Data Preprocessing and Model Validation

Data preprocessing turns raw data into a form a model can learn from. Validation measures how well the model will work on new, unseen data.

### 4.1 Handling Missing Values
- Remove rows or columns (only when few values are missing).
- Impute: mean or median (numerical), most frequent value (categorical), or model-based imputation.
- Optionally add an "is_missing" indicator feature, because missingness itself can carry information.

### 4.2 Encoding Categorical Features
- One-hot encoding: one binary column per category (nominal data, e.g., city).
- Ordinal encoding: integers that keep a natural order (e.g., low < medium < high).
- Target encoding: replace a category with the mean target value. It must be computed on training data only to avoid leakage.

### 4.3 Feature Scaling
- Standardization: $z = \dfrac{x - \mu}{\sigma}$ (mean 0, unit variance).
- Min-Max normalization: $x' = \dfrac{x - x_{min}}{x_{max} - x_{min}}$ (range 0–1).
- Required for distance- and margin-based methods (KNN, K-Means, SVM, PCA) and helps gradient descent. Tree-based models do not need it.

### 4.4 Train / Validation / Test Split
- Training set: fits the model.
- Validation set: used to tune hyperparameters and select models.
- Test set: used once, at the end, for an unbiased performance estimate.
- Use a stratified split for classification, and a time-based split for time series (never shuffle future data into training).
- Fit preprocessing steps (scalers, imputers, encoders) on the training data only; scikit-learn `Pipeline` does this automatically.

### 4.5 Cross-Validation
- K-fold: split the data into k folds, train on k−1 folds and validate on the remaining one, then rotate and average the scores.
- Gives a more reliable estimate than a single split, and is the standard way to tune hyperparameters (e.g., `GridSearchCV`, `RandomizedSearchCV`).

### 4.6 Bias–Variance Trade-off
- High bias (underfitting): the model is too simple and performs poorly on both training and test data.
- High variance (overfitting): the model memorizes the training data and performs well on training but poorly on test data.
- Remedies for overfitting: more data, regularization, simpler models, pruning, ensembles, early stopping.

### 4.7 Imbalanced Data
- When one class is rare (fraud, disease), accuracy is misleading.
- Use precision, recall, F1 and PR-AUC; use class weights, resampling (over/under-sampling, SMOTE) or decision-threshold tuning.

---

## 5. Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) is the process of analyzing and visualizing data to understand its main characteristics, find patterns, detect anomalies and get insights before building a machine learning model.

### 5.1 Goals of EDA
- Understand the structure of the data.
- Find missing values and outliers.
- Analyze distributions of features.
- Find relationships between variables.
- Discover patterns and trends.
- Get insights for feature engineering.
- Make better decisions for model building.

### 5.2 Summary Statistics
Help to understand the central tendency, spread and shape of the data. Common statistics:

| Measure | Meaning |
|---|---|
| Count | Total number of values |
| Mean | Average value |
| Median | Middle value |
| Std Deviation | How much data is spread |
| Min / Max | Smallest and largest value |
| Percentiles | Value below which a % of data falls |

### 5.3 Univariate Analysis
- Analyze each feature individually.
- Understand the distribution, range and outliers.
- Common plots: Histogram, Box Plot, KDE Plot.

*Charts:* Histogram (Age), Box Plot (Salary)

### 5.4 Bivariate Analysis
- Analyze the relationship between two variables.
- Used to find correlation, trends and patterns.
- Common plots: Scatter Plot, Line Plot, Bar Plot.

*Charts:* Scatter Plot (Age vs Salary), Bar Plot (City vs Count: Delhi, Mumbai, Bangalore)

### 5.5 Multivariate Analysis
- Analyze multiple variables together.
- Helps to find complex relationships and patterns.
- Common plots: Pair Plot, Heatmap.

*Charts:* Pair Plot (Age, Salary, Spending Score); Correlation Heatmap (illustrative values):

| | Age | Salary | Spending | Exp |
|---|---|---|---|---|
| Age | 1.0 | 0.6 | 0.2 | 0.4 |
| Salary | 0.6 | 1.0 | 0.3 | 0.5 |
| Spending | 0.2 | 0.3 | 1.0 | 0.1 |
| Experience | 0.4 | 0.5 | 0.1 | 1.0 |

Note on correlation:
- `df.corr()` computes Pearson correlation, which measures only *linear* relationships.
- Use Spearman (`method="spearman"`) for monotonic or ranked relationships.
- Correlation does not imply causation.

### 5.6 Using Pandas for EDA
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")
# Summary statistics
df.describe()
# Check missing values
df.isnull().sum()
# Correlation matrix (numeric columns only; required in pandas >= 2.0)
df.corr(numeric_only=True)
# Plot histogram
df["Age"].plot(kind="hist")
plt.show()
# Plot boxplot
df.boxplot(column=["Salary"])
plt.show()
# Plot heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()
```

### 5.7 Key Insights from EDA
- Identify important features.
- Detect outliers (they are handled later, during preprocessing).
- Find correlations between variables.
- Understand the distribution and skewness.
- Get ideas for feature engineering.
- Help in selecting the right algorithm.

### 5.8 Popular Visualization Libraries

| Library | Use |
|---|---|
| Matplotlib | Basic plots (histogram, scatter, etc.) |
| Seaborn | Advanced statistical visualizations |
| Plotly | Interactive visualizations |
| Pandas | Quick and easy plots |
| Missingno | Visualize missing values |

---

## 6. Machine Learning Algorithms – Overview

Machine Learning algorithms are the core models that learn from data and make predictions or decisions. Here are the most important classic algorithms.

### 6.1 Linear Regression
- Used for predicting continuous values.
- Finds the best fit line (y = mx + b).
- Minimizes Mean Squared Error (MSE).
- Example: Predict house price, sales.

*Box:* Equation: $y = mx + b$ (Simple Linear Regression)

### 6.2 Logistic Regression
- Used for classification problems.
- Predicts probability (0 to 1) using the sigmoid function.
- Natively binary; extended to multi-class with softmax (multinomial) or one-vs-rest.
- Example: Spam detection, disease prediction.

*Box:* Sigmoid function: $p = \dfrac{1}{1+e^{-z}}$

### 6.3 Decision Tree
- Tree-based model for classification & regression.
- Uses feature conditions (if-else rules).
- Easy to interpret and visualize.
- Example: Loan approval, customer churn.

*Diagram:* Income > 50k? → Yes: Buy (Yes) / No: Buy (No)

### 6.4 Random Forest
- Ensemble of multiple decision trees.
- Improves accuracy and reduces overfitting.
- Works for classification and regression.
- Example: Disease prediction, fraud detection.

*Diagram:* Tree 1, Tree 2, Tree 3 → Majority Vote

### 6.5 Support Vector Machine (SVM)
- Finds the best hyperplane to separate classes.
- Works well in high-dimensional space.
- Uses kernel trick for non-linear data.
- Example: Text classification. It was historically used for image classification on hand-crafted features; today images are handled mainly by deep learning (CNNs, vision transformers).

*Diagram:* two classes (dots / crosses) separated by a dashed line

### 6.6 Naive Bayes
- Based on Bayes theorem and the assumption that features are *conditionally independent given the class*.
- Works well for text and categorical data.
- Fast and scalable.
- Example: Spam filter, document classification.

*Box:* $P(C|X) = \dfrac{P(X|C)\,P(C)}{P(X)}$

### 6.7 K-Nearest Neighbors (KNN)
- Simple instance-based learning algorithm.
- Predicts based on k nearest data points.
- Works for classification and regression.
- Example: Handwritten digit recognition.

*Diagram:* unknown point "?" among two classes, K = 3

---

## 7. Common Machine Learning Algorithms by Learning Type

Machine Learning algorithms are of different types and are used to solve different kinds of problems. The choice of algorithm depends on the type of data, the problem statement and the expected output.

### 7.1 Types of Machine Learning Algorithms
*Diagram:* Machine Learning Algorithms →
- Supervised Learning → Labeled data (classification, regression)
- Unsupervised Learning → Unlabeled data (clustering, dimensionality reduction)
- Semi-Supervised Learning → Labeled + unlabeled data
- Self-Supervised Learning → Labels generated from the data itself (pre-training)
- Reinforcement Learning → Learn from interaction with environment

### 7.2 Supervised Learning Algorithms
- Use labeled data (input + output).
- Divided into Classification and Regression.
- Learn a mapping from input to output.

| Algorithm | Use Case |
|---|---|
| Linear Regression | Predict continuous values |
| Logistic Regression | Binary/multi-class classification |
| Decision Tree | Classification and regression |
| Random Forest | Reduces the variance (overfitting) of single decision trees by averaging many decorrelated trees |
| Gradient Boosting (XGBoost, LightGBM, CatBoost) | Builds trees sequentially, each correcting the previous errors; often the strongest choice for tabular data |
| Support Vector Machine (SVM) | Classification with optimal (maximum-margin) boundary |
| K-Nearest Neighbors (KNN) | Classification and regression |
| Naive Bayes | Text classification, spam detection |

### 7.3 Unsupervised Learning Algorithms
- Use unlabeled data (only input).
- Find patterns and structure in data.

| Algorithm | Use Case |
|---|---|
| K-Means Clustering | Group similar data points |
| Hierarchical Clustering | Builds a hierarchy of clusters |
| DBSCAN | Finds clusters with noise |
| PCA (Principal Component Analysis) | Dimensionality reduction |
| t-SNE / UMAP | Visualize high-dimensional data in 2D/3D (visualization only: they do not preserve global distances, so they are not general-purpose feature reduction) |

### 7.4 Semi-Supervised Learning
- Uses a small amount of labeled data and a large amount of unlabeled data.
- Useful when labeling data is expensive.

| Algorithm | Use Case |
|---|---|
| Self-Training | Uses confident model predictions as labels |
| Label Propagation | Spreads labels to similar data points |
| Semi-Supervised SVM | Classification with limited labels |

### 7.5 Reinforcement Learning
- Learns through interaction with the environment.
- Receives a reward or penalty for actions.
- Goal is to maximize cumulative reward.

| Algorithm | Use Case |
|---|---|
| Q-Learning | Value-based: learns the action-value function Q(s, a); the policy is to pick the action with the highest Q |
| Deep Q-Network (DQN) | Q-Learning with a neural network approximating Q |
| Policy Gradient | Policy-based: learns the policy directly |
| Actor-Critic | Combines value-based (critic) and policy-based (actor) methods |

### 7.6 How to Choose the Right Algorithm
- If the output is categorical → use Classification algorithms.
- If the output is continuous → use Regression algorithms.
- If the data is unlabeled → use Clustering or Dimensionality Reduction.
- If you have sequential decision making → use Reinforcement Learning.
- Consider dataset size, number of features, interpretability and performance.
- Start with a simple baseline (linear/logistic regression), then try tree ensembles.

*Diagram:* Problem Type →
- Classification (Categorical): Logistic Regression, SVM, Decision Tree, Random Forest, Gradient Boosting
- Regression (Continuous): Linear Regression, Decision Tree, Random Forest, Gradient Boosting

---

## 8. Regression Algorithms

Regression is a supervised learning task used to predict continuous numerical values. It finds the relationship between independent variables (features) and a dependent variable (target) by learning from training data.

### 8.1 Types of Regression Algorithms
- Linear Regression
- Polynomial Regression
- Ridge Regression
- Lasso Regression
- Elastic Net Regression
- Support Vector Regression (SVR)
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting Regression (XGBoost, LightGBM)

### 8.2 Linear Regression
- Assumes a linear relationship between input features and target.
- Simple and easy to implement.
- Works well when the relationship is linear.
- Model equation: $y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n$
- $\beta_0$: Intercept; $\beta_1, \beta_2, \dots$: Coefficients

### 8.3 Example: Simple Linear Regression
Predict house price based on area.

| Area (sq ft) | Price (in Lakhs) |
|---|---|
| 500 | 20 |
| 1000 | 35 |
| 1500 | 50 |
| 2000 | 70 |

*Chart:* Price vs Area (sq ft) with fitted line – "Linear Regression (House Price Prediction)"

Python implementation:
```python
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[500], [1000], [1500], [2000]])
y = np.array([20, 35, 50, 70])
model = LinearRegression()
model.fit(X, y)
print("Coefficient:", model.coef_)      # ≈ [0.033]  (lakhs per sq ft)
print("Intercept:", model.intercept_)   # ≈ 2.5
```

### 8.4 Multiple Linear Regression
- Uses multiple independent variables.
- Model equation: $y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n$
- You need clearly more rows than parameters. Strongly correlated features (multicollinearity) make the coefficients unstable and hard to interpret.

Example: predict house price using area, bedrooms and age (a small illustrative sample; real models need far more data).

| Area | Bedrooms | Age | Price |
|---|---|---|---|
| 800 | 2 | 15 | 30 |
| 1000 | 2 | 5 | 40 |
| 1200 | 3 | 20 | 42 |
| 1500 | 3 | 3 | 55 |
| 1800 | 4 | 10 | 62 |
| 2000 | 4 | 2 | 70 |

### 8.5 Polynomial Regression
- Used when the relationship is non-linear.
- Transforms input features using polynomial terms.
- Example: $y = \beta_0 + \beta_1x + \beta_2x^2 + \beta_3x^3$
- It is still *linear regression*: the model is linear in the parameters β, applied to transformed features.
- Low degree may underfit; high degree quickly overfits. Choose the degree with cross-validation.

*Charts:* Linear Regression (Underfitting) vs Polynomial Regression (Better Fit)

### 8.6 Regularization in Regression
- Helps to prevent overfitting by adding a penalty term on coefficient size.
- Types:
  - Ridge Regression (L2 Regularization): shrinks coefficients towards zero.
  - Lasso Regression (L1 Regularization): can set coefficients to exactly zero, so it also performs feature selection.
  - Elastic Net (combination of L1 and L2).

### 8.7 Evaluation Metrics

| Metric | Formula | Use Case |
|---|---|---|
| MAE | $\frac{1}{n}\sum \lvert y-\hat{y}\rvert$ | Average error; less sensitive to outliers than MSE |
| MSE | $\frac{1}{n}\sum (y-\hat{y})^2$ | Penalizes large errors |
| RMSE | $\sqrt{\frac{1}{n}\sum (y-\hat{y})^2}$ | More interpretable (same unit as target) |
| R² Score | $1 - \frac{SS_{res}}{SS_{tot}}$ | Share of variance explained. At most 1 (perfect fit) and 0 for a model that always predicts the mean; it can be **negative** on test data when the model is worse than that. |

---

## 9. Classification Algorithms

Classification is a supervised learning task used to predict a categorical output (class label) based on input features. It learns a decision boundary from training data to classify new, unseen data into predefined classes.

### 9.1 Common Classification Algorithms
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Gradient Boosting (XGBoost, LightGBM, CatBoost)
- Neural Network (for complex problems)

### 9.2 Logistic Regression
- Used for binary classification; extended to multi-class with softmax or one-vs-rest.
- Predicts probability using the sigmoid function.
- Model equation: $P(y=1) = \dfrac{1}{1+e^{-(\beta_0+\beta_1x_1+\dots+\beta_nx_n)}}$
- Default rule: $P(y=1) \ge 0.5$ → Class 1, otherwise Class 0.
- The 0.5 threshold is a default, not a rule; tune it to the relative cost of false positives and false negatives (e.g., fraud, medical diagnosis).

### 9.3 Decision Tree
- Splits the data based on feature values.
- Forms a tree-like structure.
- Easy to interpret and visualize.
- Handles numerical and categorical data in principle; scikit-learn's implementation generally expects categorical features to be encoded as numbers.

*Diagram:* Income > 50K? → Yes: Credit Score > 700? (Yes → Approve, No → Reject) | No: Employment Stable? (Yes → Approve, No → Reject)

### 9.4 Random Forest
- Ensemble of multiple decision trees.
- Uses bagging (bootstrap aggregating).
- Reduces overfitting and improves accuracy.
- Final prediction is based on voting (for classification).

*Diagram:* Training Data → Tree 1 (Prediction: 1), Tree 2 (Prediction: 0), Tree 3 (Prediction: 1) → Majority Voting → Final Prediction: 1

### 9.5 Support Vector Machine (SVM)
- Finds the best hyperplane to separate classes.
- Works well in high dimensional space.
- Useful for both linear and non-linear data (using the kernel trick).

*Diagram:* two classes, optimal hyperplane, margin

### 9.6 K-Nearest Neighbors (KNN)
- Classifies based on the majority class of the K nearest points.
- Simple and easy to implement.
- Sensitive to the value of K and to feature scaling.

*Diagram:* Class 0, Class 1, new point (star), K nearest neighbours circle, K = 5

### 9.7 Naive Bayes
- Based on Bayes Theorem.
- Assumes features are conditionally independent given the class (the "naive" assumption).
- Works well for text classification (spam detection).

$P(A|B) = \dfrac{P(B|A)\,P(A)}{P(B)}$

### 9.8 Model Evaluation for Classification

| Metric | Formula / Description | Use Case |
|---|---|---|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | Overall performance on *balanced* data; misleading on imbalanced data (99% negatives → "always negative" scores 99%) |
| Precision | TP / (TP + FP) | When false positives are costly |
| Recall | TP / (TP + FN) | When false negatives are costly |
| F1-Score | 2 × (Precision × Recall) / (Precision + Recall) | Balances precision and recall |
| ROC-AUC | Area under the TPR vs FPR curve across all thresholds | Threshold-independent ranking quality |
| PR-AUC | Area under the precision–recall curve | Preferred for strongly imbalanced data |
| Confusion Matrix | TP, TN, FP, FN | Detailed evaluation |

---

## 10. Clustering Algorithms

Clustering is an unsupervised learning technique used to group similar data points together based on their features. It finds hidden patterns in data without using labeled outputs.

### 10.1 Common Clustering Algorithms
- K-Means Clustering
- Hierarchical Clustering (Agglomerative and Divisive)
- DBSCAN (Density Based)
- Gaussian Mixture Model (GMM)
- Mean Shift
- Spectral Clustering

### 10.2 K-Means Clustering
- Divides data into K clusters.
- Each point is assigned to the nearest centroid.
- Iteratively updates centroids.
- Goal: Minimize within-cluster sum of squares (WCSS).
- Steps: Choose K → Initialize Centroids (k-means++) → Assign Points → Update Centroids → Repeat until convergence

*Charts:* Before Clustering (unlabelled points) → After K-Means (K=3)

### 10.3 Hierarchical Clustering
- Builds a hierarchy of clusters (dendrogram).
- Two approaches:
  - Agglomerative (Bottom-Up): start with single points and merge clusters.
  - Divisive (Top-Down): start with all points and split clusters.
- Distance metrics: Euclidean, Manhattan, Cosine, etc.
- Linkage criterion (how distance between clusters is measured) often matters more than the metric:
  - Single: closest points.
  - Complete: farthest points.
  - Average: mean distance.
  - Ward: smallest increase in variance.

*Chart:* dendrogram (Distance vs Data Points)

### 10.4 DBSCAN (Density Based Clustering)
- Groups points that are closely packed together.
- Can find clusters of arbitrary shape.
- Handles noise (outliers) well.
- Does not require the number of clusters.
- Parameters:
  - eps: Maximum distance between two points to be considered neighbors.
  - minPts (`min_samples` in scikit-learn): Minimum number of points within eps (including the point itself) for a point to be a *core point*. Clusters grow from core points; points that are neither core nor reachable from one are noise.

*Chart:* Cluster 1, Cluster 2, Cluster 3, × Noise (Outlier)

### 10.5 Gaussian Mixture Model (GMM)
- Assumes data is generated from a mixture of multiple Gaussian distributions.
- Each cluster is a Gaussian with its own mean and covariance.
- Uses the Expectation-Maximization (EM) algorithm.
- Soft clustering (probability of belonging to each cluster).
- Useful when clusters are overlapping or elliptical.
- Requires the number of components (like K in K-Means).

*Chart:* Gaussian 1, 2, 3 ellipses

### 10.6 Choosing the Right Clustering Algorithm

| Algorithm | Best Use Case | Advantages | Limitations |
|---|---|---|---|
| K-Means | Spherical, similar-sized clusters | Simple, fast | Sensitive to outliers, requires K |
| Hierarchical | Small datasets | No need to specify K up front; dendrogram shows structure | Computationally expensive |
| DBSCAN | Arbitrary shapes, noise | Handles outliers; no K needed | Struggles with varying density |
| GMM | Overlapping clusters | Probabilistic (soft) clustering | Assumes Gaussian distribution; requires number of components |

### 10.7 Evaluating Clustering and Choosing K
- Silhouette Score: measures how similar a point is to its own cluster compared to other clusters (range: −1 to 1; higher is better).
- Davies-Bouldin Index: lower value is better.
- Calinski-Harabasz Index: higher value is better.
- Elbow Method: a visual heuristic, not a metric. Plot WCSS against K and look for the "elbow" where improvement slows; the elbow is often ambiguous, so combine it with the silhouette score.

---

## 11. Linear Regression

### 11.1 What is Linear Regression?
- Linear Regression is a supervised learning algorithm used for predicting a continuous numerical value.
- It models the relationship between input features (X) and a continuous target variable (y) using a linear equation.
- It assumes a linear relationship between the independent and dependent variables.
- It is one of the simplest and most widely used regression algorithms.

### 11.2 Linear Regression Model
The equation of a simple linear regression model is: $y = mx + b$
- y: Predicted output (dependent variable)
- x: Input feature (independent variable)
- m: Slope (coefficient)
- b: Intercept (bias)

*Chart:* data points with regression line

### 11.3 Assumptions
- Linearity: the relationship between features and target is linear.
- Independence: errors are independent of each other.
- Homoscedasticity: errors have constant variance.
- Normality of residuals: needed for confidence intervals and significance tests.
- No strong multicollinearity between features.

### 11.4 Cost Function (Mean Squared Error)
Linear Regression finds the values of m and b that minimize the Mean Squared Error (MSE).

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

n: number of training examples; $y_i$: actual value; $\hat{y}_i$: predicted value; $(y_i-\hat{y}_i)^2$: squared error

### 11.5 Parameter Estimation
Parameters can be learned using:
- Normal Equation (analytical solution): $\beta = (X^TX)^{-1}X^Ty$. It is exact, but the matrix inversion costs about $O(d^3)$, so it becomes impractical with very many features.
- Gradient Descent (iterative optimization): scales to large datasets and many features; benefits from feature scaling.

### 11.6 Types of Linear Regression

| Type | Description | Use Case |
|---|---|---|
| Simple Linear Regression | One independent variable | Predict house price based on area |
| Multiple Linear Regression | Multiple independent variables | Predict salary based on experience, skills, etc. |

### 11.7 Evaluation Metrics
- Mean Squared Error (MSE): average squared difference.
- Root Mean Squared Error (RMSE): √MSE (same unit as the target, easier to interpret).
- Mean Absolute Error (MAE): average of absolute errors.
- R² Score: share of variance explained. At most 1, 0 for a model that predicts the mean, and negative on test data if the model is worse than that.

---

## 12. Logistic Regression

### 12.1 What is Logistic Regression?
- Logistic Regression is a supervised learning algorithm used for classification problems.
- It predicts the probability that an input belongs to a particular class (usually binary classification).
- It uses the logistic (sigmoid) function to map any real-valued number to a value between 0 and 1.
- It is used as a classification algorithm. Statistically it is a regression model of the **log-odds** (a generalized linear model), which is where the name comes from.

### 12.2 Logistic Function (Sigmoid)
Logistic Regression uses the sigmoid function to convert the linear equation output into a probability.

$$\sigma(z) = \frac{1}{1+e^{-z}}$$

- z = wx + b (linear combination)
- Output is always strictly between 0 and 1.

*Chart:* S-shaped sigmoid curve, z from −10 to 10, crossing 0.5 at z = 0

### 12.3 Hypothesis Function
$$h(x) = \sigma(wx+b) = \frac{1}{1+e^{-(wx+b)}}$$
h(x) gives the probability that the input belongs to class 1.

### 12.4 Decision Boundary
A threshold (0.5 by default) decides the class:
- h(x) ≥ 0.5 → Class 1
- h(x) < 0.5 → Class 0

The decision boundary (wx + b = 0) is **linear** in feature space. The threshold can be tuned to trade precision against recall.

*Chart:* Class 0 (○), Class 1 (△), dashed linear decision boundary

### 12.5 Cost Function (Binary Cross Entropy)
Logistic Regression uses Binary Cross Entropy (Log Loss) as the cost function.

$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}\left[y_i\log(h(x_i)) + (1-y_i)\log(1-h(x_i))\right]$$

m: number of training examples; $y_i$: actual label (0 or 1); $h(x_i)$: predicted probability; θ: model parameters (w, b)

In practice a regularization term is added (scikit-learn uses L2 by default; strength controlled by `C`).

### 12.6 Advantages and Disadvantages

| Advantages | Disadvantages |
|---|---|
| Simple and easy to implement | Assumes a linear relationship between features and log-odds |
| Works well for binary classification | Performance may be poor with non-linear data (unless features are engineered) |
| Outputs probabilities, usually well calibrated | Sensitive to outliers |
| Coefficients are interpretable | Not suitable for very complex datasets |
| Less prone to overfitting compared to complex models | |

---

## 13. Decision Tree

### 13.1 What is a Decision Tree?
- A Decision Tree is a supervised learning algorithm used for both classification and regression.
- It splits the data into subsets based on the value of input features.
- It works by asking a series of questions on features, creating a tree-like structure.
- It is easy to understand, interpret and visualize.
- It can handle both numerical and categorical data in principle; scikit-learn's implementation generally requires categorical features to be encoded as numbers.

### 13.2 Structure of a Decision Tree
- Root Node: represents the entire dataset.
- Internal Node: decision based on a feature.
- Branch: outcome of a decision (feature value).
- Leaf Node: final prediction (class label or value).

*Diagram:* Root Node → Internal Nodes → Leaf Nodes

### 13.3 How a Decision Tree Works
1. Select the best feature to split the data.
2. Split the dataset into subsets based on the feature value.
3. Repeat the process recursively for each subset.
4. Stop when a stopping condition is met (e.g., max depth, minimum samples, or pure node).
5. Make predictions using the leaf node.

*Diagram:* Outlook → Sunny: Humidity (High → No, Normal → Yes) | Overcast → Yes | Rain: Wind (Weak → Yes, Strong → No)

### 13.4 Feature Selection Criteria
- For Classification: Information Gain (Entropy), Gain Ratio, or Gini Index.
- For Regression: Mean Squared Error (MSE) or Mean Absolute Error (MAE).

### 13.5 Entropy and Information Gain (Classification)
- Entropy measures the impurity in a dataset.
- It is 0 when all samples belong to the same class.

$$Entropy(S) = -\sum_{i=1}^{n} p_i \log_2(p_i)$$
($p_i$: proportion of class i; n: number of classes)

- Information Gain measures the reduction in entropy after a split.

$$IG(S,A) = Entropy(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|}\,Entropy(S_v)$$
(S: original dataset; A: feature; $S_v$: subset for value v)

- Information Gain favors features with many distinct values. **Gain Ratio** (used by C4.5) corrects this by dividing IG by the split's own entropy.

### 13.6 Gini Index (Alternative to Entropy)
- Gini Index also measures impurity in a dataset.
- It is slightly faster to compute (no logarithm). In practice Gini and entropy produce very similar trees.

$$Gini(S) = 1 - \sum_{i=1}^{n} p_i^2$$
($p_i$: proportion of class i; n: number of classes)

### 13.7 Pruning
- Pre-pruning (early stopping): limit `max_depth`, `min_samples_split`, `min_samples_leaf`.
- Post-pruning: grow the full tree, then remove branches that do not improve validation performance (cost-complexity pruning, `ccp_alpha` in scikit-learn).

### 13.8 Advantages and Disadvantages

| Advantages | Disadvantages |
|---|---|
| Easy to understand and interpret | Prone to overfitting (especially with deep trees) |
| Works for both classification and regression | Can be unstable (small change in data can change the tree) |
| Can handle numerical and categorical data | May not perform well on very complex datasets |
| No need for feature scaling | Information-gain splitting is biased towards features with many categories |
| Can capture non-linear relationships | Usually lower accuracy than ensemble methods (Random Forest, Gradient Boosting) |

---

## 14. Random Forest

### 14.1 What is Random Forest?
- Random Forest is an ensemble learning algorithm that uses multiple decision trees to solve classification and regression problems.
- It builds many decision trees on different subsets of data and features and combines their predictions to get a more accurate and stable result.
- It reduces overfitting (variance) compared to a single decision tree.
- It works well with both numerical and categorical data.
- It is easy to implement and gives good performance in most real-world problems.

### 14.2 How Random Forest Works
1. Take multiple random samples (with replacement) from the original training data (bootstrap sampling).
2. For each sample, build a decision tree.
3. At each split, consider only a random subset of features instead of all features.
4. Train all trees independently.
5. For prediction, combine the results of all trees: voting for classification, averaging for regression.

*Diagram:* Training Data → Tree 1 … Tree N → Prediction Class A / Class B / Class A → Final Prediction (Majority Voting)

### 14.3 Prediction
- Classification (textbook / hard voting): majority vote, $\hat{y} = \text{mode}\{h_1(x), h_2(x), \dots, h_n(x)\}$.
  scikit-learn uses *soft voting*: it averages the predicted class probabilities of all trees and picks the class with the highest average.
- Regression: average of all trees' predictions, $\hat{y} = \frac{1}{n}\sum_{i=1}^{n} h_i(x)$.

### 14.4 Out-of-Bag (OOB) Error
Each tree is trained on a bootstrap sample, which leaves out roughly 37% of the rows. Evaluating each row only with the trees that did not see it gives a free validation estimate (`oob_score=True` in scikit-learn).

### 14.5 Key Parameters

| Parameter | Description | Default (sklearn) |
|---|---|---|
| n_estimators | Number of trees in the forest | 100 |
| max_depth | Maximum depth of each tree | None |
| max_features | Number of features to consider at each split | "sqrt" (classifier), 1.0 = all features (regressor) |
| min_samples_split | Minimum samples required to split a node | 2 |
| min_samples_leaf | Minimum samples required at a leaf node | 1 |
| bootstrap | Whether to use bootstrap sampling | True |

### 14.6 Advantages and Disadvantages

| Advantages | Disadvantages |
|---|---|
| High accuracy and less overfitting | Computationally expensive (many trees) |
| Works well with both numerical and categorical data | Less interpretable than a single decision tree |
| Handles large datasets | Requires more memory |
| Robust to noise and to outliers in the features (outliers in the target still affect regression) | Training time can be high for large datasets |
| Provides feature importance (the default impurity-based importance is biased towards high-cardinality features; prefer permutation importance or SHAP) | Not ideal for real-time prediction with very limited resources |
| Easy to implement and parallelize | Cannot extrapolate: regression predictions never go beyond the target range seen in training |

### 14.7 Use Cases
- Classification: Spam detection, disease prediction, customer churn, fraud detection.
- Regression: House price prediction, sales forecasting, demand prediction. For data with trends, handle the trend separately (e.g., detrend or use lag features), because the forest cannot extrapolate.

---

## 15. Support Vector Machine (SVM)

### 15.1 What is SVM?
- Support Vector Machine (SVM) is a supervised learning algorithm used for classification and regression (Support Vector Regression, SVR).
- It finds the optimal hyperplane that separates the classes with maximum margin.
- It works well in high dimensional spaces, including cases where the number of features is greater than the number of samples.
- It can be used for both linear and non-linear problems using kernel functions.

### 15.2 Key Concepts
- Hyperplane: a decision boundary that separates classes.
- Margin: distance between the hyperplane and the nearest data points of each class.
- Support Vectors: the data points closest to the hyperplane. Only support vectors affect the position of the hyperplane.
- Kernel Trick: handles non-linearly separable data.
- Soft Margin: allows some misclassification, controlled by a penalty (C).
- A larger margin generally leads to better generalization.

*Diagram:* maximum margin (dashed lines), Class A (○), Class B (△), circled support vectors

### 15.3 How SVM Works
1. Find the hyperplane with the maximum margin between classes (allowing some violations with a soft margin).
2. The support vectors define the hyperplane.
3. For non-linear data, use a kernel, which implicitly works in a higher-dimensional space.
4. Classify new data by which side of the hyperplane it falls on.

### 15.4 Mathematical Formulation (Linear SVM)
The hyperplane is: $w \cdot x + b = 0$
- w: weight vector (normal to the hyperplane)
- x: input feature vector
- b: bias term

**Hard margin** (perfectly separable data): maximizing the margin is equivalent to
minimizing $\frac{1}{2}\lVert w\rVert^2$ subject to $y_i(w \cdot x_i + b) \ge 1$ for all training samples.

**Soft margin** (real data): add slack variables $\xi_i$:
minimize $\frac{1}{2}\lVert w\rVert^2 + C\sum_i \xi_i$ subject to $y_i(w \cdot x_i + b) \ge 1 - \xi_i,\ \xi_i \ge 0$.

- Large C: narrow margin, fewer training errors, risk of overfitting.
- Small C: wider margin, more regularization, more training errors allowed.

### 15.5 Kernel Trick (Non-Linear SVM)
When data is not linearly separable, SVM uses a kernel function that computes inner products as if the data were mapped into a higher-dimensional space, without explicitly transforming the data. For the RBF kernel that space is infinite-dimensional.

| Kernel | Formula (scikit-learn form; r = `coef0`) | Use Case |
|---|---|---|
| Linear | $K(x,x') = x \cdot x'$ | Linearly or nearly separable data, high-dimensional sparse data such as text |
| Polynomial | $K(x,x') = (\gamma\, x \cdot x' + r)^d$ | Non-linear data (polynomial interactions) |
| RBF (Gaussian) | $K(x,x') = \exp(-\gamma \lVert x-x'\rVert^2)$ | Most common; works well in many cases |
| Sigmoid | $K(x,x') = \tanh(\gamma\, x \cdot x' + r)$ | Neural-network-like decision functions |

### 15.6 Types of SVM
- Linear SVM: for linearly or nearly linearly separable data (with a soft margin).
- Non-Linear SVM: uses the kernel trick for complex data.
- Support Vector Regression (SVR): for regression problems.

### 15.7 Key Parameters

| Parameter | Description | Default (sklearn `SVC`) |
|---|---|---|
| C | Regularization parameter (trade-off between margin width and misclassification) | 1.0 |
| kernel | Type of kernel function | 'rbf' |
| gamma | Kernel coefficient (for rbf, poly, sigmoid) | 'scale' |
| degree | Degree of the polynomial kernel | 3 |
| coef0 | Independent term r in poly and sigmoid kernels | 0.0 |

### 15.8 Advantages and Disadvantages

| Advantages | Disadvantages |
|---|---|
| Works well in high dimensional spaces | Kernel SVM training scales roughly O(n²)–O(n³): expensive for large datasets (use `LinearSVC` or SGD for large data) |
| Effective in both linear and non-linear cases | Requires feature scaling |
| Uses only support vectors (memory-efficient model) | Sensitive to the choice of kernel and parameters |
| Robust to overfitting with proper regularization | No native probability outputs (`probability=True` adds costly internal calibration) |
| | Less interpretable than decision trees |

### 15.9 Use Cases
- Classification: text categorization, spam detection, disease prediction, image classification on hand-crafted features.
- Regression: house price prediction, demand forecasting (using SVR).

---

## 16. K-Nearest Neighbors (KNN)

### 16.1 What is KNN?
- K-Nearest Neighbors (KNN) is a simple, instance-based machine learning algorithm used for classification and regression.
- It classifies a new data point based on the majority class of its k nearest neighbors.
- It is a non-parametric, lazy learning algorithm: there is no real model fitting (training only stores the data, possibly in a search index such as a KD-tree or Ball-tree), so the computational cost moves to prediction time.
- It works based on the distance between data points in the feature space.

### 16.2 How KNN Works
1. Scale the features (standardization or min-max), because distances are dominated by features with large ranges.
2. Choose the value of K (number of neighbors).
3. Calculate the distance between the new data point and the training data points.
4. Select the K nearest neighbors.
5. For classification: choose the class with the majority vote.
6. For regression: take the average of the target values of the K neighbors.

*Diagram:* new point, K = 5 circle, Class A (○) and Class B (△)

### 16.3 Distance Metrics
- Euclidean Distance: most commonly used.
- Manhattan Distance: less dominated by single large differences. It can discriminate slightly better than Euclidean as the number of dimensions grows, but no metric escapes the curse of dimensionality.
- Minkowski Distance: a generalization with parameter p. p = 1 gives Manhattan, p = 2 gives Euclidean, p → ∞ gives Chebyshev. (scikit-learn's default is Minkowski with p = 2.)

| Euclidean | Manhattan | Minkowski |
|---|---|---|
| $d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}$ | $d(x,y)=\sum_{i=1}^{n}\lvert x_i-y_i\rvert$ | $d(x,y)=\left(\sum_{i=1}^{n}\lvert x_i-y_i\rvert^p\right)^{1/p}$ |

### 16.4 Choosing the Value of K
- Small value of K → model is sensitive to noise (high variance).
- Large value of K → model is smoother but may miss small patterns (high bias).
- Typically, K is chosen using cross-validation.
- Use an odd value of K for binary classification to avoid ties.

### 16.5 Advantages and Disadvantages

| Advantages | Disadvantages |
|---|---|
| Simple and easy to understand | Slow and memory-heavy at prediction time for large datasets |
| No model fitting required | Performance degrades with high dimensional data (curse of dimensionality) |
| Works well for small datasets | Sensitive to irrelevant features and requires feature scaling |
| Can be used for both classification and regression | Requires choosing the value of K |

---

## 17. K-Means Clustering

### 17.1 What is K-Means Clustering?
- K-Means is an unsupervised machine learning algorithm used for clustering.
- It groups similar data points into K different clusters based on feature similarity.
- It aims to minimize the distance between data points and their respective cluster centroids.
- Each data point belongs to the cluster with the nearest centroid.
- It is simple, fast and widely used for real-world problems.

### 17.2 How K-Means Works
1. Scale the features (K-Means uses Euclidean distance).
2. Choose the number of clusters (K).
3. Initialize K centroids using **k-means++** (the default), which spreads the initial centroids apart. Plain random initialization is also possible but less reliable.
4. Assign each data point to the nearest centroid.
5. Recalculate each centroid as the mean of all points in its cluster.
6. Repeat steps 4 and 5 until the centroids no longer change or a maximum number of iterations is reached.

K-Means converges to a **local** minimum of the objective, not necessarily the global one. That is why it is run several times with different initializations (`n_init`) and the best result is kept.

*Flowchart:* Start (K clusters) → Initialize centroids (k-means++) → Assign points to nearest centroid → Recalculate centroids → Converged? — No → back to Assign; Yes → Final Clusters

### 17.3 Objective Function
K-Means minimizes the Within-Cluster Sum of Squares (WCSS): the squared distance between data points and their cluster centroid.

$$J = \sum_{k=1}^{K}\sum_{x \in C_k}\lVert x - \mu_k\rVert^2$$

K: number of clusters; $C_k$: set of points in cluster k; x: data point; $\mu_k$: centroid of cluster k; $\lVert x-\mu_k\rVert^2$: squared Euclidean distance

### 17.4 Distance Metric
K-Means uses Euclidean distance between a point x and a centroid μ, summed over the n features (index j):

$$d(x,\mu) = \sqrt{\sum_{j=1}^{n}(x_j-\mu_j)^2}$$

*Chart:* Cluster 1 (○), Cluster 2 (△), Cluster 3 (□)

### 17.5 Key Parameters

| Parameter | Description | Default (sklearn) |
|---|---|---|
| n_clusters | Number of clusters (K) | 8 |
| init | Method for centroid initialization ('k-means++', 'random') | 'k-means++' |
| n_init | Number of runs with different initial centroids (best kept) | 'auto' |
| max_iter | Maximum number of iterations | 300 |
| random_state | Controls randomness for reproducibility | None |

### 17.6 Advantages and Disadvantages

| Advantages | Disadvantages |
|---|---|
| Simple and easy to implement | Need to specify the number of clusters (K) |
| Works well on large datasets | Sensitive to initial centroid selection (mitigated by k-means++ and n_init) |
| Computationally efficient | Does not work well with non-spherical clusters |
| Good for spherical and well-separated clusters | Assumes clusters of similar size and variance |
| | Sensitive to outliers and to feature scaling |

### 17.7 Use Cases
- Customer segmentation, image compression (color quantization), document clustering, market analysis.
- Anomaly detection (as a preprocessing step), grouping similar products.

---

## 18. Principal Component Analysis (PCA)

### 18.1 What is PCA?
- Principal Component Analysis (PCA) is an unsupervised learning technique used for dimensionality reduction.
- It transforms the original correlated features into a new set of uncorrelated features called principal components.
- The principal components are ordered by the amount of variance they capture: the first captures the most.
- It reduces dimensionality while retaining as much variance as possible.
- PCA is widely used for visualization, noise reduction and as a preprocessing step for other models.

### 18.2 Why Use PCA?
- Reduces the number of features (dimensionality).
- Removes correlation between features.
- Helps in visualizing high-dimensional data (e.g., in 2D or 3D).
- Reduces noise. It *can* improve model performance, but because PCA ignores the target, the high-variance directions are not necessarily the most predictive ones, so it can also hurt. Validate the effect.
- Speeds up training time by reducing computational complexity.

*Flowchart:* Original Data (X: n × d) → Center (and optionally standardize) Data → Compute Covariance Matrix → Find Eigenvalues and Eigenvectors → Select Top k Components → Transform Data (X_new: n × k)

### 18.3 How PCA Works
1. Center the data (subtract the mean of each feature). If features have different units or scales, also standardize them (unit variance).
2. Compute the covariance matrix.
3. Calculate eigenvalues and eigenvectors.
4. Sort eigenvectors by eigenvalues (descending order).
5. Select the top k eigenvectors (principal components).
6. Project the data onto the new feature space.

In practice, libraries such as scikit-learn compute PCA via **Singular Value Decomposition (SVD)** of the centered data, which gives the same components without explicitly forming the covariance matrix. scikit-learn's `PCA` only centers the data; for standardization add a `StandardScaler` before it.

### 18.4 Mathematical Formulation
Given a dataset X (n samples, d features):
1. Standardize the data: $Z = \dfrac{X - \mu}{\sigma}$ (or only center: $Z = X - \mu$)
2. Covariance matrix: $C = \dfrac{1}{n-1} Z^T Z$
3. Eigen decomposition: $C v = \lambda v$
4. Transform data: $X_{new} = Z \cdot W_k$

λ: eigenvalue (variance along that component); v: eigenvector (principal component direction); $W_k$: matrix of the top k eigenvectors

### 18.5 Choosing the Number of Components
- Explained variance ratio of component i: $\lambda_i / \sum_j \lambda_j$.
- Keep enough components to explain a target share of variance (e.g., 95%). In scikit-learn, `PCA(n_components=0.95)` does this directly.
- Or use a scree plot (variance per component) and look for the point where it levels off.

### 18.6 Example (2D to 1D Projection)
*Charts:* Original Data (2D, points along a diagonal) → PCA (2D to 1D) → Data after PCA (1D, points on Principal Component 1)

The new axis (Principal Component 1) captures the maximum variance in the data.

### 18.7 Key Parameters (sklearn.decomposition.PCA)

| Parameter | Description | Default |
|---|---|---|
| n_components | Number of components to keep (int), or share of variance to keep (float between 0 and 1) | None (keep all) |
| whiten | Scale each component to unit variance (the components are already uncorrelated) | False |
| svd_solver | SVD solver to use ('auto', 'full', 'arpack', 'randomized', and 'covariance_eigh' in newer versions) | 'auto' |
| random_state | Controls randomness for reproducibility (randomized/arpack solvers) | None |

### 18.8 Advantages and Disadvantages

| Advantages | Disadvantages |
|---|---|
| Reduces dimensionality and computation time | Components can be hard to interpret |
| Removes feature correlation | Loses some information |
| Helps in visualization | Captures only linear structure |
| Can improve model performance | Sensitive to feature scaling |
| Reduces noise in the data | Ignores the target, so it may discard predictive low-variance directions |

For non-linear structure use **Kernel PCA** or autoencoders for feature reduction, and **t-SNE / UMAP** for visualization.

### 18.9 Use Cases
- Image compression, data visualization, gene expression analysis, customer segmentation (as preprocessing), noise reduction.
