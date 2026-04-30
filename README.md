# Applied Machine Learning Projects

A collection of three end-to-end machine learning projects covering the core paradigms of supervised and unsupervised learning — regression, classification, and clustering.

---

## Projects Overview

| # | Project | Type | Dataset | Best Model |
|---|---------|------|---------|------------|
| 1 | Titanic Survival Prediction | Classification | Titanic (seaborn) | Decision Tree (tuned) |
| 2 | California Housing Prices | Regression | California Housing CSV | Gradient Boosting (tuned) |
| 3 | Mall Customer Segmentation | Unsupervised / Clustering | Mall Customers CSV | K-Means (K=5) + PCA |
| 4 | MNIST Handwritten Digit Classification | Classification | MNIST | - |
| 5 | Fraud Detection | Classification | Creditcard | Random Forest|
---

## 1. Titanic Survival Prediction

**Notebook:** `Titanic-Classification.ipynb`

### Goal
Predict whether a passenger survived the Titanic disaster using passenger features. Since predicting a survivor as dead is a more costly error than the reverse, **recall** was prioritized as the primary evaluation metric over accuracy.

### Key Steps
- **EDA:** Analyzed survival rates across sex, passenger class, fare, and embarkation port. Sex was the strongest predictor — 74% of women survived vs. 18% of men. Survival rates showed a clear class gradient: 62% (1st), 47% (2nd), 24% (3rd).
- **Feature Engineering:**
  - Dropped redundant columns (`alive`, `class`, `who`, `adult_male`, `deck`, etc.)
  - Imputed missing `age` with mean, `embarked` with mode
  - One-hot encoded `sex` and `embarked`
  - Engineered `family_size` (sibsp + parch + 1) and `log_fare` (to reduce skew)
- **Modeling:** Compared Logistic Regression, Decision Tree, and Random Forest using 5-fold cross-validation
- **Tuning:** Applied `GridSearchCV` on Decision Tree (`max_depth`, `min_samples_split`, `min_samples_leaf`)

### Results

| Model | Train Acc | Test Acc | ROC-AUC |
|-------|-----------|----------|---------|
| Logistic Regression | 0.795 | 0.810 | 0.791 |
| Random Forest | 0.983 | 0.810 | 0.791 |
| Decision Tree (original) | 0.983 | 0.810 | 0.802 |
| Decision Tree (tuned) | 0.837 | 0.788 | — |

The tuned Decision Tree reduced the train/test accuracy gap from **0.173 → 0.041**, resolving overfitting while maintaining the lowest false negative count (16 missed survivors).

### Libraries
`pandas` · `numpy` · `seaborn` · `matplotlib` · `scikit-learn`

---

## 2. California Housing Prices

**Notebook:** `Housing_Prices-Regression.ipynb`

### Goal
Predict median house values across California census blocks using demographic and geographic features.

### Key Steps
- **EDA:** Identified that `median_house_value` is capped at $500,001 — the spike at the right of the histogram is artificial, meaning the model cannot learn what drives luxury pricing
- **Feature Engineering:**
  - Imputed missing `total_bedrooms` with median
  - Engineered `rooms_per_household`, `bedrooms_per_household`, and `household_size`
  - One-hot encoded `ocean_proximity`
  - Dropped raw aggregate columns (`total_rooms`, `total_bedrooms`, `population`, `households`, `longitude`, `latitude`)
- **Modeling:** Compared Linear Regression, Decision Tree Regressor, and Gradient Boosting Regressor using 5-fold cross-validation (RMSE scoring)
- **Tuning:** Applied `GridSearchCV` on Gradient Boosting (`n_estimators`, `learning_rate`, `max_depth`, `subsample`)

### Results

| Model | R² | MAE |
|-------|-----------|-----|
| Linear Regression | 0.493 | ~$54,471 |
| Decision Tree | 0.479 | ~$56,811 |
| Gradient Boosting (original) | 0.708 | ~$43,464 |
| **Gradient Boosting (tuned)** | **0.725** | **~$41,686** |

The model performs well in the $100k–$350k range where data is most concentrated. Performance degrades at the upper end due to the $500,001 dataset cap.

### Potential Improvements
- Remove or model the capped values separately
- Add geographic clustering features from latitude/longitude
- Try XGBoost or LightGBM

### Libraries
`pandas` · `numpy` · `matplotlib` · `scikit-learn`

---

## 3. Mall Customer Segmentation

**Notebook:** `Mall_Customers-Unsupervised.ipynb`

### Goal
Group mall customers into meaningful segments based on annual income and spending behavior with no labels (unsupervised learning).

### Key Steps
- **Feature Selection:** Used `Annual Income (k$)` and `Spending Score (1-100)` as primary features; standardized with `StandardScaler` to prevent income from dominating distance calculations
- **Finding K:** Used the **Elbow Method** (inertia/WCSS) and **Silhouette Score** across K=2–10 to confirm K=5 as the optimal number of clusters
- **Cluster Profiling:** Computed mean Age, Income, and Spending Score per cluster to label each segment

### Cluster Labels (K=5)

| Cluster | Income | Spending | Label |
|---------|--------|----------|-------|
| 0 | High | Low | **Frugal Elite** — have money but don't spend it |
| 1 | Low | Low | **Budget Conscious** — low income, low spending |
| 2 | Medium | Medium | **Averages** — balanced profile |
| 3 | High | High | **VIPs** — best customers, retain them |
| 4 | Low | High | **High-Potential Youth** — spend despite low income |

- **PCA Extension:** Added `Age` as a third feature and applied PCA to reduce to 2 components for visualization. PC1 and PC2 together captured the majority of variance, with PCA-based K-Means clustering revealing similar groupings in the compressed space.

### Libraries
`pandas` · `numpy` · `matplotlib` · `scikit-learn` (KMeans, StandardScaler, PCA, silhouette_score)

---

## 4. MNIST Handwritten Digit Classification
**Type:** Neural Network — Multiclass Classification  
**Dataset:** MNIST (70,000 images, 28×28 pixels)  
**Goal:** Classify handwritten digits 0-9  

**Architecture:** Flatten → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)  
**Result:** 97.74% test accuracy  

**Key skills:** Neural network architecture, forward propagation,
activation functions, overfitting detection, confusion matrix interpretation  

### Libraries
`pandas` · `numpy` · `matplotlib` · `tenserflow` · `seaborn`

---

## Repository Structure

```
├── Titanic-Classification.ipynb
├── Housing_Prices-Regression.ipynb
├── Mall_Customers-Unsupervised.ipynb
├── MNIST.ipynb
├── data/
│   ├── housing.csv
│   └── Mall_Customers.csv
└── README.md
```

---

## How to Run

1. Clone the repository
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn tensorflow`
3. Open any notebook with Jupyter and run cells in order


## 5. Credit Card Fraud Detection
**Type:** Supervised Learning — Imbalanced Binary Classification  
**Dataset:** Credit Card Fraud Detection (Kaggle) — 284,807 transactions  
**Link:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  

> ⚠️ Dataset not included in this repo due to file size.
> Download from the Kaggle link above and place in a `/data` folder.

**Goal:** Detect fraudulent transactions where only 0.17% of data is fraud

**Pipeline:** Stratified Split → StandardScaler → SMOTE → Random Forest → Threshold Tuning  
**Best result:** ROC-AUC = 0.964, Recall = 89% at threshold 0.3  

**Key skills:** Class imbalance handling, SMOTE, precision-recall tradeoff,
threshold tuning, ROC-AUC evaluation.
