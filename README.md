# Credit Risk Analysis & Machine Learning

## Overview

This project performs **Exploratory Data Analysis (EDA)**, **data preprocessing**, **feature engineering**, and **machine learning model development** on the German Credit dataset.

The objective is to predict whether a customer represents a **Good Credit Risk** or **Bad Credit Risk** based on demographic and financial information.

The project includes:

* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Label encoding
* Model training and hyperparameter tuning
* Model comparison
* Model serialization for deployment

---

# Dataset

Dataset: `german_credit_data.csv`

The dataset contains customer information including:

| Feature          | Description                              |
| ---------------- | ---------------------------------------- |
| Age              | Customer age                             |
| Sex              | Gender                                   |
| Job              | Employment category                      |
| Housing          | Housing status                           |
| Saving accounts  | Savings account category                 |
| Checking account | Checking account category                |
| Credit amount    | Requested credit amount                  |
| Duration         | Loan duration in months                  |
| Purpose          | Purpose of the loan                      |
| Risk             | Target variable (Good / Bad Credit Risk) |

---

# Project Workflow

## 1. Data Loading

```python
df = pd.read_csv("german_credit_data.csv")
```

Initial inspection:

* Head
* Shape
* Data types
* Missing values
* Duplicate values
* Descriptive statistics

---

## 2. Data Cleaning

### Missing Values

Rows containing missing values are removed:

```python
df = df.dropna().reset_index(drop=True)
```

### Remove Unnecessary Columns

```python
df.drop(columns="Unnamed: 0", inplace=True)
```

---

## 3. Exploratory Data Analysis (EDA)

### Numerical Feature Analysis

Features analyzed:

* Age
* Credit Amount
* Duration

Visualizations:

* Histograms
* Boxplots
* Boxenplots

Example:

```python
df[["Age","Credit amount","Duration"]].hist()
```

---

### Categorical Feature Analysis

Features:

* Sex
* Job
* Housing
* Saving Accounts
* Checking Accounts
* Purpose

Visualizations:

```python
sns.countplot()
```

Used to understand customer distributions across categories.

---

### Correlation Analysis

Correlation heatmap generated using:

```python
sns.heatmap()
```

Features:

* Age
* Job
* Credit Amount
* Duration

Purpose:

* Detect relationships
* Identify feature importance trends

---

### Risk Analysis

Comparison of customer behavior based on:

```python
Risk = Good / Bad
```

Visualizations:

* Count plots
* Boxen plots
* Grouped statistics

Metrics analyzed:

* Average Age
* Average Credit Amount
* Average Duration

---

## 4. Feature Engineering

Selected features:

```python
features = [
    "Age",
    "Sex",
    "Job",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Credit amount",
    "Duration"
]
```

Target:

```python
target = "Risk"
```

---

## 5. Encoding

### Categorical Encoding

Label Encoding applied to:

```python
Sex
Job
Housing
Saving accounts
Checking account
```

Each encoder is saved for future inference:

```python
joblib.dump(le, f"{col}_encoder.pkl")
```

Saved files:

```text
Sex_encoder.pkl
Job_encoder.pkl
Housing_encoder.pkl
Saving accounts_encoder.pkl
Checking account_encoder.pkl
```

---

### Target Encoding

```python
Risk
```

Converted into numeric labels.

Saved as:

```text
target_encoder.pkl
```

---

## 6. Train/Test Split

```python
train_test_split(
    x,
    y,
    test_size=0.20,
    stratify=y,
    random_state=1
)
```

Split:

* 80% Training
* 20% Testing

---

# Machine Learning Models

The following models are trained and compared.

---

## Decision Tree Classifier

### Hyperparameters

```python
max_depth
min_samples_split
min_samples_leaf
```

Optimization:

```python
GridSearchCV
```

---

## Random Forest Classifier

### Hyperparameters

```python
n_estimators
max_depth
min_samples_split
min_samples_leaf
```

Optimization:

```python
GridSearchCV
```

---

## Extra Trees Classifier

### Hyperparameters

```python
n_estimators
max_depth
min_samples_split
min_samples_leaf
```

Optimization:

```python
GridSearchCV
```

---

## XGBoost Classifier

### Hyperparameters

```python
n_estimators
max_depth
learning_rate
subsample
colsample_bytree
```

Class imbalance handled using:

```python
scale_pos_weight
```

Optimization:

```python
GridSearchCV
```

---

# Model Evaluation

Metric used:

```python
Accuracy Score
```

Example:

```python
accuracy_score(y_test, y_pred)
```

Each model is evaluated on the test dataset.

Comparison includes:

* Decision Tree Accuracy
* Random Forest Accuracy
* Extra Trees Accuracy
* XGBoost Accuracy

---

# Best Model

The best-performing model is saved:

```python
joblib.dump(
    best_et,
    "extra_trees_credit_model.pkl"
)
```

Saved model:

```text
extra_trees_credit_model.pkl
```

---

# Generated Files

```text
extra_trees_credit_model.pkl
target_encoder.pkl

Sex_encoder.pkl
Job_encoder.pkl
Housing_encoder.pkl
Saving accounts_encoder.pkl
Checking account_encoder.pkl
```

---

# Visualization Outputs

The project generates:

### Histograms

* Age Distribution
* Credit Amount Distribution
* Duration Distribution

### Boxplots

* Outlier Detection
* Distribution Analysis

### Countplots

* Categorical Variable Distribution
* Risk Distribution

### Scatterplots

* Age vs Credit Amount
* Duration-based sizing

### Violin Plots

* Credit Amount by Savings Category

### Correlation Heatmaps

* Numerical Feature Relationships

---

# Project Structure

```text
German-Credit-Risk-Analysis/
│
├── german_credit_data.csv
├── notebook.ipynb
│
├── extra_trees_credit_model.pkl
├── target_encoder.pkl
│
├── Sex_encoder.pkl
├── Job_encoder.pkl
├── Housing_encoder.pkl
├── Saving accounts_encoder.pkl
├── Checking account_encoder.pkl
│
├── README.md
│
└── requirements.txt
```

---

# Installation

```bash
git clone https://github.com/yourusername/german-credit-risk-analysis.git

cd german-credit-risk-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Requirements

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
joblib
```

Install manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

---

# Future Improvements

* Feature scaling experiments
* SMOTE for class imbalance
* ROC-AUC evaluation
* Precision/Recall metrics
* SHAP explainability
* Model deployment using Flask or FastAPI
* Streamlit dashboard
* Real-time prediction API

---

# Author

**Kirubel Ayele**

Machine Learning • Data Science • AI Engineering

---

# License

This project is released under the MIT License.

---
