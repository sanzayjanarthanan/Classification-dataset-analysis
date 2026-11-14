# Task 36 – Classification Dataset Analysis (Breast Cancer Dataset)

**Veda Technology · AI & ML Track · Level 1 · Day 36**

## Objective
Select a simple classification dataset and identify its features, target variable and possible target classes, to understand the structure of a basic classification problem.

## Tools Used
- Python
- Pandas
- Scikit-learn
- Jupyter Notebook

## Selected Dataset
**Breast Cancer Wisconsin (Diagnostic) dataset**, loaded with `sklearn.datasets.load_breast_cancer`. It is a built-in Scikit-learn dataset, so no download is needed.

| Property | Value |
|----------|-------|
| Samples (rows) | 569 |
| Feature columns | 30 |
| Target column | 1 (`target`) |
| Missing values | 0 |
| Duplicate rows | 0 |
| Feature data type | All float64 (numerical) |

Each row describes a tumour sample using measurements computed from a digitised image of a cell nucleus.

## Files in this Repository
| File | Description |
|------|-------------|
| `classification_dataset_analysis.ipynb` | Notebook with the analysis |
| `breast_cancer.csv` | Dataset saved as CSV |
| `README.md` | Features, target, classes and explanation |

## Code
```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)
df = data.frame
df.to_csv("breast_cancer.csv", index=False)

print(df.shape)
print(data.feature_names)
print(data.target_names)
print(df["target"].value_counts())
```

## 1. Feature Identification
The **features (X)** are the 30 numerical measurements of the cell nucleus. They are the inputs the model uses. They are 10 measurements, each recorded in three forms: **mean**, **standard error** and **worst** (largest value).

The 10 base measurements are: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension.

Examples of feature columns: `mean radius`, `mean texture`, `mean perimeter`, `mean area`, `worst radius`, `worst concavity`.

```python
X = df.drop(columns=["target"])   # 30 feature columns
```

## 2. Target Identification
The **target variable (y)** is the column `target`. It is the value the model should predict: whether a tumour is malignant or benign.

```python
y = df["target"]
```

## 3. Target Classes
The unique values of the target column were checked with `df["target"].value_counts()`.

| Value | Class | Meaning | Count | Share |
|-------|-------|---------|-------|-------|
| 0 | malignant | Cancerous tumour | 212 | 37.3% |
| 1 | benign | Non-cancerous tumour | 357 | 62.7% |

The target is **categorical** (two possible classes), which is why this is a classification problem and not a regression problem.

## 4. Type of Classification Problem
This is a **binary classification** problem, because the target has exactly two classes (malignant and benign). The classes are slightly **imbalanced**: benign is about 63% of the data and malignant about 37%. This is not extreme, but it is worth remembering when evaluating a model later, because accuracy alone can hide mistakes on the smaller class.

## 5. Short Explanation of the Classification Problem
Given the 30 measurements of a cell nucleus, the goal is to predict whether the tumour is malignant or benign. The model learns from rows where the correct answer is already known (supervised learning) and then predicts the class for new samples. Since the output is a category and not a number, it is a classification task.

## Key Learnings
- A classification target is categorical, and its unique values are the classes.
- The number of classes decides the type: two classes is binary, three or more is multiclass.
- Always check the class counts, because imbalance affects how a model should be evaluated.
- The target column must never be included among the features.

## Interview Questions – Key Answers
1. **What is classification?** A supervised learning task where the model predicts a category (class label) for a sample, such as malignant or benign.
2. **What is binary classification?** Classification with exactly two possible classes, for example malignant vs benign or spam vs not spam.
3. **What is multiclass classification?** Classification with three or more possible classes, for example the Iris dataset, where a flower is one of three species.

## How to Run
```bash
pip install pandas scikit-learn jupyter
jupyter notebook classification_dataset_analysis.ipynb
```
