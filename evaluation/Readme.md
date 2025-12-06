# Lead Scoring Classification - Machine Learning Assignment

## Overview
This assignment focuses on building and evaluating a binary classification model to predict whether a lead (potential customer) will convert or not. We use logistic regression with various evaluation metrics to assess model performance.

## Dataset
The dataset is the **Lead Scoring Dataset** containing information about potential customers and whether they converted to actual customers.

**Source:** [course_lead_scoring.csv](https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv)

**Target Variable:** `converted` - indicates if the client signed up to the platform (1) or not (0)

## Data Preparation

### Missing Value Handling
- **Categorical features:** Missing values replaced with 'NA'
- **Numerical features:** Missing values replaced with 0.0

### Data Splitting
The dataset was split into three parts using `train_test_split` from scikit-learn:
- **Training set:** 60%
- **Validation set:** 20%
- **Test set:** 20%
- **Random state:** 1 (for reproducibility)

## Libraries and Tools Used

### Core Libraries
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computations
- `matplotlib` - Data visualization

### Scikit-Learn Components
- `sklearn.model_selection`:
  - `train_test_split` - Splitting dataset into train/validation/test sets
  - `KFold` - K-fold cross-validation
- `sklearn.feature_extraction`:
  - `DictVectorizer` - One-hot encoding for categorical features
- `sklearn.linear_model`:
  - `LogisticRegression` - Binary classification model
- `sklearn.metrics`:
  - `roc_auc_score` - ROC AUC evaluation metric
  - `precision_recall_curve` - Precision and recall calculations

## Methodology

### 1. Feature Importance Analysis 
Used ROC AUC as a metric to evaluate the predictive power of individual numerical features. For each numerical variable, calculated AUC using the feature as a score and the target variable as ground truth. If AUC < 0.5, the feature was inverted (negated) to flip negative correlation to positive.

### 2. Model Training 
- Applied **one-hot encoding** using `DictVectorizer` to convert categorical variables into numerical format
- Trained **Logistic Regression** model with parameters:
  - `solver='liblinear'`
  - `C=1.0`
  - `max_iter=1000`
- Evaluated model performance on validation set using ROC AUC score

### 3. Precision-Recall Analysis 
- Computed precision and recall at various decision thresholds (0.0 to 1.0 with 0.01 step)
- Visualized precision-recall curves
- Identified the threshold where precision and recall curves intersect

### 4. F1 Score Optimization 
- Calculated F1 score for all thresholds using the formula: F1 = 2 × (P × R) / (P + R)
- Identified the threshold that maximizes F1 score
- F1 score balances precision and recall, providing a single metric for model evaluation

### 5. Cross-Validation 
- Implemented **5-Fold Cross-Validation** using `KFold` with:
  - `n_splits=5`
  - `shuffle=True`
  - `random_state=1`
- Trained logistic regression on each fold
- Evaluated model stability by calculating standard deviation of AUC scores across folds

### 6. Hyperparameter Tuning 
- Performed grid search over regularization parameter `C` values: [0.000001, 0.001, 1]
- Used 5-fold cross-validation for each C value
- Selected best C based on:
  1. Highest mean AUC score
  2. Lowest standard deviation (in case of ties)
  3. Smallest C value (if still tied)

## Key Concepts

### One-Hot Encoding
Transformed categorical variables into binary columns (0/1) for each category. This allows the logistic regression model to work with categorical data.

### ROC AUC Score
Area Under the Receiver Operating Characteristic curve - measures model's ability to distinguish between classes. Values range from 0 to 1, where 0.5 is random guessing and 1.0 is perfect classification.

### Precision and Recall
- **Precision:** Of all positive predictions, how many were correct? (TP / (TP + FP))
- **Recall:** Of all actual positives, how many did we catch? (TP / (TP + FN))

### Cross-Validation
Technique to assess model performance and stability by training on different subsets of data. Helps detect overfitting and provides more reliable performance estimates.

### Regularization Parameter (C)
Controls the trade-off between fitting the training data and keeping the model simple. Smaller C values mean stronger regularization (simpler models).

## Results
All metrics are computed and displayed at the end of the script execution, including:
- Best numerical feature by AUC
- Model validation AUC
- Precision-Recall intersection threshold
- Optimal F1 threshold
- Cross-validation standard deviation
- Best regularization parameter C

## Running the Project
Simply execute the Python script - it will automatically download the dataset, perform all analyses, and output the answers to each question.