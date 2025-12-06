# Lead Scoring Homework - Bank Marketing Dataset

This project contains the complete solution for the Lead Scoring homework assignment using the Bank Marketing dataset.

## 📋 Overview

The goal of this homework is to build a classification model to predict whether a client will sign up to the platform (`converted` variable) using various features from the Bank Marketing dataset.

## 🗂️ Dataset

**Dataset Name:** `course_lead_scoring.csv`

**Download:** 
```bash
wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv
```

**Target Variable:** `converted` - indicates whether the client signed up to the platform

## 🔧 Requirements

### Python Libraries
- pandas
- numpy
- scikit-learn

### Installation
```bash
pip install pandas numpy scikit-learn
```

## 📊 Assignment Questions

### Question 1: Mode of Industry Column
Find the most frequent observation (mode) for the column `industry`.

**Options:**
- NA
- technology
- healthcare
- retail

### Question 2: Correlation Analysis
Create a correlation matrix for numerical features and identify the two features with the biggest correlation.

**Pairs to Consider:**
- interaction_count and lead_score
- number_of_courses_viewed and lead_score
- number_of_courses_viewed and interaction_count
- annual_income and interaction_count

### Question 3: Mutual Information Score
Calculate mutual information score between `converted` and categorical variables using the training set.

**Variables:**
- industry
- location
- lead_source
- employment_status

### Question 4: Logistic Regression
Train a logistic regression model with one-hot encoding and calculate validation accuracy.

**Model Parameters:**
```python
LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
```

**Expected Accuracy Options:**
- 0.64
- 0.74
- 0.84
- 0.94

### Question 5: Feature Elimination
Identify the least useful feature using feature elimination technique.

**Features to Test:**
- industry
- employment_status
- lead_score

### Question 6: Regularization
Test different values of regularization parameter C and find the best value.

**C Values:** [0.01, 0.1, 1, 10, 100]

## 🚀 Usage

1. **Download the dataset:**
   ```bash
   wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv
   ```

2. **Run the Python script or Jupyter notebook:**
   ```bash
   python lead_scoring_homework.py
   ```
   or open the notebook in Jupyter:
   ```bash
   jupyter notebook lead_scoring_homework.ipynb
   ```

3. **View the results:**
   The script will output answers for all 6 questions and provide a summary at the end.

## 📝 Data Preparation

The solution includes proper data preparation:

1. **Missing Value Handling:**
   - Categorical features: Replace with 'NA'
   - Numerical features: Replace with 0.0

2. **Data Splitting:**
   - Training: 60%
   - Validation: 20%
   - Test: 20%
   - Random seed: 42

3. **Feature Engineering:**
   - One-hot encoding for categorical variables using DictVectorizer

## 🔍 Solution Approach

### Step 1: Data Loading and Preparation
- Load dataset
- Identify categorical and numerical columns
- Handle missing values appropriately

### Step 2: Exploratory Data Analysis
- Calculate mode for categorical features
- Create correlation matrix for numerical features
- Analyze feature relationships

### Step 3: Data Splitting
- Split data into train/validation/test sets (60/20/20)
- Ensure reproducibility with random_state=42

### Step 4: Feature Analysis
- Calculate mutual information scores for categorical features
- Identify most informative features

### Step 5: Model Training
- Implement one-hot encoding
- Train logistic regression model
- Evaluate on validation set

### Step 6: Feature Selection
- Test feature importance using elimination technique
- Identify least useful features

### Step 7: Hyperparameter Tuning
- Test different regularization strengths (C values)
- Select optimal model



## 📁 Project Structure

```
.
├── README.md
├── course_lead_scoring.csv          # Dataset
├── lead_scoring_homework.py         # Python script
└── lead_scoring_homework.ipynb      # Jupyter notebook (optional)
```

## 🎯 Key Features

- ✅ Clean, well-commented code
- ✅ Proper handling of missing values
- ✅ Reproducible results (seed=42)
- ✅ Efficient one-hot encoding
- ✅ Comprehensive feature analysis
- ✅ Model performance evaluation
- ✅ Hyperparameter optimization

## 📚 Learning Objectives

This homework covers:
- Data preprocessing and cleaning
- Exploratory data analysis
- Feature engineering and encoding
- Train/validation/test splitting
- Logistic regression modeling
- Feature selection techniques
- Model regularization
- Performance evaluation

## 🤝 Contributing

This is a homework assignment solution. Feel free to use it as a reference for learning purposes.

## 📄 License

This project is for educational purposes.

## 👤 Author

Created as part of a machine learning course assignment.

---

**Note:** Make sure the dataset file `course_lead_scoring.csv` is in the same directory as your script before running.