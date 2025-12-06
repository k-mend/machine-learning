# Car Fuel Efficiency Prediction - Linear Regression Assignment

## 📋 Project Overview

This project implements linear regression models to predict car fuel efficiency (MPG - Miles Per Gallon) using various vehicle characteristics. The assignment explores different aspects of linear regression including handling missing data, regularization, and model stability analysis.

## 🎯 Objectives

- Build a regression model to predict fuel efficiency
- Handle missing values using different strategies
- Implement linear regression with and without regularization
- Analyze model performance and stability across different data splits
- Evaluate the impact of regularization parameters

## 📊 Dataset

**Source:** [Car Fuel Efficiency Dataset](https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv)

**Features Used:**
- `engine_displacement` - Engine displacement in cubic inches
- `horsepower` - Engine horsepower
- `vehicle_weight` - Vehicle weight in pounds
- `model_year` - Year the car was manufactured
- `fuel_efficiency_mpg` - Target variable (Miles Per Gallon)

**Dataset Characteristics:**
- Contains missing values in one feature
- Requires preprocessing and train/validation/test splitting

## 🔧 Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **matplotlib** - Data visualization
- **seaborn** - Statistical visualizations

## 📁 Project Structure

```
.
├── README.md
├── car_fuel_efficiency_assignment.py    # Main solution script
├── requirements.txt                      # Python dependencies
└── results/
    ├── correlation_matrix.png
    ├── predictions_vs_actual.png
    └── residual_analysis.png
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

### Download the Dataset

```bash
wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv
```

### Run the Analysis

```bash
python car_fuel_efficiency_assignment.py
```

## 📝 Assignment Questions & Solutions

### EDA
**Question:** Does `fuel_efficiency_mpg` have a long tail?
- Analyzed the distribution using histogram and boxplot
- **Answer:** Yes, the distribution shows a right-skewed pattern with a long tail

### Question 1: Missing Values
**Question:** Which column has missing values?
- **Answer:** `horsepower`

### Question 2: Median Horsepower
**Question:** What's the median (50% percentile) for variable 'horsepower'?
- **Options:** 49, 99, 149, 199
- **Answer:** 99

### Question 3: Handling Missing Values
**Question:** Fill missing values with 0 vs mean - which gives better RMSE?
- Filled missing values with 0: RMSE = X.XX
- Filled missing values with mean: RMSE = X.XX
- **Answer:** [Fill with 0 / Fill with mean / Both equally good]

### Question 4: Regularization
**Question:** Which regularization parameter `r` gives the best RMSE?
- Tested values: [0, 0.01, 0.1, 1, 5, 10, 100]
- **Answer:** 0

### Question 5: Model Stability
**Question:** What's the standard deviation of RMSE across different seeds?
- Tested seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- **Options:** 0.001, 0.006, 0.060, 0.600
- **Answer:** [Your answer]

### Question 6: Final Model Performance
**Question:** What's the RMSE on the test dataset with r=0.001?
- Split with seed 9
- Combined train and validation sets
- **Options:** 0.15, 0.515, 5.15, 51.5
- **Answer:** 0.52 (closest to 0.515)

## 🔍 Methodology

### 1. Data Preparation
- Selected relevant features
- Split data into train (60%), validation (20%), and test (20%) sets
- Used seed 42 for reproducibility

### 2. Linear Regression Implementation

**Without Regularization:**
```python
w = (X^T X)^(-1) X^T y
```

**With Regularization (Ridge):**
```python
w = (X^T X + r*I)^(-1) X^T y
```

### 3. Model Evaluation
- **RMSE (Root Mean Squared Error):** Primary metric for model evaluation
- **R² Score:** Coefficient of determination
- **Residual Analysis:** Check for patterns in prediction errors

## 📈 Key Findings

1. **Missing Data:** Horsepower has missing values that need to be handled
2. **Regularization:** Model performs best with minimal or no regularization (r=0)
3. **Stability:** The model shows [low/moderate/high] variance across different data splits (std = X.XXX)
4. **Performance:** Final test RMSE of ~0.52 indicates reasonable prediction accuracy

## 🎓 Learning Outcomes

- Implemented linear regression from scratch using matrix operations
- Understood the impact of missing value imputation strategies
- Explored the bias-variance tradeoff through regularization
- Analyzed model stability and reproducibility

## 📊 Visualizations

The project generates three main visualizations:

1. **Correlation Matrix:** Shows relationships between features
2. **Predictions vs Actual:** Compares model predictions to true values
3. **Residual Analysis:** Examines prediction errors and their distribution

## 🤝 Contributing

This is an assignment project. If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## 📄 License

This project is part of a machine learning course assignment and is available for educational purposes.

## 👤 Author

[Your Name]
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## 🙏 Acknowledgments

- Dataset provided by [Alexey Grigorev](https://github.com/alexeygrigorev)
- Assignment from Machine Learning Zoomcamp / ML Course

## 📚 References

- [Linear Regression Documentation](https://scikit-learn.org/stable/modules/linear_model.html)
- [Ridge Regression](https://en.wikipedia.org/wiki/Ridge_regression)
- [RMSE Explanation](https://en.wikipedia.org/wiki/Root-mean-square_deviation)

---

**Note:** Replace placeholder values (X.XX, [Your answer]) with actual results from running the analysis.