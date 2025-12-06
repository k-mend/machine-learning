# Vehicle Fuel Efficiency Prediction

A comprehensive machine learning project that predicts vehicle fuel efficiency using tree-based regression models. This project explores various ensemble methods including Decision Trees, Random Forests, and XGBoost to build accurate predictive models for miles per gallon (MPG) estimation.

## Project Overview

Fuel efficiency is a critical factor in vehicle selection and environmental impact assessment. This project leverages historical vehicle data to build robust regression models that can predict fuel efficiency based on various vehicle characteristics such as weight, horsepower, engine specifications, and more.

## Dataset

The project uses a comprehensive vehicle fuel efficiency dataset containing multiple features:

- **Vehicle characteristics**: Weight, horsepower, acceleration, engine displacement
- **Temporal data**: Model year
- **Categorical features**: Origin, fuel type
- **Target variable**: Fuel efficiency in miles per gallon (MPG)

### Data Acquisition

```bash
wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv
```

## Methodology

### Data Preprocessing

1. **Missing Value Handling**: All missing values are imputed with zeros to ensure model compatibility
2. **Train/Validation/Test Split**: 60%/20%/20% distribution for robust model evaluation
3. **Feature Encoding**: DictVectorizer with sparse matrices for efficient memory usage

### Models Explored

#### 1. Decision Tree Regressor
A baseline model with controlled depth to understand feature importance and prevent overfitting. Initial experiments with shallow trees (`max_depth=1`) help identify the most influential features for fuel efficiency prediction.

#### 2. Random Forest Regressor
An ensemble approach that combines multiple decision trees to improve prediction accuracy and reduce variance. Key experiments include:

- **Model Size Optimization**: Testing various numbers of estimators (10-200 trees) to find the optimal forest size
- **Tree Depth Tuning**: Evaluating different `max_depth` values [10, 15, 20, 25] to balance model complexity and performance
- **Feature Importance Analysis**: Extracting and ranking features by their contribution to prediction accuracy

#### 3. XGBoost
A gradient boosting framework that builds trees sequentially, with each tree learning from the errors of previous ones. The project explores:

- **Learning Rate Optimization**: Comparing different `eta` values (0.3 vs 0.1) to find the optimal balance between training speed and model accuracy
- **Advanced Parameters**: Fine-tuning `max_depth`, `min_child_weight`, and other hyperparameters

## Key Findings

### Feature Importance
The analysis reveals which vehicle characteristics are most predictive of fuel efficiency. Understanding these relationships helps in:
- Vehicle design optimization
- Consumer decision-making
- Environmental impact assessment

### Model Performance
Performance is evaluated using Root Mean Squared Error (RMSE) on the validation set, allowing for:
- Fair comparison across different algorithms
- Identification of optimal hyperparameters
- Assessment of model generalization capability

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost
```

## Usage

Run the complete analysis pipeline:

```python
python fuel_efficiency_analysis.py
```

The script will:
1. Download and prepare the dataset
2. Train multiple models with various configurations
3. Evaluate performance metrics
4. Display feature importance rankings
5. Generate comprehensive results for each modeling approach

## Results Structure

The analysis produces detailed outputs for each experiment:

- **Decision Tree Analysis**: Identifies primary splitting features
- **Random Forest Tuning**: RMSE scores across different configurations
- **Hyperparameter Optimization**: Mean RMSE comparisons for model selection
- **Feature Rankings**: Importance scores for interpretability
- **XGBoost Comparison**: Performance metrics for different learning rates

## Project Insights

This project demonstrates:
- The power of ensemble methods for regression tasks
- The importance of systematic hyperparameter tuning
- Trade-offs between model complexity and performance
- Feature engineering and selection for automotive data
- Best practices in model evaluation and validation

## Future Enhancements

Potential areas for extension:
- Deep learning approaches for non-linear relationships
- Additional feature engineering from raw specifications
- Time-series analysis for temporal trends in fuel efficiency
- Cross-validation for more robust performance estimates
- Production deployment with model serving infrastructure

## License

This project is open-source and available for educational and research purposes.

## Acknowledgments

Dataset provided by the ML engineering community for educational purposes in predictive modeling and regression analysis.