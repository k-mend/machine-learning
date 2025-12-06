import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Download and load the dataset
url = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv'
df = pd.read_csv(url)

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())
print("\nMissing values:")
print(df.isnull().sum())

# Prepare the dataset
# Fill missing values with zeros
df = df.fillna(0)

# Prepare features and target
y = df['fuel_efficiency_mpg'].values
X = df.drop('fuel_efficiency_mpg', axis=1)

# Train/validation/test split with 60%/20%/20% distribution
X_full_train, X_test, y_full_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

X_train, X_val, y_train, y_val = train_test_split(
    X_full_train, y_full_train, test_size=0.25, random_state=1  # 0.25 * 0.8 = 0.2
)

print(f"\nDataset splits:")
print(f"Train: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
print(f"Validation: {len(X_val)} ({len(X_val)/len(df)*100:.1f}%)")
print(f"Test: {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")

# Convert to dictionaries and vectorize
dv = DictVectorizer(sparse=True)

train_dicts = X_train.to_dict(orient='records')
X_train_vec = dv.fit_transform(train_dicts)

val_dicts = X_val.to_dict(orient='records')
X_val_vec = dv.transform(val_dicts)

test_dicts = X_test.to_dict(orient='records')
X_test_vec = dv.transform(test_dicts)

print(f"\nVectorized training matrix shape: {X_train_vec.shape}")

# ============================================================================
# Question 1: Decision Tree with max_depth=1
# ============================================================================
print("\n" + "="*70)
print("QUESTION 1: Decision Tree (max_depth=1)")
print("="*70)

dt = DecisionTreeRegressor(max_depth=1, random_state=1)
dt.fit(X_train_vec, y_train)

# Get the feature used for splitting
feature_names = dv.get_feature_names_out()
tree = dt.tree_
feature_idx = tree.feature[0]  # Root node feature
splitting_feature = feature_names[feature_idx]

print(f"Feature used for splitting: {splitting_feature}")

# ============================================================================
# Question 2: Random Forest with n_estimators=10
# ============================================================================
print("\n" + "="*70)
print("QUESTION 2: Random Forest (n_estimators=10)")
print("="*70)

rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf.fit(X_train_vec, y_train)

y_pred = rf.predict(X_val_vec)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print(f"RMSE on validation data: {rmse:.3f}")

# ============================================================================
# Question 3: Experiment with n_estimators
# ============================================================================
print("\n" + "="*70)
print("QUESTION 3: Experimenting with n_estimators (10 to 200, step 10)")
print("="*70)

n_estimators_range = range(10, 201, 10)
rmse_scores = []

for n_est in n_estimators_range:
    rf = RandomForestRegressor(n_estimators=n_est, random_state=1, n_jobs=-1)
    rf.fit(X_train_vec, y_train)
    y_pred = rf.predict(X_val_vec)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    rmse_scores.append(rmse)
    print(f"n_estimators={n_est:3d}, RMSE={rmse:.3f}")

# Find when RMSE stops improving (3 decimal places)
min_rmse = min(rmse_scores)
min_idx = rmse_scores.index(min_rmse)
best_n_estimators = list(n_estimators_range)[min_idx]

print(f"\nBest n_estimators: {best_n_estimators} with RMSE={min_rmse:.3f}")
print(f"RMSE stops improving after n_estimators={best_n_estimators}")

# ============================================================================
# Question 4: Best max_depth using mean RMSE
# ============================================================================
print("\n" + "="*70)
print("QUESTION 4: Finding best max_depth using mean RMSE")
print("="*70)

max_depth_values = [10, 15, 20, 25]
mean_rmse_by_depth = {}

for max_depth in max_depth_values:
    print(f"\nTesting max_depth={max_depth}:")
    rmse_list = []
    
    for n_est in range(10, 201, 10):
        rf = RandomForestRegressor(
            n_estimators=n_est,
            max_depth=max_depth,
            random_state=1,
            n_jobs=-1
        )
        rf.fit(X_train_vec, y_train)
        y_pred = rf.predict(X_val_vec)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_list.append(rmse)
    
    mean_rmse = np.mean(rmse_list)
    mean_rmse_by_depth[max_depth] = mean_rmse
    print(f"  Mean RMSE: {mean_rmse:.3f}")

best_max_depth = min(mean_rmse_by_depth, key=mean_rmse_by_depth.get)
print(f"\nBest max_depth: {best_max_depth} with mean RMSE={mean_rmse_by_depth[best_max_depth]:.3f}")

# ============================================================================
# Question 5: Feature Importance
# ============================================================================
print("\n" + "="*70)
print("QUESTION 5: Feature Importance")
print("="*70)

rf = RandomForestRegressor(
    n_estimators=10,
    max_depth=20,
    random_state=1,
    n_jobs=-1
)
rf.fit(X_train_vec, y_train)

# Get feature importances
feature_importances = rf.feature_importances_
feature_names = dv.get_feature_names_out()

# Create a dataframe for better visualization
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
print(importance_df.head(10))

# Find the most important among the specified features
target_features = ['vehicle_weight', 'horsepower', 'acceleration', 'engine_displacement']
print(f"\nImportance of specified features:")
for feat in target_features:
    importance = importance_df[importance_df['feature'] == feat]['importance'].values
    if len(importance) > 0:
        print(f"  {feat}: {importance[0]:.6f}")

most_important = importance_df[importance_df['feature'].isin(target_features)].iloc[0]
print(f"\nMost important feature among the 4: {most_important['feature']} ({most_important['importance']:.6f})")

# ============================================================================
# Question 6: XGBoost with different eta values
# ============================================================================
print("\n" + "="*70)
print("QUESTION 6: XGBoost with different eta values")
print("="*70)

try:
    import xgboost as xgb
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train_vec, label=y_train)
    dval = xgb.DMatrix(X_val_vec, label=y_val)
    
    # Create watchlist
    watchlist = [(dtrain, 'train'), (dval, 'val')]
    
    # Test eta=0.3
    print("\nTraining with eta=0.3:")
    xgb_params_03 = {
        'eta': 0.3, 
        'max_depth': 6,
        'min_child_weight': 1,
        'objective': 'reg:squarederror',
        'nthread': 8,
        'seed': 1,
        'verbosity': 1,
    }
    
    model_03 = xgb.train(
        xgb_params_03,
        dtrain,
        num_boost_round=100,
        evals=watchlist,
        verbose_eval=False
    )
    
    y_pred_03 = model_03.predict(dval)
    rmse_03 = np.sqrt(mean_squared_error(y_val, y_pred_03))
    print(f"RMSE with eta=0.3: {rmse_03:.4f}")
    
    # Test eta=0.1
    print("\nTraining with eta=0.1:")
    xgb_params_01 = {
        'eta': 0.1, 
        'max_depth': 6,
        'min_child_weight': 1,
        'objective': 'reg:squarederror',
        'nthread': 8,
        'seed': 1,
        'verbosity': 1,
    }
    
    model_01 = xgb.train(
        xgb_params_01,
        dtrain,
        num_boost_round=100,
        evals=watchlist,
        verbose_eval=False
    )
    
    y_pred_01 = model_01.predict(dval)
    rmse_01 = np.sqrt(mean_squared_error(y_val, y_pred_01))
    print(f"RMSE with eta=0.1: {rmse_01:.4f}")
    
    print(f"\nComparison:")
    print(f"  eta=0.3: RMSE={rmse_03:.4f}")
    print(f"  eta=0.1: RMSE={rmse_01:.4f}")
    
    if rmse_03 < rmse_01:
        print(f"  Winner: eta=0.3")
    elif rmse_01 < rmse_03:
        print(f"  Winner: eta=0.1")
    else:
        print(f"  Both give equal value")
        
except ImportError:
    print("XGBoost is not installed. Please install it with: pip install xgboost")

print("\n" + "="*70)
print("HOMEWORK COMPLETE!")
print("="*70)