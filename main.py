# PHASE 1: IMPORTS
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import joblib

# PHASE 2: DATA LOADING
df = pd.read_csv("kc_house_data.csv")
print("✅ Data Loaded Successfully")
print(df.shape)
print(df.head())

# PHASE 3: DATA CLEANING AND EXPLORATION
print("\n🔍 Checking for Missing Values:")
print(df.isnull().sum())

print("\n📊 Basic Statistics:")
print(df.describe())

# Drop unnecessary columns
df = df.drop(['id', 'date'], axis=1)

# Check correlation heatmap (optional)
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap")
plt.show()

# PHASE 4: FEATURE ENGINEERING
# Handle missing values if any (none expected in kc_house_data)
df = df.dropna()

# Define features (X) and target (y)
X = df.drop('price', axis=1)
y = df['price']

# Scale features that are numeric
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PHASE 5: DATA SPLITTING
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# PHASE 6: MODEL TRAINING AND EVALUATION
# Use a baseline model (Linear Regression)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("\n📈 Linear Regression Performance:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_lr):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.2f}")
print(f"R²: {r2_score(y_test, y_pred_lr):.2f}")

# Try a more advanced model (Random Forest)
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n🌲 Random Forest Performance:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_rf):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}")
print(f"R²: {r2_score(y_test, y_pred_rf):.2f}")

# PHASE 7: HYPERPARAMETER TUNING AND IMPROVEMENT
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
}
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("\n🎯 Best Parameters from Grid Search:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

print("\n🏆 Tuned Random Forest Performance:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_best):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_best)):.2f}")
print(f"R²: {r2_score(y_test, y_pred_best):.2f}")

# PHASE 8: VISUALIZATION AND REPORTING
# Plot actual vs predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

# Feature importance
importances = pd.Series(best_model.feature_importances_, index=df.drop('price', axis=1).columns)
importances.sort_values(ascending=False).head(10).plot(kind='bar', figsize=(8,5))
plt.title("Top 10 Important Features")
plt.show()

# Save model and scaler
joblib.dump(rf_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and Scaler saved successfully.")