import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the data
file_path = 'combined_typing_data.csv'
data = pd.read_csv(file_path)

# Select the features and target
features = [
    'accel_change_x', 'accel_change_y', 'accel_change_z', 'gyro_change_x', 
    'gyro_change_y', 'gyro_change_z', 'accel_magnitude', 'gyro_magnitude', 
    'accel_mean_x', 'accel_mean_y', 'accel_mean_z', 'gyro_mean_x', 'gyro_mean_y', 
    'gyro_mean_z', 'accel_std_x', 'accel_std_y', 'accel_std_z', 'gyro_std_x', 
    'gyro_std_y', 'gyro_std_z', 'accel_energy', 'gyro_energy'
]

# Ensure 'keystrokes' column exists in data as target variable
if 'keystrokes' not in data.columns:
    data['keystrokes'] = 464  # Placeholder if 'keystrokes' doesn't exist

# Prepare feature (X) and target (y) data
X = data[features]
y = data['keystrokes']

# Preprocess the data: scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set up K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define regressors to compare
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Support Vector Regressor": SVR(kernel='linear'),
    "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5)
}

# Evaluate each model
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    mse_scores = []
    mae_scores = []
    
    for train_index, val_index in kf.split(X_scaled):
        X_train, X_val = X_scaled[train_index], X_scaled[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict and calculate metrics
        y_pred = model.predict(X_val)
        mse_scores.append(mean_squared_error(y_val, y_pred))
        mae_scores.append(mean_absolute_error(y_val, y_pred))
    
    # Calculate and print the average scores for each metric
    avg_mse = np.mean(mse_scores)
    avg_mae = np.mean(mae_scores)
    print(f"{name} - Mean MSE: {avg_mse:.4f}, Mean MAE: {avg_mae:.4f}")
